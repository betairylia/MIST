#import libraries
import argparse
import sys

import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

import contextlib

from scipy.sparse import csr_matrix, triu
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix
from scipy.sparse import identity
from sklearn.neighbors import NearestNeighbors

import os
import time

from munkres import Munkres

from datasets import *
from utils import *
from SiameseLoss import *
from MIST_utils import *
from tau_optim import *

def main():

    parser = argparse.ArgumentParser(description='SIM experiments')

    # Set-up
    parser.add_argument('--name', type=str, default='Untitled')
    parser.add_argument('--dataset', type=str, default='mnist')

    # General HP
    parser.add_argument('--batchsize', type=int, default=250, help='batchsize')
    parser.add_argument('--epochs', type=int, default=50, help='epoch')
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate')

    # HP
    parser.add_argument('--alpha_vat', type=float, default=0.25, help='alpha_vat')
    parser.add_argument('--alpha', type=int, default=1, choices=[0,1,2], help='alpha')
    parser.add_argument('--INCEvar', type=int, default=0, choices=[0,1,2], help='INCEvar')
    parser.add_argument('--beta', type=float, default=0.0, help='beta')
    parser.add_argument('--K', type=int, default=200, help='K (maximum)')
    parser.add_argument('--K0', type=int, default=5, help='K0 (SIM)')
    parser.add_argument('--K_vat', type=int, default=10, help='K (VAT)')
    parser.add_argument('--xi', type=float, default=10.0, help='xi (VAT)')
    parser.add_argument('--mu', type=float, default=0.045, help='mu')
    parser.add_argument('--gamma', type=float, default=1.5, help='gamma')
    parser.add_argument('--eta', type=float, default=5.0, help='eta (Marginal entropy)')
    # parser.add_argument('--eta2', type=float, default=1.0, help='eta2 (Positive pairs)')
    parser.add_argument('--tau', type=float, default=0.05, help='tau')
    parser.add_argument('--optimize_tau', type=int, default=0, help='0 = Fix tau; 1 = Optimize tau')
    # parser.add_argument('--tau_neg', type=float, default=0.01, help='tau_negative') # tau_neg = tau_pos 
    parser.add_argument('--rate', type=float, default=1.0, help='Negative sampling rate (0~1)')
    parser.add_argument('--aug_func', type=str, default='kNNG', help='Augmentation function (not used now)')
    
    # Pre-defined pairs
    parser.add_argument('--K0BetaPair', type=int, default=-1, choices=[-1,0,1], help='K0BetaPair')
    parser.add_argument('--MuEtaGamma', type=int, default=-1, choices=[-1,0,1,2,3], help='MuEtaGamma Pair')

    # Worker meta-data
    parser.add_argument('--runid', type=int, default=0, help='Run ID.')
    parser.add_argument('--keyargs', type=str, default="", help='Key variables in HP tune, splitted in commas')

    parser.add_argument('--geodesic', type=int, default=0, help='Use geodesic distance based augmentation or not')

    args = parser.parse_args()

    # Log into WandB
    wandb.init(project = 'sim', config = vars(args), group = GetArgsStr(args))

    # InceVariants = [SiameseLoss_UB, SiameseLoss_Exact, SiameseLoss_Symmetrized]
    # SiameseLoss = InceVariants[args.INCEvar]

    K0_beta_pairs = {
        "mnist":        [(5, 0), (7, 0)],
        "stl":          [(5, 0), (7, 0)],
        "omniglot":     [(5, 0), (7, 0)],
        "cifar100":     [(7, 0), (10,0)],
        "svhn":         [(7, 0), (10,0)],
        "cifar10":      [(10,0), (15,0)],
        "reuters10k":   [(50, 2.0 / 3.0), (50, 4.0 / 5.0)],
        "20news":       [(200, 2.0 / 3.0), (200, 4.0 / 5.0)],
    }

    Mu_Eta_Gamma_pairs = [(0.045, 6.0, 1.5), (0.05, 5.0, 1.5), (0.04, 6.0, 1.5), (0.045, 5.0, 1.5)]

    if args.K0BetaPair >= 0:
        args.K0 = K0_beta_pairs[args.dataset][args.K0BetaPair][0]
        args.beta = K0_beta_pairs[args.dataset][args.K0BetaPair][1]

    if args.MuEtaGamma >= 0:
        args.mu = Mu_Eta_Gamma_pairs[args.MuEtaGamma][0]
        args.eta = Mu_Eta_Gamma_pairs[args.MuEtaGamma][1]
        args.gamma = Mu_Eta_Gamma_pairs[args.MuEtaGamma][2]

    ##########################################################################
    ''' Dataset '''
    ##########################################################################

    dsetname = args.dataset
    X, Y, _, dim, C = GetData(args.dataset)

    print(dim)
    print(C)
    print(X.shape[0])

    # by using below command, gpu is available
    dev = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    ##########################################################################
    ''' kNNG '''
    ##########################################################################

    start = time.time()

    alpha_vat = args.alpha_vat
    K = args.K # Number of neighbors

    knncachestr = "%s-k%d.npy" % (dsetname, K+1)

    if dsetname != "unknown" and os.path.exists(knncachestr):
        print("Loaded cached kNN from %s" % knncachestr)
        distances = np.load(knncachestr)
        indices = np.load("i" + knncachestr)

    else:
        nbrs = NearestNeighbors(n_neighbors=K+1, algorithm='brute').fit(X)
        distances, indices = nbrs.kneighbors(X)
        np.save(knncachestr, distances)
        np.save("i" + knncachestr, indices)

    K_vat = args.K_vat
    R = alpha_vat*distances[:,K_vat]
    R = R.reshape(X.shape[0],1)

    if args.geodesic > 0:

        # how many neighbors should be used to define geodesic metric, K_g =< K
        # two rings -> Kg=15
        K_g = K

        # build geodesically nearest neighbor graph, i.e., returns goeodic_distances and geodesic indices are returned
        _, g_indices = build_gnng(K_g, indices)

    end = time.time()
    print(f"{end-start} seconds by Brute-Force KNN.")

    ##########################################################################
    ''' Dataloader '''
    ##########################################################################

    R = torch.tensor(R.astype('f')).to(dev)
    X = torch.tensor(X.astype('f')).to(dev) # this unlabeled dataset (set of feature vectors) is input of IMSAT

    # define archtechture of MLP(Multi Layer Perceptron). 
    # in this net, batch-normalization (bn) is used. 
    # bn is very important to stabilize the training of net. 
    #torch.manual_seed(0)
    class Net(nn.Module): 

        def __init__(self):
            super(Net, self).__init__()

            self.l1 = nn.Linear(dim, 1200)
            self.bn1 = nn.BatchNorm1d(1200)

            self.l2 = nn.Linear(1200,1200)
            self.bn2 = nn.BatchNorm1d(1200)

            self.l3 = nn.Linear(1200,C)
            
        def forward(self, x):
        
            x = F.relu(self.bn1(self.l1(x)))
            x = F.relu(self.bn2(self.l2(x)))
            x = self.l3(x)
            
            return x

    def _l2_normalize(d):
        d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
        d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
        return d

    def return_vat_Loss(model, x, xi, eps):

        optimizer.zero_grad()
        
        with disable_tracking_bn_stats(model):
            with torch.no_grad():
                target = torch.softmax(model(x), 1) 
            
            d = torch.randn(x.shape, device = dev)
            d = _l2_normalize(d)
            d.requires_grad_()
            out_vadv = model(x + xi*d)
            hat = torch.softmax(out_vadv, 1)
            adv_distance = kl(target, hat)

            adv_distance.backward()
            
            d = _l2_normalize(d.grad)
            r_adv = eps * d
            out_vadv = model(x + r_adv)
            hat = torch.softmax(out_vadv, 1)
            R_vat = kl(target, hat)

        return R_vat

    net = Net()

    # throw net object to gpu
    net = net.to(dev)

    ##########################################################################
    ''' Training of IMSAT '''
    ##########################################################################

    # decide hyperparameter values for imsat training
    epochs = args.epochs # number of epochs

    xi = args.xi       # xi is used for making adversarial vector. if xi becomes smaller, 
                    # theoretically obtained r_vadv becomes more priecise


    mini_size = args.batchsize # size of mini-batch training dataset

    m = X.shape[0]//mini_size # number of iteration at each epoch 


    mu = args.mu
    gamma = args.gamma
    eta = args.eta




    ## define optimizer for set of parameters in deep neural network
    ## lr is the learning rate of parameter vector, betas are the lr of gradient and second moment of gradient
    optimizer = optim.Adam(net.parameters(), 
                            lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)


    ## tau_pos = 2 or 5 is good for reuters, c=10,20 tau_pos =.75?
    tau = args.tau
    alpha = args.alpha ## alpha-exp family -> choose alpha = 0 or 1 or 2
    # tau_neg = tau_pos # TODO: args.tau_neg

    #should be smaller than K=200, here we define T(x) for each x, K0 = 5, 7, 15, 100, 200(=K)
    #mnist, svhn, stl, omniglot, cifar100 -> K0=7
    #reuters10k -> K0=50
    #20news -> K0=200
    #cifar10 -> K0=15
    K0 = args.K0


    beta = args.beta ## 20news, reuters10K -> beta=4/5, the others -> beta=0

    # given the number of negative, how much rate of them should be used for trainingm, rate=1
    # means 100% used
    # rate = args.rate

    #itr_num = 0
    print("Start training of MIST_agm.")
    for epoch in range(epochs):
        print("At ", epoch, "-th epoch, ")

        # set empiricial loss to 0 in the beginning of epoch
        empirical_objective_loss = 0.0

        idx_eph = np.random.permutation(X.shape[0])
        
        net.train()

        for itr in range(m):
        
            ## chose a core idx of mini_batch
            idx_itr = idx_eph[itr*mini_size:(itr+1)*mini_size]

            # define components at each iteration
            X_itr = X[idx_itr,:]

            ########################## geodesic based transformation function related part ##############
            if args.geodesic > 0:

                lgth = g_indices.shape[1]
                if beta == 0:
                    X_agm1_itr, idx_agm1_itr = get_agm(X, idx_itr, g_indices[:,1:lgth]) 
                else:
                    X_agm1_itr, idx_agm1_itr = get_agm(X, idx_itr, g_indices[:,int(beta*lgth):lgth])

            # ########################## knn based transformation function related part ##############
            else:

                if beta == 0:
                    X_agm1_itr, idx_agm1_itr = get_agm(X, idx_itr, indices[:,1:K0+1]) #except 20news reuters10k
                else:
                    X_agm1_itr, idx_agm1_itr = get_agm(X, idx_itr, indices[:,int(beta*K0):K0+1]) #20news or reuters10k

            #################### VAT loss ################
            l_vat = return_vat_Loss(net, X_itr, xi, R[idx_itr,:])
            l_vat1 = return_vat_Loss(net, X_agm1_itr, xi, R[idx_agm1_itr,:])

            soft_out_itr = torch.softmax(net(X_itr) , 1)

            ####################### Siamese loss (or I_nce) related part ###########
            ## define positive and negative loss, where tau is a fixed value
            l_s, soft_out_agm1_itr = SiameseLoss(net, soft_out_itr, X_agm1_itr, tau, alpha, args.INCEvar)
            
            ######################## I(X;Y) related part #######################
            ## define entropy of y
            ent_y = entropy(torch.cat((soft_out_itr, soft_out_agm1_itr),0))

            ## define shannon conditional entropy loss H(p(y|x)) named by c_ent.
            c_ent = conditional_entropy(torch.cat((soft_out_itr, soft_out_agm1_itr),0))
            
            ######################### our objective defined ##################        
            # objective of sim
            objective = ((l_vat + l_vat1)/2) - mu*(  (eta*ent_y - c_ent) + gamma*( l_s )  )

            if itr % 25 == 0:
                wandb.log({
                    "loss": objective.detach().cpu().data,
                    "tau":  tau,
                })

            # update the set of parameters in deep neural network by minimizing loss
            optimizer.zero_grad() 
            objective.backward()
            optimizer.step()

            empirical_objective_loss = empirical_objective_loss + objective.data

            if args.optimize_tau:
                tau = minimize_tau(net, X_itr, X_agm1_itr, alpha, args.INCEvar)

        # #empirical_loss = running_loss/m
        empirical_objective_loss = empirical_objective_loss.cpu().numpy()

        print("average empirical objective loss is", empirical_objective_loss/m, ',')
        print("tau is now", tau, ".")

        net.eval()

        # at each epoch, prediction accuracy is displayed
        with torch.no_grad():
            out = net(X)
            preds = torch.argmax(out, dim=1)
            preds = preds.cpu().numpy()
            preds = preds.reshape(1, preds.shape[0])
            clustering_acc = ReturnACC(preds[0], Y[0], C)
        print("and current clustering accuracy is", clustering_acc )

        wandb.log({"accuracy": clustering_acc, "epoch": epoch})

if __name__ == "__main__":
    main()
    sys.stdout.flush()
