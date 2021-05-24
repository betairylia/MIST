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
    parser.add_argument('--alpha', type=float, default=0.25, help='alpha')
    parser.add_argument('--K', type=int, default=200, help='K (maximum)')
    parser.add_argument('--K0', type=int, default=5, help='K0 (SIM)')
    parser.add_argument('--K_vat', type=int, default=10, help='K (VAT)')
    parser.add_argument('--xi', type=float, default=10.0, help='xi (VAT)')
    parser.add_argument('--mu', type=float, default=0.1, help='mu')
    parser.add_argument('--gamma', type=float, default=0.55, help='gamma')
    parser.add_argument('--eta1', type=float, default=5.0, help='eta1 (Marginal entropy)')
    parser.add_argument('--eta2', type=float, default=1.0, help='eta2 (Positive pairs)')
    parser.add_argument('--tau_pos', type=float, default=0.01, help='tau_positive')
    # parser.add_argument('--tau_neg', type=float, default=0.01, help='tau_negative') # tau_neg = tau_pos 
    parser.add_argument('--rate', type=float, default=1.0, help='Negative sampling rate (0~1)')
    parser.add_argument('--aug_func', type=str, default='kNNG', help='Augmentation function (not used now)')

    # Worker meta-data
    parser.add_argument('--runid', type=int, default=0, help='Run ID.')
    parser.add_argument('--keyargs', type=str, default="", help='Key variables in HP tune, splitted in commas')

    args = parser.parse_args()

    # Log into WandB
    wandb.init(project = 'sim', config = vars(args), group = GetArgsStr(args))

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

    alpha = args.alpha
    K = args.K # Number of neighbors

    if False: # X.shape[0] > 5*10**5:

        nms = NMSlibTransformer(n_neighbors = K) # Compute approximated knn graph
        Knn_graph = nms.fit_transform(X)

        # define adaptie radius for VAT
        Knn_dist = Knn_graph.data.reshape(X.shape[0],K+1)
        R = alpha*Knn_dist[:,K]
        R = R.reshape(X.shape[0],1) 

        del Knn_graph, Knn_dist

        end = time.time()
        print(f"{end-start} seconds by Approximated KNN.")

    else:

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
        R = alpha*distances[:,K_vat]
        R = R.reshape(X.shape[0],1)


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




    net = Net()

    # throw net object to gpu
    net = net.to(dev)

    def conditional_entropy(soft):
        loss = torch.sum(-soft*torch.log(soft + 1e-8)) / soft.shape[0]
        return loss

    def entropy(soft):
        avg_soft = torch.mean(soft, 0, True) 
        loss = -torch.sum(avg_soft * torch.log(avg_soft + 1e-8))
        return loss

    def kl(p, q):
        loss = torch.sum(p * torch.log((p + 1e-8) / (q + 1e-8))) / p.shape[0]
        return loss

    @contextlib.contextmanager
    def _disable_tracking_bn_stats(model):

        def switch_attr(m):
            if hasattr(m, 'track_running_stats'):
                m.track_running_stats ^= True
                
        model.apply(switch_attr)
        yield
        model.apply(switch_attr)

    def _l2_normalize(d):
        d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
        d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
        return d

    def return_vat_Loss(model, x, xi, eps):

        optimizer.zero_grad()
        
        with _disable_tracking_bn_stats(model):
            with torch.no_grad():
                target = torch.softmax(model(x), 1) 
            
            d = torch.randn(x.shape).to(dev)
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

    def SiameseLoss(model, soft_out1, x_agm, t1, t2, r):

        m = soft_out1.shape[0]
        a = np.ones((m,m)) - np.eye(m)
        a = csr_matrix(a)

        neg_idx_i, neg_idx_j = a.nonzero()
        del a

        l_neg = neg_idx_i.shape[0]
        num_neg_pairs = int(l_neg*r)

        
        with _disable_tracking_bn_stats(model):
            soft_out2 = torch.softmax(model(x_agm), 1)

            s = np.random.choice(list(range(l_neg)), size=num_neg_pairs, replace=False)

            neg_ip = soft_out1[neg_idx_i[s],:] * soft_out2[neg_idx_j[s],:]
            neg_ip = torch.sum(neg_ip, 1)
            neg_loss = torch.log(1 + t2*(1-neg_ip))
            neg_loss = -torch.sum(neg_loss) / num_neg_pairs


            pos_ip = soft_out1 * soft_out2
            pos_ip = torch.sum(pos_ip, 1)
            pos_loss = torch.log(1 + t1*(1-pos_ip))
            pos_loss = torch.sum(pos_loss) / m



        return pos_loss, neg_loss, soft_out2

    def ReturnACC(cluster, target_cluster, k):
        """ Compute error between cluster and target cluster
        :param cluster: proposed cluster
        :param target_cluster: target cluster
        k: number of classes
        :return: error
        """
        n = np.shape(target_cluster)[0]
        M = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                M[i][j] = np.sum(np.logical_and(cluster == i, target_cluster == j))
        m = Munkres()
        indexes = m.compute(-M)
        corresp = []
        for i in range(k):
            corresp.append(indexes[i][1])
        pred_corresp = [corresp[int(predicted)] for predicted in cluster]
        acc = np.sum(pred_corresp == target_cluster) / float(len(target_cluster))
        return acc 

    def get_agm(x, idx_itr, knn_idx):
        knn_idx_itr = knn_idx[idx_itr,:]
        for i in range(knn_idx_itr.shape[0]):
            v = np.random.permutation(knn_idx_itr[i,:])
            knn_idx_itr[i,:] = v

        v = np.random.choice(list(range(knn_idx_itr.shape[1])), size=1, replace=False)
        idx0 = knn_idx_itr[:,v[0]]
        x_agm0 = x[idx0,:]
        
        return x_agm0, idx0

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
    eta1 = args.eta1




    ## define optimizer for set of parameters in deep neural network
    ## lr is the learning rate of parameter vector, betas are the lr of gradient and second moment of gradient
    optimizer = optim.Adam(net.parameters(), 
                            lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)


    ## tau_pos = 2 or 5 is good for reuters, c=10,20 tau_pos =.75?
    tau_pos = args.tau_pos
    tau_neg = tau_pos # TODO: args.tau_neg


    # given the number of negative, how much rate of them should be used for trainingm, rate=1
    # means 100% used
    rate = args.rate


    #itr_num = 0
    print("Start training of SIM_agm.")
    for epoch in range(epochs):
        print("At ", epoch, "-th epoch, ")

        # set empiricial loss to 0 in the beginning of epoch
        empirical_loss = 0.0

        idx_eph = np.random.permutation(X.shape[0])
        
        net.train()

        for itr in range(m):
        
            ## chose a core idx of mini_batch
            idx_itr = idx_eph[itr*mini_size:(itr+1)*mini_size]

            # define components at each iteration
            X_itr = X[idx_itr,:]


            K0 = args.K0 #should be smaller than K=200, here we define T(x) for each x, K0 = 5, 10, 15, 20, 25, 50, 100, 150, 200(=K)
            
            # TODO: args.aug_func
            # X_agm1_itr, idx_agm1_itr = get_agm(X, idx_itr, indices[:,1:K0+1])
            X_agm1_itr, idx_agm1_itr = get_agm(X, idx_itr, indices[:,int(2*K0/3):K0+1])
            # X_agm1_itr, idx_agm1_itr = get_agm(X, idx_itr, indices[:,int(K0/2):K0+1])
            


            R_vat = return_vat_Loss(net, X_itr, xi, R[idx_itr,:])
            R_vat1 = return_vat_Loss(net, X_agm1_itr, xi, R[idx_agm1_itr,:])



            soft_out_itr = torch.softmax(net(X_itr) , 1)
        

            ## define positive and negative loss
            l_p, l_n, soft_out_agm1_itr = SiameseLoss(net, soft_out_itr, X_agm1_itr, tau_pos, tau_neg, rate)

            ## define entropy of y
            ent_y = entropy(torch.cat((soft_out_itr, soft_out_agm1_itr),0))

            ## define shannon conditional entropy loss H(p(y|x)) named by c_ent.
            c_ent = conditional_entropy(torch.cat((soft_out_itr, soft_out_agm1_itr),0))
            
                    
            # objective of sim
            eta2 = args.eta2
            objective = ((R_vat + R_vat1)/2) - mu*( (1-gamma)*(eta1*ent_y - c_ent) + gamma*eta2*(-l_p - l_n)  )

            wandb.log({"loss": objective.detach().cpu().data})



            # update the set of parameters in deep neural network by minimizing loss
            optimizer.zero_grad() 
            objective.backward()
            optimizer.step()

            empirical_loss = empirical_loss + objective.data



        #empirical_loss = running_loss/m
        empirical_loss = empirical_loss.cpu().numpy()
        print("average empirical loss is", empirical_loss/m, ',')

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
