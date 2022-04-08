import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from munkres import Munkres
import contextlib

from scipy.sparse import csr_matrix, triu
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix
from scipy.sparse import identity

from MIST_utils import disable_tracking_bn_stats, kl

def iic(p1, p2):

    P = torch.matmul(torch.t(p1), p2) / p1.shape[0]
    P = ( torch.t(P) +  P ) / 2
    P_marginal1 = torch.sum(P, dim=1)
    P_marginal2 = torch.sum(P, dim=0)
    P_marginal_mat = torch.matmul(P_marginal2, P_marginal1)

    loss = -kl(P, P_marginal_mat)

    return loss

def SiameseLoss(model, soft_out1, x_agm, t1, alpha, var):

    m = soft_out1.shape[0]
    a = csr_matrix(np.ones((m,m)))
   
    idx_i, idx_j = a.nonzero()
    del a

     
    with disable_tracking_bn_stats(model):
        soft_out2 = torch.softmax(model(x_agm), 1)

        ip = soft_out1[idx_i,:] * soft_out2[idx_j,:]
        ip = torch.sum(ip, 1).reshape(m,m)

        ###### define critic based on each alpha
        if alpha==0:
          critic = torch.log( F.relu(1 + t1*(ip - 1)) )


        elif alpha==1:
          critic = t1 * ip

          
        else:
          critic = -torch.log( 1 + t1*(1-ip) )


        ########### define info nce variants based on var==0(upper),1(exact),2(symmetrized)
        if var==1:
          ## exact info nce
          I_nce = np.log(m) + torch.mean( torch.log( torch.diag( torch.softmax(critic,1),0 ) ) )
          l_s = I_nce
        
        elif var==0:
          ## upper of info nce
          lp = torch.mean(torch.diag(critic,0)) 
          ln = -torch.sum(torch.tril(critic,-1) + torch.triu(critic,1)) /(m*(m-1))
          l_s = (1 - (1/m))*(lp + ln)

        elif var == 2:
          ## symmetrized info nce
          I_nce1 = np.log(m) + torch.mean( torch.log( torch.diag( torch.softmax(critic,1),0 ) ) )
          I_nce0 = np.log(m) + torch.mean( torch.log( torch.diag( torch.softmax(critic,0),0 ) ) )
          l_s = (I_nce1 + I_nce0) / 2
    
        elif var == 3:
          ## IIC-ish
          l_s = iic(soft_out1, soft_out2)
        

    return l_s, soft_out2
