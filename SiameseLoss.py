import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from munkres import Munkres
import contextlib

from scipy.sparse import csr_matrix, triu
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix
from scipy.sparse import identity

from SIM_utils import disable_tracking_bn_stats

## upper-bound of I_nce (0)
def SiameseLoss_UB(model, soft_out1, x_agm, t1, alpha):

    m = soft_out1.shape[0]
    a = csr_matrix(np.ones((m,m)))
   
    idx_i, idx_j = a.nonzero()
    del a

     
    with disable_tracking_bn_stats(model):
        soft_out2 = torch.softmax(model(x_agm), 1)

        ip = soft_out1[idx_i,:] * soft_out2[idx_j,:]
        ip = torch.sum(ip, 1).reshape(m,m)

        if alpha==0:
          critic = torch.log( 1 + t1*(ip - 1) )

        if alpha==1:
          critic = t1 * ip
          
        if alpha==2:
          critic = -torch.log( 1 + t1*(1-ip) )

        lp = torch.mean(torch.diag(critic,0)) 
        ln = -torch.sum(torch.tril(critic,-1) + torch.triu(critic,1)) /(m*(m-1))
        l_s = (1 - (1/m))*(lp + ln)

    return l_s, soft_out2





## exact I_nce (1)
def SiameseLoss_Exact(model, soft_out1, x_agm, t1, alpha):

    m = soft_out1.shape[0]
    a = csr_matrix(np.ones((m,m)))
   
    idx_i, idx_j = a.nonzero()
    del a

     
    with disable_tracking_bn_stats(model):
        soft_out2 = torch.softmax(model(x_agm), 1)

        ip = soft_out1[idx_i,:] * soft_out2[idx_j,:]
        ip = torch.sum(ip, 1).reshape(m,m)

        if alpha==0:
          critic = torch.log( 1 + t1*(ip - 1) )

        if alpha==1:
          critic = t1 * ip
          
        if alpha==2:
          critic = -torch.log( 1 + t1*(1-ip) )


        I_nce = np.log(m) + torch.mean( torch.log( torch.diag( torch.softmax(critic,1),0 ) ) )

        l_s = I_nce

    return l_s, soft_out2



## symmetrized I_nce (2)
def SiameseLoss_Symmetrized(model, soft_out1, x_agm, t1, alpha):

    m = soft_out1.shape[0]
    a = csr_matrix(np.ones((m,m)))
   
    idx_i, idx_j = a.nonzero()
    del a

     
    with disable_tracking_bn_stats(model):
        soft_out2 = torch.softmax(model(x_agm), 1)

        ip = soft_out1[idx_i,:] * soft_out2[idx_j,:]
        ip = torch.sum(ip, 1).reshape(m,m)

        if alpha==0:
          critic = torch.log( 1 + t1*(ip - 1) )

        if alpha==1:
          critic = t1 * ip
          
        if alpha==2:
          critic = -torch.log( 1 + t1*(1-ip) )
        

        I_nce1 = np.log(m) + torch.mean( torch.log( torch.diag( torch.softmax(critic,1),0 ) ) )
        I_nce0 = np.log(m) + torch.mean( torch.log( torch.diag( torch.softmax(critic,0),0 ) ) )


        l_s = (I_nce1 + I_nce0) / 2

    return l_s, soft_out2