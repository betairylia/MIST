import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from munkres import Munkres
import contextlib

from scipy.sparse import csr_matrix, triu
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix
from scipy.sparse import identity

from MIST_utils import disable_tracking_bn_stats

def relu(x):
    relued_x = x * (x>0)
    return relued_x

def minimize_tau(model, x, x_agm, alpha, var):
    
    from scipy.special import softmax
    from scipy.optimize import minimize_scalar

    m = x.shape[0]
    a = csr_matrix(np.ones((m,m)))

    idx_i, idx_j = a.nonzero()
    del a

    with disable_tracking_bn_stats(model):
        with torch.no_grad():
            y = torch.softmax(model(x), 1).cpu().numpy()
            y_agm = torch.softmax(model(x_agm), 1).cpu().numpy()
            ip = y[idx_i,:] * y_agm[idx_j,:]
            ip = np.sum(ip, 1).reshape(m,m)
    
    if alpha == 1 and var==0:
      def f(t, ip):
        critic = t * ip
        lp = np.mean(np.diag(critic,0))
        ln = -np.sum(np.tril(critic,-1) + np.triu(critic,1)) /(m*(m-1))
        l_s = (1 - (1/m))*(lp + ln)
        return -l_s
    
    elif alpha == 1 and var==1:
      def f(t, ip):
        critic = t * ip
        I1 = np.mean( np.log( np.diag( softmax(critic,1),0 ) ) ) 
        return -I1
    
    elif alpha == 1 and var==2:
      def f(t, ip):
        critic = t * ip
        I1 = np.mean( np.log( np.diag( softmax(critic,1),0 ) ) )
        I0 = np.mean( np.log( np.diag( softmax(critic,0),0 ) ) ) 
        return -(I1+I0)/2
    
    elif alpha == 2 and var==0:
      def f(t, ip):
        critic = -np.log( 1 + t*(1-ip) )
        lp = np.mean(np.diag(critic,0))
        ln = -np.sum(np.tril(critic,-1) + np.triu(critic,1)) /(m*(m-1))
        l_s = (1 - (1/m))*(lp + ln)
        return -l_s

    elif alpha == 2 and var==1:
      def f(t, ip):
        critic = -np.log( 1 + t*(1-ip) )
        I1 = np.mean( np.log( np.diag( softmax(critic,1),0 ) ) ) 
        return -I1
    
    elif alpha == 2 and var==2:
      def f(t, ip):
        critic = -np.log( 1 + t*(1-ip) )
        I1 = np.mean( np.log( np.diag( softmax(critic,1),0 ) ) )
        I0 = np.mean( np.log( np.diag( softmax(critic,0),0 ) ) ) 
        return -(I1+I0)/2

    elif alpha == 0 and var==0:
      def f(t, ip):
        critic = np.log( relu(1 + t*(ip-1)) )
        lp = np.mean(np.diag(critic,0))
        ln = -np.sum(np.tril(critic,-1) + np.triu(critic,1)) /(m*(m-1))
        l_s = (1 - (1/m))*(lp + ln)
        return -l_s

    elif alpha == 0 and var==1:
      def f(t, ip):
        critic = np.log( relu(1 + t*(ip-1)) )
        I1 = np.mean( np.log( np.diag( softmax(critic,1),0 ) ) ) 
        return -I1
    
    elif alpha == 0 and var==2:
      def f(t, ip):
        critic = np.log( relu(1 + t*(ip-1)) )
        I1 = np.mean( np.log( np.diag( softmax(critic,1),0 ) ) )
        I0 = np.mean( np.log( np.diag( softmax(critic,0),0 ) ) ) 
        return -(I1+I0)/2

    
    res = minimize_scalar(f, args=ip, bounds=(0, 0.1), method='bounded')
    tau = res.x

    return tau