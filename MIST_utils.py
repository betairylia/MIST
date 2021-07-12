import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from munkres import Munkres
import contextlib

import time
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix
from scipy.sparse import identity
import scipy as sp

def Knn2Adj(idx, k):

    n = idx.shape[0]

    idx_i = np.array(range(n))
    idx_i = np.tile(idx_i, (k+1,1)).transpose().reshape(1, n*(k+1))[0]
    idx_j = idx.reshape(1, n*(k+1))[0]

    adj = csc_matrix((np.ones(n*(k+1)),(idx_i,idx_j)), shape=(n,n)).astype(float)
    e = identity(n)
    adj = adj - e
    adj = adj + adj.transpose() - adj.multiply(adj.transpose())

    del idx_i, idx_j, e

    return adj



def build_gnng(K_g, indices):
    KnnIDXwithKg = indices[:,0:K_g+1]
    A = Knn2Adj(KnnIDXwithKg, K_g)

    start = time.time()
    # geodesic_dist = csc_matrix(sp.sparse.csgraph.johnson(A))
    g_distances = sp.sparse.csgraph.johnson(A)
    del A
    g_distances = np.where( (g_distances==np.inf) | (g_distances==0), 0, g_distances) 
    m = np.min(np.count_nonzero(g_distances, axis=1))
    n = g_distances.shape[1]
    g_indices = np.argsort(g_distances, axis=0)
    #g_indices = g_indices[:,n-K_g:n]
    g_indices = g_indices[:,n-m:n]
    g_distances = csc_matrix(g_distances)
    #del geodesic_dist
    #geodesic_dist = np.where(geodesic_dist>0, 1, geodesic_dist)
    #geodesic_nng = geodesic_dist.nonzero()
    end = time.time()
    print(f"{end-start} seconds by computing geodesic distance.")

    return g_distances, g_indices

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
def disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
            
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)

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
