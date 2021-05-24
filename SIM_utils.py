import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from munkres import Munkres
import contextlib

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
