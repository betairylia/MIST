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

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

import os
import time

from munkres import Munkres

from datasets import *
from utils import *

import signal
from contextlib import contextmanager


@contextmanager
def timeout(time):
    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.alarm(time)

    try:
        yield
    except TimeoutError:
        print("Timeout (%dsec)." % time)
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


def raise_timeout(signum, frame):
    raise TimeoutError

def main():

    parser = argparse.ArgumentParser(description='SIM experiments')

    # Set-up
    parser.add_argument('--name', type=str, default='Untitled')
    parser.add_argument('--dataset', type=str, default='mnist')

    # General HP
    parser.add_argument('--method', type=str, default='kmeans', help='clustering method')
    args = parser.parse_args()

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

    print("Start clustering")

    try:

        with timeout(4200):
        
            if args.method == 'kmeans':
                method = KMeans(n_clusters = C)

            elif args.method == 'sc':
                method = SpectralClustering(n_clusters = C, verbose = True, affinity = 'nearest_neighbors', n_neighbors = 50, eigen_solver = 'amg')
                # method = SpectralClustering(n_clusters = C, verbose = True, affinity = 'rbf', eigen_solver = 'amg')

            elif args.method == 'gmmc':
                method = GaussianMixture(n_components = C, covariance_type = 'diag', verbose = 2)

            pred = method.fit_predict(X)

            acc = ReturnACC(pred, Y.squeeze(), C)
            print("Acc: %f" % acc)

            with open("out.temp", 'w') as fp:
                fp.write("%f" % acc)

    except TimeoutError:
        print("Timeout.")

if __name__ == "__main__":
    main()
    sys.stdout.flush()

# Results
'''
KMeans

Spectral (kNNG, 0/1 weight, pyAmg solver; k = 50 (others) / 200 (20news))
mnist       63.6543
svhn        26.9526
stl         83.0577
cifar10     36.5550
cifar100    Failed (?)
omniglot    Failed (cannot handle 100 classes ?)
20news      Failed (?)
reuters10k  43.5100

GMMC (diagnoal cov)
mnist       
svhn        
stl         
cifar10     
cifar100    
omniglot    
20news      
reuters10k  

'''
