import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random

# Helpers
def get_custom_feat(data_dir):

    data = torch.load(data_dir)
    
    X = torch.cat([data['trainX'], data['testX']], 0)
    Y = torch.cat([data['trainY'], data['testY']], 0)

    X = X.reshape(X.shape[0], -1)
    Y = Y.reshape(Y.shape[0],)

    # dataset = dict()
    # dataset['X']=X
    # dataset['Y']=Y

    # dataloader=DataLoader(TensorDataset(X,Y),batch_size=batch_size,shuffle=True,num_workers=4)

    meta = {'dim': X.shape[1], 'nClasses': int(max(Y).item())+1}

    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    Y = Y.reshape(1,Y.shape[0])

    return X, Y, X.shape[0], meta['dim'], meta['nClasses']


def get_custom_feat_npy(data_dir_X, data_dir_Y):

    X = torch.FloatTensor(np.load(data_dir_X))
    Y = torch.LongTensor(np.load(data_dir_Y))
    
    X = X.reshape(X.shape[0], -1)
    Y = Y.reshape(Y.shape[0],)

    # dataset = dict()
    # dataset['X']=X
    # dataset['Y']=Y

    # dataloader=DataLoader(TensorDataset(X,Y),batch_size=batch_size,shuffle=True,num_workers=4)

    meta = {'dim': X.shape[1], 'nClasses': int(max(Y).item())+1}

    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    Y = Y.reshape(1,Y.shape[0])

    return X, Y, X.shape[0], meta['dim'], meta['nClasses']

def get_two_rings():

    # Make Two-Rings
    from sklearn.datasets import make_circles
    X, Y = make_circles(n_samples=5000,
                  shuffle = None,
                  noise = 0.01,
                  random_state=True,
                  factor = 0.35)

    Y = Y.reshape(1,Y.shape[0])
    dim = 2
    C = 2

    # print(X)
    # print(Y)

    # dsetname = "Two-Rings"

    return X, Y, None, dim, C

# TODO: change from rings to moons
def get_two_moons():

    # Make Two-Rings
    from sklearn.datasets import make_circles
    X, Y = make_moons(n_samples=5000,
                  shuffle = None,
                  noise = 0.05,
                  random_state = True)

    Y = Y.reshape(1,Y.shape[0])
    dim = 2
    C = 2

    # print(X)
    # print(Y)

    # dsetname = "Two-Rings"

    return X, Y, None, dim, C

# download mnist dataset from keras, then define training dataset X and its true labels Y
from tensorflow.keras.datasets import mnist

def get_mnist():

    (X_tr, Y_tr), (X_tst, Y_tst) = mnist.load_data()

    (num_tr, depth, _) = X_tr.shape
    (num_tst, _, _) = X_tst.shape

    dim = depth**2

    X_tr = X_tr.reshape(num_tr, dim) # from 28*28 image to naive 784 dim feature vector 
    X_tst = X_tst.reshape(num_tst, dim)
    X = np.r_[X_tr, X_tst]
    print("Size of training dataset is", X.shape)
    X = X/255
    # X = (2*(X/255)) - 1

    Y = np.r_[Y_tr, Y_tst]
    Y = Y.reshape(1, Y.shape[0])

    C = 10 # the number of classes for MNIST

    return X, Y, None, dim, C

# Make Fshion Mnist dataset
from torchvision.datasets import FashionMNIST

def get_fashion(data_dir = './data/fashion/', batch_size=128, shuffle = True):

    train = FashionMNIST(root = data_dir, train = True, download = True)
    test = FashionMNIST(root = data_dir, train = False, download = True)

    X=torch.cat([train.data.float().view(-1,784)/255.,test.data.float().view(-1,784)/255.],0)
    Y=torch.cat([train.targets,test.targets],0)

    meta = {'dim': 784, 'nClasses': 10}

    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    Y = Y.reshape(1,Y.shape[0])

    return X, Y, X.shape[0], meta['dim'], meta['nClasses']

# Make 20news dataset

from sklearn.datasets import fetch_20newsgroups, make_circles, make_moons, make_blobs, fetch_openml
from sklearn.feature_extraction.text import TfidfVectorizer


def get_20news():

    data = fetch_20newsgroups(subset = 'all')
    vectorizer = TfidfVectorizer(max_features = 2000, stop_words = 'english')

    X = vectorizer.fit_transform(data.data).todense().astype(np.float32)
    Y = data.target

    n_samples = X.shape[0]
    dim = X.shape[1]

    Y = Y.reshape(1, n_samples)

    n_classes = 20

    return X, Y, n_samples, dim, n_classes

# Make Reuters10K dataset
def get_reuters10K():
    return get_custom_feat_npy('./IMSAT_datasets/all_dataset/reuters/10k_feature.npy', './IMSAT_datasets/all_dataset/reuters/10k_target.npy')

# Make CIFAR10 dataset
# Features extracted by Zhang following IMSAT paper
def get_CIFAR10_zhang():
    return get_custom_feat('./CIFAR10_feat/data.pkl')

# Features provided with IMSAT code
# Zhang cannot get reported performance with fixed eps; did not try for adaptive epsilon.
def get_CIFAR10_IMSAT():

    PATH = './IMSAT_datasets/all_dataset/cifar/'

    y_train_ul = np.load(PATH+'train_labels.npy').astype(np.int32)
    y_test = np.load(PATH+'test_labels.npy').astype(np.int32)
    y_whole = np.concatenate((y_train_ul, y_test), axis = 0)
    x_whole = np.load(PATH+'resnet.npz')['arr_0']

    X = torch.Tensor(x_whole).reshape(x_whole.shape[0], -1)
    Y = torch.LongTensor(y_whole).reshape(y_whole.shape[0])

    meta = {'dim': X.shape[1], 'nClasses': 10}

    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    Y = Y.reshape(1,Y.shape[0])

    return X, Y, X.shape[0], meta['dim'], meta['nClasses']

# Make CIFAR100 dataset
# Features extracted by Zhang following IMSAT paper
def get_CIFAR100_zhang():
    return get_custom_feat('./CIFAR100_feat/data.pkl')

# Features provided with IMSAT code
def get_CIFAR100_IMSAT():

    PATH = './IMSAT_datasets/all_dataset/cifar100/'

    y_whole = np.load(PATH + 'y.npy')
    x_whole = np.load(PATH+'resnet.npz')['arr_0']

    X = torch.Tensor(x_whole).reshape(x_whole.shape[0], -1)
    Y = torch.LongTensor(y_whole).reshape(y_whole.shape[0])

    meta = {'dim': X.shape[1], 'nClasses': 100}

    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    Y = Y.reshape(1,Y.shape[0])

    return X, Y, X.shape[0], meta['dim'], meta['nClasses']

# Make SVHN dataset
def get_svhn():
    
    PATH = './IMSAT_datasets/all_dataset/svhn/'

    x_train_ul = np.load(PATH+'train_feature.npy').astype(np.float32)
    x_test = np.load(PATH+'test_feature.npy').astype(np.float32)
    y_train_ul = np.load(PATH+'train_target.npy').astype(np.int32)
    y_test = np.load(PATH+'test_target.npy').astype(np.int32)

    x_whole = np.concatenate((x_train_ul, x_test), axis = 0)
    y_whole = np.concatenate((y_train_ul, y_test), axis = 0)

    X = torch.Tensor(x_whole).reshape(x_whole.shape[0], -1)
    Y = torch.LongTensor(y_whole).reshape(y_whole.shape[0])

    meta = {'dim': X.shape[1], 'nClasses': int(max(Y).item())+1}

    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    Y = Y.reshape(1,Y.shape[0])

    return X, Y, X.shape[0], meta['dim'], meta['nClasses']

# Make Omniglot dataset
def get_omniglot():
    
    PATH = './IMSAT_datasets/all_dataset/omniglot/'
    scale=   1.0 / 255.0
    shift= - 0.0

    augmented_data = (np.load(PATH+"augmented_omniglot_downsampled5_data.npz")['arr_0']).astype(np.float32)*scale + shift
    augmented_target = np.load(PATH+"augmented_omniglot_downsampled5_target.npz")['arr_0']

    X = torch.Tensor(augmented_data).reshape(augmented_data.shape[0], -1)
    Y = torch.LongTensor(augmented_target).reshape(augmented_target.shape[0])

    meta = {'dim': X.shape[1], 'nClasses': int(max(Y).item())+1}

    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    Y = Y.reshape(1,Y.shape[0])

    return X, Y, X.shape[0], meta['dim'], meta['nClasses']

# Make STL dataset

def get_imsat_stl10(batch_size=128):

    PATH = './IMSAT_datasets/all_dataset/stl/'
    label = np.fromfile(PATH+'train_y.bin', dtype=np.uint8)
    test_label = np.fromfile(PATH+'test_y.bin', dtype=np.uint8)
    y_whole = (np.concatenate((label, test_label), axis = 0) - 1).astype(np.int32)
    x_whole = np.load(PATH + 'resnet.npz')['arr_0']
    X = torch.Tensor(x_whole).reshape(x_whole.shape[0], -1)
    Y = torch.LongTensor(y_whole).reshape(y_whole.shape[0])
    meta = {'dim': X.shape[1], 'nClasses': 10}

    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    Y = Y.reshape(1,Y.shape[0])

    return X, Y, X.shape[0], meta['dim'], meta['nClasses']

def GetData(datasetName):

    if datasetName == "mnist":
        X, Y, _, dim, C = get_mnist()
    elif datasetName == "fashion":
        X, Y, _, dim, C = get_fashion()
    elif datasetName == "20news":
        X, Y, _, dim, C = get_20news()
    elif datasetName == "reuters10k":
        X, Y, _, dim, C = get_reuters10K()
    elif datasetName == "cifar10":
        X, Y, _, dim, C = get_CIFAR10_IMSAT()
    elif datasetName == "cifar100":
        X, Y, _, dim, C = get_CIFAR100_IMSAT()
    elif datasetName == "svhn":
        X, Y, _, dim, C = get_svhn()
    elif datasetName == "omniglot":
        X, Y, _, dim, C = get_omniglot()
    elif datasetName == "stl":
        X, Y, _, dim, C = get_imsat_stl10()

    elif datasetName == "rings":
        X, Y, _, dim, C = get_two_rings()
    elif datasetName == "moons":
        X, Y, _, dim, C = get_two_moons()

    return X, Y, _, dim, C

def GetData_VaDEEmbeddings(datasetName):

    VaDEpath = "/home/betairya/RP_ML/VaDE-pytorch"
    return get_custom_feat_npy(
        os.path.join(VaDEpath, "VaDE_X_%s.npy" % datasetName),
        os.path.join(VaDEpath, "VaDE_Y_%s.npy" % datasetName)
    )
