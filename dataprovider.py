import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from torch import Tensor
from torch.autograd import Variable
import torch
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

def getToys(means,stds,samplesize,scale=True):
    signal = np.stack([
            np.random.normal(mu[0],sigma[0],samplesize/2) for mu,sigma in zip(means,stds)
            ]).T
    bckg = np.stack([
            np.random.normal(mu[1],sigma[1],samplesize/2)
            for mu,sigma in zip(means,stds)
            ]).T
    X = np.concatenate([signal,bckg])
    y = np.array( [1. for x in signal]+[0. for x in bckg] )
    
    scaler = StandardScaler()
    if scale:
        X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
    print 'training size', len(y_train)
    print 'validation size', len(y_val)
    print 'test size', len(y_test)
    
    X_train = Variable(Tensor(X_train))
    y_train = Variable(Tensor(y_train), requires_grad=False)
    X_val = Variable(Tensor(X_val), volatile=True)
    y_val = Variable(Tensor(y_val), volatile=True)
    X_test = Variable(Tensor(X_test), volatile=True)
    y_test = Variable(Tensor(y_test), volatile=True)
    
    return X_train,y_train,X_val,y_val,X_test,y_test,scaler

def getWeakToys(means,stds,samplesize,fractions,scaler):
    Xs = []
    ys = []
    for f in fractions:
        signal = np.stack([
                np.random.normal(mu[0],sigma[0],int(samplesize*f)) for mu,sigma in zip(means,stds)
                ]).T
        bckg = np.stack([
                np.random.normal(mu[1],sigma[1],int(samplesize*(1-f)))
                for mu,sigma in zip(means,stds)
                ]).T
        X = np.concatenate([signal,bckg])
        X = scaler.transform(X)
        y = np.array( [f for x in X] )
        Xs.append(Variable(Tensor(X)))
        ys.append(Variable(Tensor(y), requires_grad=False))
    
    return Xs,ys

def getAdvWeakToys(means,stds,samplesize,fractions,shifts,scaler):
    Xs = []
    ys = []
    for i,f in enumerate(fractions):
        signal = np.stack([
                #np.random.normal(mu[0],sigma[0],int(samplesize*f)) for mu,sigma in zip(means,stds)
                np.random.normal(mu[0]*(1+shift*i),sigma[0],int(samplesize*f)) for mu,sigma,shift in zip(means,stds,shifts)
                ]).T
        bckg = np.stack([
                np.random.normal(mu[1],sigma[1],int(samplesize*(1-f)))
                for mu,sigma in zip(means,stds)
                ]).T
        X = np.concatenate([signal,bckg])
        X = scaler.transform(X)
        y = np.array( [f for x in X] )
        Xs.append(Variable(Tensor(X)))
        ys.append(Variable(Tensor(y), requires_grad=False))
    
    return Xs,ys


#                np.random.normal(mu[1]*(1+shift*i),sigma[1],int(samplesize*(1-f)))
