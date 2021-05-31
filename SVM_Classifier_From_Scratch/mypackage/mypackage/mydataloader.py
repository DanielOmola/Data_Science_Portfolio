# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:45:01 2021

@author: daniel omola
"""


import numpy as np

np.random.seed(25)

def generate_linearly_separable_data(size=10,verbose=None):
    """
    #######################################################
    #
    #       Generate linearly separable dataset
    #           X (features) and y (target)
    #
    #######################################################
    """
    n = size
    X_p = np.stack([np.random.rand(n)*6,np.random.rand(n)*5+3],axis=1)
    y_p = np.ones(len(X_p), dtype=int)
    X_n = np.stack([np.random.rand(n)*6+2,np.random.rand(n)*5-3],axis=1)
    y_n = -np.ones(X_n.shape[0], dtype=int)
    X = np.concatenate([X_p,X_n])
    y = np.concatenate([y_p,y_n])
    if verbose:
        print("\n--- X --- \n\n", X)
        print("\n--- y --- \n\n", y)
    return X,y

def generate_non_linearly_separable_data(size=10,verbose=None):
    """
    #######################################################
    #
    #       Generate non linearly separable dataset
    #           X (features) and y (target)
    #
    #######################################################
    """
    n = size
    X_p = np.stack([np.random.rand(n)*6,np.random.rand(n)*5+3],axis=1)
    y_p = np.ones(len(X_p), dtype=int)
    X_n = np.stack([np.random.rand(n)*6+2,np.random.rand(n)*6],axis=1)
    y_n = -np.ones(X_n.shape[0], dtype=int)
    X = np.concatenate([X_p,X_n])
    y = np.concatenate([y_p,y_n])
    if verbose:
        print("\n--- X --- \n\n", X)
        print("\n--- y --- \n\n", y)
    return X,y