#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Computes the distance correlation between two matrices.
https://en.wikipedia.org/wiki/Distance_correlation
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform

######################################################
def dvar(X):
    """Computes the distance variance of a matrix X.
    """
    return np.sqrt(np.sum(X ** 2 / X.shape[0] ** 2))

######################################################
def cent_dist(X):
    """Computes the pairwise euclidean distance between rows of X and centers
     each cell of the distance matrix with row mean, column mean, and grand mean.
    """
    M = squareform(pdist(X))    # distance matrix
    rmean = M.mean(axis=1)
    cmean = M.mean(axis=0)
    gmean = rmean.mean()
    R = np.tile(rmean, (M.shape[0], 1)).transpose()
    C = np.tile(cmean, (M.shape[1], 1))
    G = np.tile(gmean, M.shape)
    CM = M - R - C + G
    return CM

######################################################
def dcov(X, Y):
    """Computes the distance covariance between matrices X and Y.
    """
    n = X.shape[0]
    XY = np.multiply(X, Y)
    cov = np.sqrt(XY.sum()) / n
    return cov

######################################################

def dcor(X, Y):
    """Computes the distance correlation between two matrices X and Y.
    X and Y must have the same number of rows.
    >>> X = np.matrix('1;2;3;4;5')
    >>> Y = np.matrix('1;2;9;4;4')
    >>> dcor(X, Y)
    0.76267624241686649
    """
    assert X.shape[0] == Y.shape[0]

    A = cent_dist(X)
    B = cent_dist(Y)

    dcov_AB = dcov(A, B)
    dvar_A = dvar(A)
    dvar_B = dvar(B)

    dcor = 0.0
    if dvar_A > 0.0 and dvar_B > 0.0:
        dcor = dcov_AB / np.sqrt(dvar_A * dvar_B)

    return dcor


######################################################

def Rmse_metric(X, Y):
    N=len(X.flatten())
    dS = (X.flatten() - Y.flatten())
    Rmse_value = np.sqrt((1/N)*dS.dot(dS))
    return Rmse_value

######################################################

def canberra_metric(X, Y):
    N=len(X.flatten())
    d = np.abs(X.flatten() - Y.flatten()) / (np.abs(X.flatten()) + np.abs(Y.flatten()))
    canberra_value = np.nansum(d)/N
    return canberra_value

######################################################

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0)) 
       
######################################################

def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)    