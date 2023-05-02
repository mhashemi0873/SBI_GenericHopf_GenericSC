#!/usr/bin/env python3
"""
@author: meysamhashemi  INS Marseille

"""
import os
import sys
import numpy as np
import scipy as scp

from scipy import signal
from scipy import stats as spstats
from scipy.signal import hilbert
from scipy.stats import moment, mode, skew, kurtosis

from numpy import linalg as LA
from sklearn.decomposition import PCA
from sklearn import manifold
import sksfa
import umap

from FCD import extract_FCD
######################################################
def calculate_summary_statistics(x, nn, features):
    """Calculate summary statistics

    Parameters
    ----------
    x : output of the simulator

    Returns
    -------
    np.array, summary statistics
    """
    
        

    X=x.reshape(nn, int(x.shape[0]/nn))

    X_t=X.transpose()

    n_summary = 16*nn+(nn*nn)+300*300

    fs = 10e3 

    wwidth=30
    maxNwindows=200
    olap=0.94
        

    n_neighbors = 10
    n_samples = int(X.shape[1])
    n_components=2


    params = {
        "n_neighbors": n_neighbors,
        "n_components": n_components,
        "eigen_solver": "auto",
    }

    
    sum_stats_vec = np.concatenate((np.mean(X, axis=1),))
        
    for item in features:

            if item is 'moments':

                        sum_stats_vec = np.concatenate((np.mean(X, axis=1), 
                                                        np.median(X, axis=1),
                                                        np.std(X, axis=1),
                                                        skew(X, axis=1), 
                                                        kurtosis(X, axis=1),
                                                        ))
                       

            if item is 'higher_moments':

                        sum_stats_vec = np.concatenate((sum_stats_vec,
                                                moment(X, moment=2, axis=1),
                                                moment(X, moment=3, axis=1),
                                                moment(X, moment=4, axis=1),
                                                moment(X, moment=5, axis=1),
                                                moment(X, moment=6, axis=1),
                                                moment(X, moment=7, axis=1),
                                                moment(X, moment=8, axis=1),
                                                moment(X, moment=9, axis=1), 
                                                moment(X, moment=10, axis=1),        
                                                                   ))
                        
            if item is 'spectral_power':

                        f, Pxx_den =  signal.periodogram(X, fs)

                        sum_stats_vec = np.concatenate((sum_stats_vec,
                                        np.max(Pxx_den, axis=1), 
                                        np.mean(Pxx_den, axis=1),
                                        np.median(Pxx_den, axis=1),
                                        np.std(Pxx_den, axis=1),
                                        skew(Pxx_den, axis=1), 
                                        kurtosis(Pxx_den, axis=1), 
                                        np.diag(np.dot(Pxx_den, Pxx_den.transpose())),
                                                       ))

       
            if item is 'envelope':

                        analytic_signal = hilbert(X)
                        amplitude_envelope = np.abs(analytic_signal)
                        instantaneous_phase = np.unwrap(np.angle(analytic_signal))

                        sum_stats_vec = np.concatenate((sum_stats_vec,
                                        np.mean(amplitude_envelope, axis=1),
                                        np.std(amplitude_envelope, axis=1),
                                        np.mean(instantaneous_phase, axis=1),
                                        np.std(instantaneous_phase, axis=1),
                                                       ))
                            

            if item is 'FC_corr':

                        FCcorr=np.corrcoef(X)
                        sum_off_diag_FC = np.sum(FCcorr) - np.trace(FCcorr)
                        FC_TRIU = np.triu(FCcorr, k=10)
                        eigen_vals_FC, _ = LA.eig(FCcorr)
                        real_eigen_vals_FC=np.real(eigen_vals_FC)

                        pca = PCA(n_components=3)
                        PCA_FC = pca.fit_transform(FCcorr)

                        Upper_FC = []
                        Lower_FC = []
                        for i in range(0,len(FCcorr)):
                            Upper_FC.extend(FCcorr[i][i+1:])
                            Lower_FC.extend(FCcorr[i][0:i])

                        sum_stats_vec = np.concatenate((sum_stats_vec,
                                        np.array([np.max(real_eigen_vals_FC.reshape(-1))]),
                                        np.array([np.min(real_eigen_vals_FC.reshape(-1))]), 
                                        np.array([np.std(real_eigen_vals_FC.reshape(-1))]),
                                        np.array([skew(real_eigen_vals_FC.reshape(-1))]),
                                        np.array([kurtosis(real_eigen_vals_FC.reshape(-1))]),

                                        np.array([np.max((real_eigen_vals_FC.reshape(-1))**2)]),
                                        np.array([np.min((real_eigen_vals_FC.reshape(-1))**2)]),
                                        np.array([np.std((real_eigen_vals_FC.reshape(-1))**2)]),
                                        np.array([np.sum((real_eigen_vals_FC.reshape(-1))**2)]),
                                        np.array([skew((real_eigen_vals_FC.reshape(-1))**2)]),
                                        np.array([kurtosis((real_eigen_vals_FC.reshape(-1))**2)]),

                                        np.array([np.max(eigen_vals_FC.reshape(-1))]),
                                        np.array([np.min(eigen_vals_FC.reshape(-1))]),
                                        np.array([np.std(eigen_vals_FC.reshape(-1))]),
                                        np.array([skew(eigen_vals_FC.reshape(-1))]),
                                        np.array([kurtosis(eigen_vals_FC.reshape(-1))]),

                                        np.array([np.sum(PCA_FC.reshape(-1))]),                                         
                                        np.array([np.max(PCA_FC.reshape(-1))]),  
                                        np.array([np.min(PCA_FC.reshape(-1))]),  
                                        np.array([np.std(PCA_FC.reshape(-1))]),  
                                        np.array([skew(PCA_FC.reshape(-1))]),
                                        np.array([kurtosis(PCA_FC.reshape(-1))]),  

                                        np.array([np.sum(Upper_FC)]),                                         
                                        np.array([np.max(Upper_FC)]),  
                                        np.array([np.min(Upper_FC)]),  
                                        np.array([np.mean(Upper_FC)]),  
                                        np.array([np.std(Upper_FC)]),  
                                        np.array([skew(Upper_FC)]),
                                        np.array([kurtosis(Upper_FC)]), 

                                        np.array([np.sum(FC_TRIU.reshape(-1))]),
                                        np.array([np.max(FC_TRIU.reshape(-1))]),
                                        np.array([np.min(FC_TRIU.reshape(-1))]),
                                        np.array([np.mean(FC_TRIU.reshape(-1))]),
                                        np.array([np.std(FC_TRIU.reshape(-1))]),
                                        np.array([skew(FC_TRIU.reshape(-1))]),
                                        np.array([kurtosis(FC_TRIU.reshape(-1))]),
                                                        
                                        np.array([sum_off_diag_FC]),
                                          
                                        np.array([np.sum(FCcorr.reshape(-1))]),
                                        np.array([np.var(FCcorr.reshape(-1))]),
                                                ))
                        
            if item is 'FCD_corr':

                        FCDcorr,Pcorr,shift=extract_FCD(X,wwidth,maxNwindows,olap,mode='corr')
                        sum_off_diag_FCD = np.sum(FCDcorr) - np.trace(FCDcorr)
                        FCD_TRIU = np.triu(FCDcorr, k=10)
                        eigen_vals_FCD, _ = LA.eig(FCDcorr)
                        real_eigen_vals_FCD=np.real(eigen_vals_FCD)

                        pca = PCA(n_components=3)
                        PCA_FCD = pca.fit_transform(FCDcorr)

                        Upper_FCD = []
                        Lower_FCD = []
                        for i in range(0,len(FCDcorr)):
                            Upper_FCD.extend(FCDcorr[i][i+1:])
                            Lower_FCD.extend(FCDcorr[i][0:i])



                        sum_stats_vec = np.concatenate((sum_stats_vec,
                                        np.array([np.max(real_eigen_vals_FCD.reshape(-1))]),
                                        np.array([np.min(real_eigen_vals_FCD.reshape(-1))]), 
                                        np.array([np.std(real_eigen_vals_FCD.reshape(-1))]),
                                        np.array([skew(real_eigen_vals_FCD.reshape(-1))]),
                                        np.array([kurtosis(real_eigen_vals_FCD.reshape(-1))]),

                                        np.array([np.max((real_eigen_vals_FCD.reshape(-1))**2)]),
                                        np.array([np.min((real_eigen_vals_FCD.reshape(-1))**2)]),
                                        np.array([np.std((real_eigen_vals_FCD.reshape(-1))**2)]),
                                        np.array([np.sum((real_eigen_vals_FCD.reshape(-1))**2)]),
                                        np.array([skew((real_eigen_vals_FCD.reshape(-1))**2)]),
                                        np.array([kurtosis((real_eigen_vals_FCD.reshape(-1))**2)]),

                                        np.array([np.max(eigen_vals_FCD.reshape(-1))]),
                                        np.array([np.min(eigen_vals_FCD.reshape(-1))]),
                                        np.array([np.std(eigen_vals_FCD.reshape(-1))]),
                                        np.array([skew(eigen_vals_FCD.reshape(-1))]),
                                        np.array([kurtosis(eigen_vals_FCD.reshape(-1))]),

                                        np.array([np.sum(PCA_FCD.reshape(-1))]),                                         
                                        np.array([np.max(PCA_FCD.reshape(-1))]),  
                                        np.array([np.min(PCA_FCD.reshape(-1))]),  
                                        np.array([np.std(PCA_FCD.reshape(-1))]),  
                                        np.array([skew(PCA_FCD.reshape(-1))]),
                                        np.array([kurtosis(PCA_FCD.reshape(-1))]),  

                                        np.array([np.sum(Upper_FCD)]),                                         
                                        np.array([np.max(Upper_FCD)]),  
                                        np.array([np.min(Upper_FCD)]),  
                                        np.array([np.mean(Upper_FCD)]),  
                                        np.array([np.std(Upper_FCD)]),  
                                        np.array([skew(Upper_FCD)]),
                                        np.array([kurtosis(Upper_FCD)]), 

                                        np.array([np.sum(FCD_TRIU.reshape(-1))]),
                                        np.array([np.max(FCD_TRIU.reshape(-1))]),
                                        np.array([np.min(FCD_TRIU.reshape(-1))]),
                                        np.array([np.mean(FCD_TRIU.reshape(-1))]),
                                        np.array([np.std(FCD_TRIU.reshape(-1))]),
                                        np.array([skew(FCD_TRIU.reshape(-1))]),
                                        np.array([kurtosis(FCD_TRIU.reshape(-1))]),
                                                        
                                        np.array([sum_off_diag_FCD]),
                                          
                                        np.array([np.sum(FCDcorr.reshape(-1))]),
                                        np.array([np.var(FCDcorr.reshape(-1))]),
                                                       ))
            
            
            if item is 'PCA':
                        pca = PCA(n_components=3)
                        pca.fit(X_t)
                        trans_data=pca.fit(X_t).fit_transform(X_t).T
                        sum_stats_vec = np.concatenate((sum_stats_vec,
                                        pca.explained_variance_ratio_.reshape(-1),
                                        pca.explained_variance_.reshape(-1),
                                        pca.singular_values_.reshape(-1),
                                        pca.noise_variance_.reshape(-1),
                                        np.min(trans_data, axis=1), 
                                        np.max(trans_data, axis=1),
                                        np.mean(trans_data, axis=1), 
                                        np.var(trans_data, axis=1),
                                        np.std(trans_data, axis=1),
                                        skew(trans_data, axis=1), 
                                        kurtosis(trans_data, axis=1),
                                        np.array([np.sum(trans_data, axis=1)]).reshape(-1),
                                        np.sum(trans_data).reshape(-1),
                                                       ))

            if item is 'LLE':

                        trans_data = manifold.LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors, method='standard').fit_transform(X_t).T
                        sum_stats_vec = np.concatenate((sum_stats_vec,
                                        trans_data.reshape(-1),
                                        np.min(trans_data, axis=1), 
                                        np.max(trans_data, axis=1),
                                        np.mean(trans_data, axis=1), 
                                        np.array([np.sum(trans_data, axis=1)]).reshape(-1),
                                        np.sum(trans_data).reshape(-1),

                                                       ))
            if item is 'Hessian_LLE':

                        trans_data = manifold.LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors, method='hessian').fit_transform(X_t).T
                        sum_stats_vec = np.concatenate((sum_stats_vec,
                                        trans_data.reshape(-1),
                                        np.min(trans_data, axis=1), 
                                        np.max(trans_data, axis=1),
                                        np.mean(trans_data, axis=1), 
                                        np.array([np.sum(trans_data, axis=1)]).reshape(-1),
                                        np.sum(trans_data).reshape(-1),

                                                       ))

            if item is 'Modified_LLE':

                        trans_data = manifold.LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors, method='modified').fit_transform(X_t).T
                        sum_stats_vec = np.concatenate((sum_stats_vec,
                                        trans_data.reshape(-1),
                                        np.min(trans_data, axis=1), 
                                        np.max(trans_data, axis=1),
                                        np.mean(trans_data, axis=1), 
                                        np.array([np.sum(trans_data, axis=1)]).reshape(-1),
                                        np.sum(trans_data).reshape(-1),
                                                       ))

            if item is 'Isomap':

                        trans_data = manifold.Isomap(n_components=n_components, n_neighbors=n_neighbors).fit_transform(X_t).T
                        sum_stats_vec = np.concatenate((sum_stats_vec,
                                        trans_data.reshape(-1),
                                        np.min(trans_data, axis=1), 
                                        np.max(trans_data, axis=1),
                                        np.mean(trans_data, axis=1), 
                                        np.var(trans_data, axis=1),
                                        np.std(trans_data, axis=1),
                                        skew(trans_data, axis=1), 
                                        kurtosis(trans_data, axis=1),
                                        np.array([np.sum(trans_data, axis=1)]).reshape(-1),
                                        np.sum(trans_data).reshape(-1),
                                                       ))

            if item is 'MDS':

                        trans_data = manifold.MDS(n_components=n_components, max_iter=100, n_init=1).fit_transform(X_t).T
                        sum_stats_vec = np.concatenate((sum_stats_vec,
                                        trans_data.reshape(-1),
                                        np.min(trans_data, axis=1), 
                                        np.max(trans_data, axis=1),
                                        np.mean(trans_data, axis=1), 
                                        np.var(trans_data, axis=1),
                                        np.std(trans_data, axis=1),
                                        skew(trans_data, axis=1), 
                                        kurtosis(trans_data, axis=1),
                                        np.array([np.sum(trans_data, axis=1)]).reshape(-1),
                                        np.sum(trans_data).reshape(-1),
                                                       ))

            if item is 'Spectral_Embedding':

                        trans_data = manifold.SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors).fit_transform(X_t).T
                        #pca = PCA(n_components=2)
                        #PCA_trans_data = pca.fit_transform(trans_data)

                        sum_stats_vec = np.concatenate((sum_stats_vec,
                                        trans_data.reshape(-1),
                                        np.min(trans_data, axis=1), 
                                        np.max(trans_data, axis=1),
                                        np.mean(trans_data, axis=1), 
                                        np.var(trans_data, axis=1),
                                        np.std(trans_data, axis=1),
                                        skew(trans_data, axis=1), 
                                        kurtosis(trans_data, axis=1),
                                        np.sum(trans_data, axis=1),
                                        np.array([np.sum(trans_data, axis=1)]).reshape(-1),
                                        np.sum(trans_data).reshape(-1),
                                                       ))


            if item is 'TSNE':

                        trans_data = manifold.TSNE(n_components, init='pca', random_state=0).fit_transform(X_t).T
                        sum_stats_vec = np.concatenate((sum_stats_vec,
                                        trans_data.reshape(-1),
                                        np.min(trans_data, axis=1), 
                                        np.max(trans_data, axis=1),
                                        np.mean(trans_data, axis=1), 
                                        np.var(trans_data, axis=1),
                                        np.std(trans_data, axis=1),
                                        skew(trans_data, axis=1), 
                                        kurtosis(trans_data, axis=1),
                                        np.array([np.sum(trans_data, axis=1)]).reshape(-1),
                                        np.sum(trans_data).reshape(-1),
                                                       ))

            if item is 'SFA':

                        trans_data = sksfa.SFA(n_components=2).fit_transform(X_t).T
                        sum_stats_vec = np.concatenate((sum_stats_vec,
                                        trans_data.reshape(-1),
                                        np.min(trans_data, axis=1), 
                                        np.max(trans_data, axis=1),
                                        np.mean(trans_data, axis=1), 
                                        np.var(trans_data, axis=1),
                                        np.std(trans_data, axis=1),
                                        skew(trans_data, axis=1), 
                                        kurtosis(trans_data, axis=1),
                                        np.array([np.sum(trans_data, axis=1)]).reshape(-1),
                                        np.sum(trans_data).reshape(-1),
                                                       ))

            if item is 'UMAP':

                        trans_data = umap.UMAP().fit_transform(X_t).T
                        sum_stats_vec = np.concatenate((sum_stats_vec,
                                        trans_data.reshape(-1),
                                        np.min(trans_data, axis=1), 
                                        np.max(trans_data, axis=1),
                                        np.mean(trans_data, axis=1), 
                                        np.var(trans_data, axis=1),
                                        np.std(trans_data, axis=1),
                                        skew(trans_data, axis=1), 
                                        kurtosis(trans_data, axis=1),
                                        np.array([np.sum(trans_data, axis=1)]).reshape(-1),
                                        np.sum(trans_data).reshape(-1),
                                                        ))


    sum_stats_vec = sum_stats_vec[0:n_summary]        


    return sum_stats_vec
