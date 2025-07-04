# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 15:57:38 2016
@author: jmaidana
@author: porio
"""
from __future__ import division
import numpy as np
import matplotlib.pylab as plt

def extract_FCD(data,wwidth=1000,maxNwindows=100,olap=0.9,coldata=False,mode='corr'):
    """
    Functional Connectivity Dynamics from a collection of time series
    Parameters:
    -----------
    data : array-like
        2-D array of data, with time series in rows (unless coldata is True)
    wwidth : integer
        Length of data windows in which the series will be divided, in samples
    maxNwindows : integer
        Maximum number of windows to be used. wwidth will be increased if necessary
    olap : float between 0 and 1
        Overlap between neighboring data windows, in fraction of window length
    coldata : Boolean
        if True, the time series are arranged in columns and rows represent time
    mode : 'corr' | 'psync' | 'plock' | 'tdcorr'
        Measure to calculate the Functional Connectivity (FC) between nodes.
        'corr' : Pearson correlation. Uses the corrcoef function of numpy.
        'psync' : Pair-wise phase synchrony.
        'plock' : Pair-wise phase locking.
        'tdcorr' : Time-delayed correlation, looks for the maximum value in a cross-correlation of the data series 
        
    Returns:
    --------
    FCDmatrix : numpy array
        Correlation matrix between all the windowed FCs.
    CorrVectors : numpy array
        Collection of FCs, linearized. Only the lower triangle values (excluding the diagonal) are returned
    shift : integer
        The distance between windows that was actually used (in samples)
            
        
    """
    
    if olap>=1:
        raise ValueError("olap must be lower than 1")
    if coldata:
        data=data.T    
    
    all_corr_matrix = []
    lenseries=len(data[0])
    
    Nwindows=min(((lenseries-wwidth*olap)//(wwidth*(1-olap)),maxNwindows))
    shift=int((lenseries-wwidth)//(Nwindows-1))
    if Nwindows==maxNwindows:
        wwidth=int(shift//(1-olap))
    
    indx_start = range(0,(lenseries-wwidth+1),shift)
    indx_stop = range(wwidth,(1+lenseries),shift)
         
    nnodes=len(data)

    for j1,j2 in zip(indx_start,indx_stop):
        aux_s = data[:,j1:j2]
        if mode=='corr':
            corr_mat = np.corrcoef(aux_s) 
        elif mode=='psync':
            corr_mat=np.zeros((nnodes,nnodes))
            for ii in range(nnodes):
                for jj in range(ii):
                    corr_mat[ii,jj]=np.mean(np.abs(np.mean(np.exp(1j*aux_s[[ii,jj],:]),0)))
        elif mode=='plock':
            corr_mat=np.zeros((nnodes,nnodes))
            for ii in range(nnodes):
                for jj in range(ii):
                    corr_mat[ii,jj]=np.abs(np.mean(np.exp(1j*np.diff(aux_s[[ii,jj],:],axis=0))))
        elif mode=='tdcorr':
            corr_mat=np.zeros((nnodes,nnodes))
            for ii in range(nnodes):
                for jj in range(ii):
                    maxCorr=np.max(np.correlate(aux_s[ii,:],aux_s[jj,:],mode='full')[wwidth//2:wwidth+wwidth//2])
                    corr_mat[ii,jj]=maxCorr/np.sqrt(np.dot(aux_s[ii,:],aux_s[ii,:])*np.dot(aux_s[jj,:],aux_s[jj,:]))
        all_corr_matrix.append(corr_mat)
        
    corr_vectors=np.array([allPm[np.tril_indices(nnodes,k=-1)] for allPm in all_corr_matrix])
    
    CV_centered=corr_vectors - np.mean(corr_vectors,-1)[:,None]
    
    
    return np.corrcoef(CV_centered),corr_vectors,shift


##############################################################################


def compute_fcd(ts, win_len=30, win_sp=1):
    """Compute dynamic functional connectivity.
    Arguments:
        ts:      time series of shape [time,nodes]
        win_len: sliding window length in samples
        win_sp:  sliding window step in samples
    Returns:
        FCD: matrix of functional connectivity dynamics
        fcs: windowed functional connectivity matrices
        speed_fcd: rate of changes between FC frames
    """
    n_samples, n_nodes = ts.shape 
    fc_triu_ids = np.triu_indices(n_nodes, 1) #returns the indices for upper triangle
    n_fcd = len(fc_triu_ids[0]) #
    fc_stack    = []
    speed_stack = []
    fc_prev = []


    for t0 in range( 0, ts.shape[0]-win_len, win_sp ):
        t1=t0+win_len
        fc = np.corrcoef(ts[t0:t1,:].T)
        fc = fc[fc_triu_ids]
        fc_stack.append(fc)
        if t0 > 0 :
            corr_fcd  = np.corrcoef([fc,fc_prev])[0,1]
            speed_fcd = 1-corr_fcd
            speed_stack.append(speed_fcd)
            fc_prev   = fc
        else:
            fc_prev   = fc
            

    fcs      = np.array(fc_stack)
    speed_ts = np.array(speed_stack)
    FCD = np.corrcoef(fcs)
    return FCD, fcs, speed_ts


##############################################################################

def compute_fcd_filt(ts, mat_filt, win_len=30, win_sp=1):
    """Compute dynamic functional connectivity with SC filtering
    Arguments:
        ts:      time series of shape [time,nodes]
        win_len: sliding window length in samples
        win_sp:  sliding window step in samples
    Returns:
        FCD: matrix of functional connectivity dynamics
        fcs: windowed functional connectivity matrices
        speed_fcd: rate of changes between FC frames
    """
    n_samples, n_nodes = ts.shape 
    fc_triu_ids = np.triu_indices(n_nodes, 1) #returns the indices for upper triangle
    n_fcd = len(fc_triu_ids[0]) #
    fc_stack    = []
    speed_stack = []


    for t0 in range( 0, ts.shape[0]-win_len, win_sp ):
        t1=t0+win_len
        fc = np.corrcoef(ts[t0:t1,:].T)
        fc = fc*(fc>0)*(mat_filt)
        fc = fc[fc_triu_ids]
        fc_stack.append(fc)
        if t0 > 0 :
            corr_fcd  = np.corrcoef([fc,fc_prev])[0,1]
            speed_fcd = 1-corr_fcd
            speed_stack.append(speed_fcd)
            fc_prev   = fc
        else:
            fc_prev   = fc
            

    fcs      = np.array(fc_stack)
    speed_ts = np.array(speed_stack)
    FCD = np.corrcoef(fcs)
    return FCD, fcs, speed_ts