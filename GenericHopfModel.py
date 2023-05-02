#!/usr/bin/env python3
"""
@author: meysamhashemi  INS Marseille

"""
import os
import sys
import numpy as np
import numba
from numba import jit


############################################################################################################
@jit(nopython=True)
def GenericHopfBOLD(eta, omega, init_conditions, sigma, G, SC, dt, dt_bold, Tmax, ds):

    epsilon = 0.5
    itaus = 1.25
    itauf = 2.5
    itauo = 1.02040816327
    ialpha = 5.
    Eo = 0.4
    V0 = 4.
    k1 = 2.77264
    k2 = 0.572
    k3 = -0.43

    ## nt: number of data points per region (at source), and nn: number of brain regions
    
    ts = np.arange(0, Tmax + dt, dt)
    nt = int(len(ts))
    nn = int(SC.shape[0])
   
    # nt_bold: number  of data points at BOLD (measurement) level following downsampling ds
    nt_bold=int(nt/ds)
    
    ## global scaling parameter
    G=100*G
    
    # state variables
    x = np.zeros(nn)
    b = np.zeros((nn, nt))
    y = np.zeros(nn)
    z = np.array([0.]*nn + [1.]*3*nn)
    BOLD= np.zeros((nn, nt_bold))


    # initial conditions (similar value for all regions)
    x_init, y_init = init_conditions[0], init_conditions[1]
    x[:] = x_init
    y[:] = y_init

    for i in range(nn):
        b[i, 0] = V0*(k1 - k1*z[3*nn+i] + k2-k2 *(z[3*nn+i]/z[2*nn+i]) + k3-k3*z[2*nn+i])

    ## integrate SDE using Euler scheme and for loop to accelerate by numba 
                
    for t in range(nt-1):
        for i in range(nn):
            gx, gy = 0.0, 0.0
            for j in range(nn):
                gx = gx + SC[j, i]*(x[j] - x[i])
                gy = gy + SC[j, i]*(y[j] - y[i])
            dx = (x[i]*(eta[i]-(x[i]*x[i])-(y[i]*y[i]))) - (omega[i]*y[i]) + (G*gx)
            dy = (y[i]*(eta[i]-(x[i]*x[i])-(y[i]*y[i]))) + (omega[i]*x[i]) + (G*gy)
            dz0 = epsilon*x[i]-itaus*z[i]-itauf*(z[nn+i]-1)
            dz1 = z[i]
            dz2 = itauo*(z[nn+i]-z[2*nn+i]**ialpha)
            dz3 = itauo * (z[nn+i]*(1-(1-Eo)**(1/z[nn+i]))/Eo -(z[2*nn+i]**ialpha)*z[3*nn+i]/z[2*nn+i])

            x[i] = x[i] + dt*dx + np.sqrt(dt) * sigma * np.random.randn()
            y[i] = y[i] + dt*dy + np.sqrt(dt) * sigma * np.random.randn()

            z[i] = z[i] + dt * dz0
            z[nn+i] = z[nn+i] + dt*dz1
            z[2*nn+i] = z[2*nn+i] + dt * dz2
            z[3*nn+i] = z[3*nn+i] + dt * dz3
            b[i, t+1] = V0*(k1 - k1*z[3*nn+i] + k2-k2 *(z[3*nn+i]/z[2*nn+i]) + k3-k3*z[2*nn+i])
            
    BOLD=(b[:, ::ds])
    return BOLD.flatten()

    ############################################################################################################
@jit(nopython=True) 
def GenericHopfmodel(eta, omega, init_conditions, sigma, G, SC, dt, Tmax):
    
    ## nt: number of data points per region and nn: number of brain regions
    ts = np.arange(0, Tmax + dt, dt)
    nt = int(len(ts))
    nn=SC.shape[0]
    
    ## global scaling parameter
    G=100*G

    # state variables
    x = np.zeros((nn, nt))  
    y = np.zeros((nn, nt))  

    ## initial conditions (similar value for all regions)
    x_init, y_init=init_conditions[0], init_conditions[1]
    x[:, 0] = x_init
    y[:, 0] = y_init
        
    ## integrate SDE using Euler scheme and for loop to accelerate by numba 
    for t in range(nt-1):    
        for i in range(nn):
                gx, gy= 0.0, 0.0
                for j in range(nn):
                        gx = gx + SC[j,i]*(x[j,t] - x[i,t])
                        gy = gy + SC[j,i]*(y[j,t] - y[i,t])
                dx =(x[i,t]*(eta[i]-(x[i,t]*x[i,t])-(y[i,t]*y[i,t]))) -(omega[i]*y[i,t]) +(G*gx)
                dy =(y[i,t]*(eta[i]-(x[i,t]*x[i,t])-(y[i,t]*y[i,t]))) +(omega[i]*x[i,t]) +(G*gy)
                x[i,t+1] = x[i,t] + dt*dx + np.sqrt(dt) * sigma * np.random.randn() 
                y[i,t+1] = y[i,t] + dt*dy + np.sqrt(dt) * sigma * np.random.randn() 
                
                
    return x.reshape(-1)  


############################################################################

# @jit(nopython=True) 
# def BalloonWindkessel(x_source, dt_bold):

#     nt_=int(x_source.size);
    
#     epsilon=0.5;
#     itaus  = 1.25;
#     itauf  = 2.5;
#     itauo  = 1.02040816327;
#     ialpha = 5.;
#     Eo     = 0.4;
#     V0     = 4.;
#     k1     = 2.77264;
#     k2     = 0.572;
#     k3     = -0.43;

#     x0 = np.array([0., 1., 1., 1.])
#     x = np.zeros((nt_,4));
#     b = np.zeros((nt_,1));

#     x[0,:] = x0;
#     b[0]  = V0*(k1- k1*x[0,3] + k2-k2*(x[0,3]/x[0,2]) + k3-k3*x[0,2]);

#     for t in range(nt_-1): 
#         x[t+1,0] = x[t,0] + dt_bold*( epsilon*x_source[t]-itaus*x[t,0]-itauf*(x[t,1]-1) );
#         x[t+1,1] = x[t,1] + dt_bold*x[t,0];
#         x[t+1,2] = x[t,2] + dt_bold*itauo*(x[t,1]-x[t,2]**ialpha);
#         x[t+1,3] = x[t,3] + dt_bold*itauo*(x[t,1]*(1-(1-Eo)**(1/x[t,1]))/Eo - (x[t,2]**ialpha)*x[t,3]/x[t,2]);
#         b[t+1]  = V0*(k1- k1*x[t+1,3] + k2-k2*(x[t+1,3]/x[t+1,2]) + k3-k3*x[t+1,2] );

#     return b

############################################################################

# @jit(nopython=True) 
# def GenericHopfBOLD(eta, omega, init_conditions, sigma, G, SC, dt, dt_bold, Tmax, ds):
      
#     ts = np.arange(0, Tmax + dt, dt)
#     nt = int(len(ts))
#     nn=SC.shape[0]
    
#     Sim=GenericHopfmodel(eta, omega, init_conditions, sigma, G, SC, dt, Tmax)
#     neural_act=Sim.reshape(nn, int(Sim.shape[0]/nn))

#     fmri = np.zeros((nt,nn,1));
#     B =np.zeros((int(nt/ds),1));
    
#     nt_bold=fmri[::ds,:].shape[0]
#     BOLD = np.zeros((nn, nt_bold))  
    
#     for node in range(nn):
#         B = BalloonWindkessel(neural_act.T[:,node], dt_bold);
#         fmri[:,node] = B;
#         BOLD[node,:]=fmri[::ds,node,0].T
        
#     return BOLD.reshape(-1) 