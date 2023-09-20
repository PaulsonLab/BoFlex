#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 10:47:01 2023

@author: kudva.7
"""
import torch
import math
import torch
import gpytorch
import sys
import pandas as pd
from gpytorch.mlls import SumMarginalLogLikelihood
import scipy.stats as st
import numpy as np
from multiprocessing import Pool
from botorch.fit import fit_gpytorch_model
from scipy.stats import qmc

from FlexBO import FlexBO, twoD_plot





def poly_cons(theta,z):
    cons1 = -2*theta - 15 - 0.5*z*torch.cos(z)
    cons2 = (theta*theta)/3 + 4*theta - 5 - z*torch.cos(z)
    cons3 = -torch.square(theta - 4)/2 + 10 - z*torch.cos(z)
    # Stack each of the tensors
    cons_tensor = torch.stack((cons1.squeeze(),cons2.squeeze(),cons3.squeeze()))
    return cons_tensor


def circ_cons(theta,z):
    cons1 = (theta + 4)**2 + (z + 3)**2 - 9
    cons2 = (theta + 2)**2 + (z + 2)**2 + theta*z - 5
    # Stack each of the tensors
    cons_tensor = torch.stack((cons1.squeeze(),cons2.squeeze()))
    return cons_tensor



def HeatX(theta, z):
    with torch.no_grad():
        cons1 = (-2/3)*z[:,0] +  torch.add(theta[:,1],-350)
        cons2 = -theta[:,2] - 0.75*theta[:,0] + 0.5*z[:,0] + torch.add(-theta[:,1],1388.5)
        cons3 = -theta[:,2] - 1.5*theta[:,0] + z[:,0] + torch.add(-2*theta[:,1],2044)
        cons4 = -theta[:,2] - 1.5*theta[:,0] + z[:,0] - 2*theta[:,1] + torch.add(-2*theta[:,3],2830)
        cons5 = theta[:,2] + 1.5*theta[:,0] - z[:,0] + 2*theta[:,1] + torch.add(3*theta[:,3],-3153)
    # Stack each of the tensors
        cons_tensor = torch.stack((cons1,cons2,cons3,cons4,cons5))
    return cons_tensor







fun = circ_cons





if fun == poly_cons:

    
    # Parameters related to the polynomial network
    ntheta = 1;
    nz = 1;
    n_cons = 3
    
    z_min = torch.tensor([5])
    z_max = torch.tensor([20])
    
    theta_nominal = torch.tensor([-2.5])
    delta_theta = torch.tensor([3.75])
    
    delta = 1
    
    theta_min = theta_nominal - delta*delta_theta
    theta_max = theta_nominal + delta*delta_theta
    
    twoD_plot(fun, float(theta_min), float(theta_max), float(z_min), float(z_max))
    
elif fun == HeatX:
    
    # Parameters related to the HEX network
    ntheta = 4;
    nz = 1;
    n_cons = 5
    
    z_min = torch.tensor([0.15*300])
    z_max = torch.tensor([0.5*300])
    
    theta_nominal = torch.tensor([620,388,583,313])
    delta_theta = torch.tensor(1)
    
    delta = 4
    
    theta_min = theta_nominal - delta*delta_theta
    theta_max = theta_nominal + delta*delta_theta
    
else: 
    
    # Parameters related to the polynomial network
    ntheta = 1;
    nz = 1;
    n_cons = 3
    
    z_min = torch.tensor([-5])
    z_max = torch.tensor([4])
    
    theta_nominal = torch.tensor([-2])
    delta_theta = torch.tensor([1.5])
    
    delta = 1
    
    theta_min = theta_nominal - delta*delta_theta
    theta_max = theta_nominal + delta*delta_theta
    
    twoD_plot(fun, float(theta_min), float(theta_max), float(z_min), float(z_max))
    



######### Start the testing here
Ninit = 10
T = 61

start_process = False


if start_process:

    def process_element(seed_num):
        #seed_num = 1000
        # Generate initial data points
        torch.manual_seed(seed_num)
    
        with torch.no_grad():
            Init_theta = qmc.LatinHypercube(d=ntheta,seed = seed_num).random(n=Ninit)
            #Init_theta = qmc.LatinHypercube(d=ntheta,seed = seed_num).random(n=Ninit + T)
            Init_theta = qmc.scale(Init_theta,theta_min,theta_max)
            Init_theta = torch.tensor(Init_theta)
    
            Init_z = qmc.LatinHypercube(d=nz,seed = seed_num).random(n=Ninit)
            #Init_z = qmc.LatinHypercube(d=nz,seed = seed_num).random(n=Ninit + T)
            Init_z = qmc.scale(Init_z,z_min,z_max)
            Init_z = torch.tensor(Init_z)
    
            Ninit_vals = fun(Init_theta,Init_z)
    
        #FlexBO_obj = FlexBO(theta_min,theta_max,z_min,z_max,Init_theta[:Ninit],Init_z[:Ninit],Ninit_vals[:,:Ninit],sqrt_beta=2.,ntheta_grid = 5000,nz_grid = 500)
        FlexBO_obj = FlexBO(theta_min,theta_max,z_min,z_max,Init_theta,Init_z,Ninit_vals,sqrt_beta=2.,ntheta_grid = 1000,nz_grid = 500)
        UCB_LCB_dict = {"UCB_vals":[],"LCB_vals":[]}   
        
        
        
        theta_t = []
        z_t = []
    
    
        for i in range(T):
            if i == 0:
                FlexBO_obj.train()
            else:
                new_vals = fun(theta_t.unsqueeze(0),z_t.unsqueeze(0))
    
                if FlexBO_obj.ntheta == 1:
                    FlexBO_obj.update(theta_t.unsqueeze(0),z_t.unsqueeze(0),new_vals.unsqueeze(1))  
                else:
                    FlexBO_obj.update(theta_t.unsqueeze(0),z_t.unsqueeze(0),new_vals)  
    
                FlexBO_obj.train()
    
            
            print('Iteration Number',i)
            FlexBO_obj.eval()
            
            if i%10 == 0:
                FlexBO_obj.plot()
            
            try:
                MaxMinSoln = FlexBO_obj.MaxMin()
    
                theta_t = MaxMinSoln['max_theta_UCB']
                
                #theta_t = Init_theta[Ninit+i]
    
                print('theta_t = ',theta_t)
    
                # Update the Max Min UCB list and Max Min LCB list:
                UCB_LCB_dict["UCB_vals"].append(float(MaxMinSoln['MaxMin_UCB']))
                UCB_LCB_dict["LCB_vals"].append(float(MaxMinSoln['MaxMin_LCB']))
    
                # Quantile:
                z_t = FlexBO_obj.Min_LCB(theta_t)
    
                # Quantile max-min UCB
                #z_t = MaxMinSoln['min_z_LCB']
                #z_t = Init_z[Ninit+i] # For rand
    
                print('z_t = ',z_t)
    
                print('UCB:',MaxMinSoln['MaxMin_UCB'])
                
                # if i > 5:
                #      if UCB_LCB_dict['UCB_vals'][-2] < 0:
                #          break
                
                # print('LCB:',MaxMinSoln['MaxMin_LCB'])
                
            except:
                break
            
        return UCB_LCB_dict
    
    
    
    results = {}
    num_repeats = 1
    
    for t in range(num_repeats):
        print('Run Number',t)
        results[t] = process_element(t*1000)
        print(results[t])
        
    df_LCB_list=[]
    df_UCB_list=[]
    
    # Create a list constaining all solutions
    for i in range(num_repeats):
        df_UCB = pd.DataFrame({'UCB':results[i]["UCB_vals"]})
        df_UCB_list.append(df_UCB)
        df_LCB = pd.DataFrame({'LCB':results[i]["LCB_vals"]})
        df_LCB_list.append(df_LCB)
        
    # Create a pandas dataframe
    combined_df_UCB = pd.concat(df_UCB_list,axis=1)
    combined_df_UCB.to_csv(r"HeatX_FlexBORand_UCB_init5.csv") 
    combined_df_LCB = pd.concat(df_LCB_list,axis=1)
    combined_df_LCB.to_csv(r"HeatX_FlexBORand_LCB_init5.csv") 
    
