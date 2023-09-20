
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

from FlexBO import BoFlex, twoD_plot, active_corners, plot_theta_projection, true_chi
import random
import matplotlib.pyplot as plt



def process_element(seed_num,algo_type,beta,Ninit,T,ntheta_grid,nz_grid):
    """
    Parameters
    ----------
    seed_num : Seed number for a random initialization -- int
    algo_type : Baselines : 'BoFlex','BoFlexMaxMin', 'BoFlexRand', 'ARBO'--string
    beta : Exploration parameter for ucb: mean + beta*std -- float
    Ninit : Number of initial samples -- int
    T : Budget for additional samples -- int
    ntheta_grid: Number of sobol samples in set \Theta
    nz_grid: Number of sobol samples in set Z

    Returns
    -------
    UCB_LCB_dict : A dictionary consisting of estimated upper,lower bound of \chi
    for each repeat                                     -- dict

    """
    
    soboleng = torch.quasirandom.SobolEngine(dimension=ntheta, seed = 1000)   
    soboleng2 = torch.quasirandom.SobolEngine(dimension=nz, seed = 1000)      
    
    # Generate Vertex of hyper rectangle of theta + Grid for max min eval:
    theta_grid = soboleng.draw(ntheta_grid)
     
    #ntheta_grid = ntheta_grid + 2**ntheta

    # Create a list for z
    z_grid = soboleng2.draw(nz_grid)

    theta_grid_eval = theta_grid.repeat_interleave(nz_grid,dim = 0)
    z_grid_eval = z_grid.repeat(ntheta_grid,1)

    # We will use the below points for grid based evaluation
    test_x = torch.cat((theta_grid_eval,z_grid_eval),dim = 1)    
    
    
    random.seed(seed_num)
    
    # Generate some random numbers
    if algo_type == 'BoFlexRand':
        random_numbers_1 = [random.randint(0, ntheta_grid*nz_grid) for _ in range(Ninit+T)]
    else:
        random_numbers_1 = [random.randint(0, ntheta_grid*nz_grid) for _ in range(Ninit)]
    
    
    selected = test_x[random_numbers_1]
    
    Init_theta = (selected[:,0].unsqueeze(-1)*(theta_max - theta_min) + theta_min).type(torch.DoubleTensor)
    
    Init_z = (selected[:,1].unsqueeze(-1)*(z_max - z_min) + z_min).type(torch.DoubleTensor)
    
    ##########################
    Ninit_vals = torch.zeros(n_cons,Ninit)
    
    for i in range(Ninit):
        Ninit_vals[:,i] = fun(Init_theta[i],Init_z[i])
    
    Ninit_vals = Ninit_vals.type(torch.DoubleTensor)
    
    print('Initial theta values', Init_theta)
    print('Initial z values', Init_z)
    print(' Get the function values', Ninit_vals)
    
    if algo_type == 'ARBO':
        Ninit_vals = (Ninit_vals.max(dim = 0).values).unsqueeze(dim = -1)
    
    
    # If you want to print the two D test functions -- Warning - dont use for bioreactor case
    # will take ~25 years to generate the plot!    
    
    # print('Plotting true function')
    #twoD_plot(fun, float(theta_min), float(theta_max), float(z_min), float(z_max))
    # # # # plt.scatter(Init_theta, Init_z, color = 'red')
    #plt.show()
    
    
   
    if algo_type == 'ARBO':
        FlexBO_obj = BoFlex(theta_min,theta_max,z_min,z_max,Init_theta,Init_z,Ninit_vals,sqrt_beta=beta,ntheta_grid = ntheta_grid,nz_grid = nz_grid, ARBO = True)
    elif algo_type == 'BoFlexRand': 
        FlexBO_obj = BoFlex(theta_min,theta_max,z_min,z_max,Init_theta[:Ninit],Init_z[:Ninit],Ninit_vals[:,:Ninit],sqrt_beta=beta,ntheta_grid = ntheta_grid,nz_grid = nz_grid, ARBO = False)
    else:
        FlexBO_obj = BoFlex(theta_min,theta_max,z_min,z_max,Init_theta,Init_z,Ninit_vals,sqrt_beta=beta,ntheta_grid = ntheta_grid,nz_grid = nz_grid, ARBO = False)
    
    # If you want to find the true solution based on the discretised space
    
    # Find the true flexibility test solution
    #maxmin = true_chi(fun, FlexBO_obj)  
    
    UCB_LCB_dict = {"UCB_vals":[],"LCB_vals":[]}
    
    theta_t = []
    z_t = []

    
    for i in range(T):
        if i == 0:
            
            if algo_type == 'ARBO':
                FlexBO_obj.train_ARBO()
            else:
                FlexBO_obj.train() #train_hyp = False)
            
            
            
        else:
            new_vals = fun(theta_t.unsqueeze(0),z_t.unsqueeze(0))
            
            # ARBO Case
            if algo_type == 'ARBO':
                new_vals = new_vals.max(dim = 0).values
            
            if FlexBO_obj.ARBO:
                FlexBO_obj.update(theta_t.unsqueeze(0),z_t.unsqueeze(0),new_vals)
            else:    
                if FlexBO_obj.ntheta == 1:
                    FlexBO_obj.update(theta_t.unsqueeze(0),z_t.unsqueeze(0),new_vals.unsqueeze(1))   
                else:
                    FlexBO_obj.update(theta_t.unsqueeze(0),z_t.unsqueeze(0),new_vals)  
            
           
            
        if algo_type == 'ARBO':
            FlexBO_obj.train_ARBO()
        else:
            FlexBO_obj.train()
            #FlexBO_obj.train(train_hyp = False)
            
        print('Iteration Number',i)
        FlexBO_obj.eval()
        
        # In you want to plot the upper and lower worst case constraint like Fig 1
        # if i%3 == 0:
        #     FlexBO_obj.plot()
            
        
        #try:
        
        if algo_type == 'ARBO':
            # ARBO
            MaxMinSoln = FlexBO_obj.MaxMin_ARBO()
        else:
            MaxMinSoln = FlexBO_obj.MaxMin()
        
        
        if algo_type == 'BoFlexRand':
            theta_t = Init_theta[Ninit+i]
        else:
            theta_t = MaxMinSoln['max_theta_UCB']
        
        #

        print('theta_t = ',theta_t)                                                                

        # Update the Max Min UCB list and Max Min LCB list:
        UCB_LCB_dict["UCB_vals"].append(float(MaxMinSoln['MaxMin_UCB']))
        UCB_LCB_dict["LCB_vals"].append(float(MaxMinSoln['MaxMin_LCB']))

        # Quantile:
        if algo_type == 'BoFlex':
            z_t = FlexBO_obj.Min_LCB(theta_t)

        # Quantile max-min UCB
        if algo_type == 'BoFlexMaxMin':
            z_t = MaxMinSoln['min_z_LCB']
        if algo_type == 'BoFlexRand':
            z_t = Init_z[Ninit+i] # For rand
        
        # ARBO
        if algo_type == 'ARBO':
            z_t = FlexBO_obj.Min_LCB_ARBO(theta_t)

        print('z_t = ',z_t)

        print('UCB:',MaxMinSoln['MaxMin_UCB'])
        print('LCB:',MaxMinSoln['MaxMin_LCB'])
        
        # If you want to plot the projected upper bound like Fig 1
        # if i%3 == 0:                    
        #     plot_theta_projection(fun, FlexBO_obj,MaxMinSoln['max_theta_UCB'],MaxMinSoln['max_theta_LCB'],MaxMinSoln['MaxMin_UCB'],MaxMinSoln['MaxMin_LCB'])
        
        
        # except:
        #     print('Something went wrong!, breaking !!')
        #     break
            
        
    return UCB_LCB_dict 


def circ_cons(theta,z):
    """
    This is the simple 
    """
    
    cons1 = (theta + 4)**2 + (z + 3)**2 - 9
    cons2 = (theta + 2)**2 + (z + 2)**2 + (theta*z) - 5
    #cons3 = (theta - 2)**2  + (z - 1)**2 - 0.2
    # Stack each of the tensors
    #cons3 = (theta - 1)**2 + (z - 1)**2 
    cons_tensor = torch.stack((cons1.squeeze(),cons2.squeeze()))
    return cons_tensor



def HeatX(theta, z):
    
    """
    Grid: 10,000 X 800
    Ninit = 10
    """
    try:
        k = theta.size()[1]
        a = 100
    except:
        theta = theta.unsqueeze(0)
        z = z.unsqueeze(0)
        a = 1
    
        
    
    T1 = theta[:,0]
    T3 = theta[:,1]
    T5 = theta[:,2]
    T8 = theta[:,3]
    
    Qc = z[:,0]
    
    # Write the constraints below:
    
    cons1 = -0.67*Qc + T3 - 350 
    cons2 = -T5 - 0.75*(1 + 0.02*torch.cos(T1/4))*T1 + 0.5*Qc - T3 + 1388.5 
    cons3 = -T5 -1.5*(1 + 0.01*torch.cos(T1/4))*T1 + Qc -2*T3 + 2044 
    cons4 = -T5 -1.5*(1 + 0.01*torch.cos(T1/4))*T1 + Qc -2*T3 -2*T8 +2830 
    cons5 = T5 + 1.5*(1 + 0.01*torch.cos(T1/4))*T1 - Qc + 2*T3 + 3*T8 -3153 
    
     # Stack each of the tensors
    if a == 1:
        cons_tensor = torch.stack((cons1,cons2,cons3,cons4,cons5)).squeeze()   
    else:
        cons_tensor = torch.stack((cons1,cons2,cons3,cons4,cons5))  
    
    return cons_tensor



def HeatXNL(theta,z):
    """
    Non linear heat Exchanger case - two dimensions
    
    1000 X 1000
    
    Cases:
    Case 1: theta : 1 - 1.5; Ninit = 5
    Case 2: theta: 0.55 - 1.05; Ninit = 5
    """
    
    cons1 = -25 + z*((1/theta) - 0.5) + 10/theta  
    cons2 = -190 + (10/theta) + (z/theta)
    cons3 = -270 + (250/z) + (z/theta)
    cons4 = 260 - (250/theta) - (z/theta)
    
    
    cons_tensor = torch.stack((cons1.squeeze(),cons2.squeeze(),cons3.squeeze(),cons4.squeeze()))
    
    return cons_tensor


def bioreactor(theta,z):
    
    """
    Bio Reactor studies by Chen et al. MATLAB code
    Requirements:
    1) DFBAlab + Gurobi/Yalmip
    2) Refer: https://pubmed.ncbi.nlm.nih.gov/26932448/
        for othe rrequirements
    """
    import matlab.engine
    eng = matlab.engine.start_matlab()
    a = eng.bioreactor_consOneD(float(theta),float(z))
    a = (torch.tensor(a)).squeeze(0)
    ethanol_acet = torch.tensor([13.5 - a[0],a[1] - 8.5])
    
    return ethanol_acet



if __name__ == "__main__":

    # Select which function is to be tested    
    fun = HeatXNL
    ##### Start the testing here
    Ninit = 10
    T = 40    
    # Set the grid numbers: 
    ntheta_grid = 10000
    nz_grid = 10000
    
        
    if fun == HeatX:
        """
        Heat exchanger with 4 uncertain variables (Example 2)
        """
        
        # Parameters related to the HEX network
        ntheta = 4;
        nz = 1;
        n_cons = 5
        
        z_min = torch.tensor([0.01*300])
        z_max = torch.tensor([0.5*300])
        
        theta_nominal = torch.tensor([620,388,583,313])
        delta_theta = torch.tensor(1)
        
        delta = 8
        
        theta_min = theta_nominal - delta*delta_theta
        theta_max = theta_nominal + delta*delta_theta
        
    elif fun == circ_cons: 
        """
        Illustrative example from figure 1
        """
        
        # Parameters related to the polynomial network
        ntheta = 1;
        nz = 1;
        n_cons = 2
        
        z_min = torch.tensor([-3])
        z_max = torch.tensor([0])
        
        theta_nominal = torch.tensor([-2])
        delta_theta = torch.tensor([1.5])
        
        delta = 1.
        
        theta_min = theta_nominal - delta*delta_theta
        theta_max = theta_nominal + delta*delta_theta
        
        #twoD_plot(fun, float(theta_min), float(theta_max), float(z_min), float(z_max))
        
        
    elif fun == HeatXNL:
        """
        Heat exhanger with 1 uncertain variable (Example 1)
        """
        
        ntheta = 1
        nz = 1
        
        n_cons = 4
        
        z_min = torch.tensor([1])
        z_max = torch.tensor([0.333*300])
        
        # theta_min = torch.tensor([0.55])
        # theta_max = torch.tensor([1.05 ])
        
        theta_min = torch.tensor([0.6])
        theta_max = torch.tensor([1.4 ])
    
    elif fun == bioreactor:
        """
        Bio reactor case study (Example 3)
        
        """
        
        ntheta = 1
        nz = 1
        
        n_cons = 2
        
        z_min = torch.tensor([10.])
        z_max = torch.tensor([14.])
        
        theta_nominal = torch.tensor([310.25])
    
        delta_theta = torch.tensor([2.])
    
        theta_min = theta_nominal - delta_theta
        theta_max = theta_nominal + delta_theta       
        
        
    
        
    # Below loop saves data from elements
    results = {}
    num_repeats = 1
    algo_type = 'BoFlex' # Other baselines: 'BoFlexMaxMin', 'BoFlexRand', 'ARBO'
    beta = 3
    
    for t in range(num_repeats):
        print('Run Number',t)
        results[t] = process_element((t)*1000,algo_type,beta = beta,Ninit = Ninit, T = T,ntheta_grid=ntheta_grid,nz_grid = nz_grid)
        print(results[t])
 
       


