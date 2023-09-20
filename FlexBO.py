#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 10:40:22 2023

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
import copy
import matplotlib.pyplot as plt


# Create a function to generate active corners:

def active_corners(theta_min,theta_max):
    """
    This code is mainly used to generate all corners of box constraints.
    Will be incorporated in the Object that will give us the bounds.

    inputs:
    theta_min -- N dimensional tensor
    theta_max -- N dimensional tensor

    output -- 2^(N) X N dimensional tensor
    """
    size_t1 = torch.Tensor.size(theta_min)
    size_t2 = torch.Tensor.size(theta_max)

    # Show error if dimensions dont match:
    if size_t1 != size_t2:
        sys.exit('The dimensions of bounds dont match: Please enter valid inputs')

    val = size_t1[0]
    size_out = 2**(val)
    output = torch.zeros(size_out,val)
    output_iter = torch.zeros(size_out)

    for i in range(val):
        div_size = int(size_out/(2**(i+1)))
        divs = int(size_out/div_size)
        div_count = 0
        for j in range(divs):
            if bool(j%2):
                output_iter[div_count:div_count+div_size] = theta_min[i]*torch.ones(div_size)
            else:
                output_iter[div_count:div_count+div_size] = theta_max[i]*torch.ones(div_size)
            div_count = div_count + div_size
        output[:,i] = output_iter
    return output




# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5)) 
        #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) 
        #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def trainGPModel_ARBO(train_x,train_y):
    """
    Trains an exact GP model for the ARBO procedure
    Parameters. A typical gpytorch class
    """
    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    return model, likelihood


def trainGPModel(train_x,train_y,hyp_tune,training_iter = 1000):
    """
    Parameters
    ----------
    train_x : training features -- torch double
    train_y : training function values -- torch double
    hyp_tune : This is just for research purposes -- boolean
    training_iter : 

    Returns
    -------
    model : gpytorch model -- gpytorch class
    likelihood : gpytorch likelhood -- gpytorch class

    """
    # initialize likelihood and model
    n_cons = torch.Tensor.size(train_y, dim = 0)

    if n_cons == 1:
        # initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_x, train_y, likelihood)
    else:
        # If more than one constraint
        likelihoodnum = [None] * n_cons
        model_num = [None] * n_cons
        mod_lik = [None] * n_cons

        for i in range(n_cons):
            #likelihoodnum[i] = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(lower_bound=0.45,upper_bound=0.5))
            likelihoodnum[i] = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-4))
            model_num[i] =  ExactGPModel(train_x, train_y[i], likelihoodnum[i])
            mod_lik[i] = model_num[i].likelihood

        # Initialize likelihood list and model list
        model = gpytorch.models.IndependentModelList(*model_num)
        likelihood = gpytorch.likelihoods.LikelihoodList(*mod_lik)
    
    if n_cons == 1:
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    else:
        mll = SumMarginalLogLikelihood(likelihood, model)


    # Find optimal model hyperparameters
    if hyp_tune:
        model.train()
        likelihood.train()
        
        # Using BO torch optimizer in this run
        BO_torch_optim = 1
    
        if BO_torch_optim:
    
            #try:        
                fit_gpytorch_model(mll)
                #warnings.simplefilter('error')
                
            # except:
            #     print(' Unable to optimize hyperparameters')
            #     sys.exit()
    
            
        else:
            """
            Precomputed Heat Exchanger hyper parameters to verify proposed theory
            """
            # Note: Below hyper parameters are poor - DO NOT USE!
            l = [torch.tensor([[493.6208]]), torch.tensor([[181.0259]]), torch.tensor([[423.5137]]), torch.tensor([[385.3452]]), torch.tensor([[418.3785]])]
            sigma_f = [torch.tensor(40233.3203), torch.tensor(27180.5195), torch.tensor(44928.7227), torch.tensor(43138.1875), torch.tensor(44229.1172)]
            noise = [torch.tensor([0.0005]), torch.tensor([0.0005]), torch.tensor([0.0003]), torch.tensor([0.0003]), torch.tensor([0.0003])]
            
            with torch.no_grad():
                for i in range(n_cons):
                    model.models[i].covar_module.base_kernel.lengthscale  = l[i]
                    model.models[i].covar_module.outputscale = sigma_f[i]
                    model.models[i].likelihood.noise = noise[i]           

    return model, likelihood

def twoD_plot(fun,theta_min,theta_max,z_min,z_max):
    """
    Generates a 5000 X 5000 countour plot of the true function
    Not recommended for real expensive functions

    Parameters
    ----------
    fun : true function -- python function
    theta_min :    -- torch tensor
    theta_max :  -- torch tensor
    z_min :  -- torch tensor
    z_max :  -- torch tensor

    Returns
    -------
    A countor plot

    """
    
    n = 5000
    
    x_axis = torch.linspace(theta_min,theta_max,n)
    y_axis = torch.linspace(z_min,z_max,n)
    
    x,y = torch.meshgrid(x_axis,y_axis)
    
    # reshape x and y to match the input shape of fun
    xy = torch.stack([x.flatten(), y.flatten()], axis=1)
    
    inner_max = (torch.max(fun(xy[:,0],xy[:,1]),axis = 0)).values
    #inner_max = fun(xy[:,0],xy[:,1])
    
    c_plot = torch.reshape(inner_max,x.size())
    
    
    fig, ax = plt.subplots(1, 1) 
    cmap  = plt.set_cmap("jet")
    #cmap.set_under('navy')
    
    #vmin = -2
    #vmax = 5 #For the simple case study example c_plot.max() otherwise
    
    vmin = -100
    vmax = 100
    
    levels = torch.linspace(vmin,vmax,100)
    
    
    
    contour_plot = ax.contourf(x,y,c_plot, levels = levels, cmap = cmap,extend = 'both')
    #contour_plot = ax.contourf(x,y,c_plot, cmap = cmap,extend = 'both')
    
    #fig.colorbar(contour_plot)
    #cbar.set_label('Value')
    
    
    
    # Get the countour
    
    values = fun(xy[:, 0], xy[:, 1])
    epsilon = 0.001
    
    a = inner_max < epsilon
    b = inner_max > -1*epsilon
    
    c = torch.logical_and(a, b)
    near_zero_points = xy[c]
    #ax.scatter(near_zero_points[:, 0], near_zero_points[:, 1], color='black', linewidth=0.5)
    
    
    
    ax.set_xlabel(r'$\theta$', fontsize = 20)
    ax.set_ylabel('z', fontsize = 20)
    
    #fig.colorbar(contour_plot,format='%.1f')
    plt.show()
    

def true_chi(fun,obj):
    """
    Obtain the true chi values based on the selected decision space

    Parameters
    ----------
    fun : true function -- python function
    obj : the BoFlex Object

    Returns
    -------
    soln : dict with value and args

    """
    
    rand_theta_grid = obj.theta_grid*(obj.theta_max - obj.theta_min) + obj.theta_min
    rand_z_grid = obj.z_grid*(obj.z_max - obj.z_min) + obj.z_min
    
    theta_grid_eval = rand_theta_grid.repeat_interleave(obj.nz_grid,dim = 0)
    z_grid_eval = rand_z_grid.repeat(obj.ntheta_grid,1)
    
    # We will use the below points for grid based evaluation
    
    a = fun(theta_grid_eval,z_grid_eval)
    true_fun_vals = (torch.max(a,0)).values
    
    z_list1 = torch.zeros(obj.ntheta_grid,obj.nz)
    val_list1 = torch.zeros(obj.ntheta_grid)
    
    
    for i in range(obj.ntheta_grid):
        val = torch.min(true_fun_vals[i*obj.nz_grid:(i+1)*obj.nz_grid], dim = 0)
        index_val = int(val.indices)
        z_list1[i,:] = rand_z_grid[index_val,:]
        val_list1[i] = val.values
        
        # Solve the outer max problem
        val = torch.max(val_list1,0)
        argmax_theta = rand_theta_grid[val.indices,:]
        argmin_z = z_list1[val.indices,:]
        val = val.values
    
        soln = {'MaxMin': val,'theta':argmax_theta,'z':argmin_z}
        
    return soln
     
    
    


def plot_theta_projection(fun,obj,thetaU,thetaL,MaxMinU,MaxMinL):
    
    """
    Plots the projected true function, upper and lower bounds
    See third row, figure 1 of the paper
    
    """
    
    rand_theta_grid = obj.theta_grid*(obj.theta_max - obj.theta_min) + obj.theta_min
    rand_z_grid = obj.z_grid*(obj.z_max - obj.z_min) + obj.z_min
    
    
    
    theta_grid_eval = rand_theta_grid.repeat_interleave(obj.nz_grid,dim = 0)
    z_grid_eval = rand_z_grid.repeat(obj.ntheta_grid,1)

    # We will use the below points for grid based evaluation
    test_x = torch.cat((theta_grid_eval,z_grid_eval),dim = 1)
    
    a = fun(test_x[:,0],test_x[:,1])
    true_fun_vals = (torch.max(a,0)).values
    
    z_list1 = torch.zeros(obj.ntheta_grid,obj.nz)
    val_list1 = torch.zeros(obj.ntheta_grid)
    
    
    for i in range(obj.ntheta_grid):
        val = torch.min(true_fun_vals[i*obj.nz_grid:(i+1)*obj.nz_grid], dim = 0)
        index_val = int(val.indices)
        z_list1[i,:] = obj.z_grid[index_val,:]
        val_list1[i] = val.values
    
    # Sort the theta grid
    theta_grid_sq = rand_theta_grid.squeeze()
    sorted_theta = theta_grid_sq.sort()
    
    index = sorted_theta.indices
    theta_vals = sorted_theta.values
    
    sorted_true = val_list1[index]
    z_sorted = z_list1.squeeze()*(obj.z_max - obj.z_min) + obj.z_min
    ###############################################################
 

    # Scatter plot:
    best_val = (torch.max(obj.Y,dim =0)).values
    scatter_x = obj.Init_theta
    
    
    # Upper and lower bound:
    LCB_proj = obj.LCB_val_proj[index]
    UCB_proj = obj.UCB_val_proj[index]
    
    # Plot the cloud while avaoiding weird numerical issues
    a = (torch.ceil(torch.linspace(0,obj.ntheta_grid -1 ,50))).detach().numpy()
    a = list(a.astype(int))
    
    # Scatter plot of the maxmin ucb and lcb
    
    
    plt.plot(theta_vals,sorted_true,'k',linewidth = 5)
    
    # for i in range(100):
    #     plt.scatter(theta_vals,torch.max(fun(theta_vals,z_sorted + 3*torch.rand(1).squeeze()),dim = 0).values,s = 10)
    
    #torch.normal(0.,0.1,size = (1,))
    
    # Plot the cloud enclosing the UCB and LCB:
    plt.fill_between(theta_vals[a],LCB_proj[a],UCB_proj[a], alpha = 0.2, color = 'purple')
    
    #plt.ylim(-12,8)  
    plt.ylim(-300,300)
    plt.axhline(y = 0, color = 'k', linestyle = 'dashed',linewidth = 2)   
    plt.xlim(obj.theta_min,obj.theta_max)
    plt.xlabel(r'$\theta$', fontsize = 20)
    plt.ylabel(r'$\psi(\theta)$', fontsize = 20)
    
    # The upper and lower confidence bounds
    plt.scatter(thetaU, MaxMinU, c = 'maroon', marker = '*', s = 100)
    plt.axhline(y = MaxMinU, color = 'maroon')   
    
    plt.scatter(thetaL, MaxMinL, c = 'blue', marker = '*', s = 100)
    plt.axhline(y = MaxMinL, color = 'blue')  
    
    plt.show()
    
    
    a = 0
    
    



class BoFlex():
    
    """
    BoFlex class
    
    Arguments:
    theta_min -- torch_tensor dim: Ntheta: Lower bound of \theta
    theta_max -- torch_tensor dim: Ntheta: Upper bound of \theta
    z_min -- torch_tensor dim: nz: Lower bound of z
    z_max -- torch_tensor dim: nz: Upper bound of z
    Init_theta -- Initial theta values generated for the loop
    Init_z -- Initial z values generated for the loop
    Ninit_vals -- Corresponding values of function generated for the loop
    sqrt_beta -- Exploration Parameter for UCB and LCB
    """

    def __init__(self,theta_min,theta_max,z_min,z_max,Init_theta,Init_z,Ninit_vals,sqrt_beta = 2, ntheta_grid = 1000, nz_grid = 1000,theta_min_scale = None,theta_max_scale = None, ARBO = False):

        # Initialize
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.z_min = z_min
        self.z_max = z_max
        self.sqrt_beta = sqrt_beta
        self.ntheta = torch.Tensor.size(theta_min,dim = 0)
        self.nz = torch.Tensor.size(z_min,dim = 0)
        self.n_cons = torch.Tensor.size(Ninit_vals,dim = 0)
        
        self.ARBO = ARBO
        
        self.theta_min_scale = theta_min_scale
        self.theta_max_scale = theta_max_scale
        
        # Scale features
        self.Init_theta = Init_theta
        self.Init_z = Init_z
        self.train_theta, self.train_z = self.ScaleFeatures()
        self.train_x = torch.cat((self.train_theta,self.train_z),dim = 1)

        # Scale mapping
        self.Y = Ninit_vals
        max_vals = torch.max(self.Y,dim = 0)
        self.KS_val = max_vals.values
        self.train_y = self.ScaleMapping()
        
        
        # SOBOL engine initiated for replicability
        soboleng = torch.quasirandom.SobolEngine(dimension=self.ntheta, seed = 1000)   
        soboleng2 = torch.quasirandom.SobolEngine(dimension=self.nz, seed = 1000)  
        
        # Generate Vertex of hyper rectangle of theta + Grid for max min eval:
        corners_theta_grid = active_corners(torch.zeros(self.ntheta),torch.ones(self.ntheta))
        rand_theta_grid = soboleng.draw(ntheta_grid)
        self.theta_grid = torch.cat((rand_theta_grid,corners_theta_grid))
        ntheta_grid = ntheta_grid + 2**self.ntheta
        self.ntheta_grid = ntheta_grid

        # Create a list for z
        corners_z_grid = active_corners(torch.zeros(self.nz),torch.ones(self.nz))
        rand_z_grid = soboleng2.draw(nz_grid)
        self.z_grid = torch.cat((rand_z_grid,corners_z_grid))
        nz_grid = nz_grid + 2**self.nz
        self.nz_grid = nz_grid


        theta_grid_eval = self.theta_grid.repeat_interleave(nz_grid,dim = 0)
        z_grid_eval = self.z_grid.repeat(ntheta_grid,1)

        # We will use the below points for grid based evaluation
        self.test_x = torch.cat((theta_grid_eval,z_grid_eval),dim = 1)

    def ScaleFeatures(self):
        """
        Simple scaling to 0-1

        Returns
        -------
        train_theta : torch double
        train_z : torch double

        """
        # Scales the Features according to the max and min values
        if self.theta_min_scale:
            train_theta = (self.Init_theta - self.theta_min_scale)/(self.theta_max_scale - self.theta_min_scale)                
        else:
            train_theta = (self.Init_theta - self.theta_min)/(self.theta_max - self.theta_min)
            
        # Scale z    
        train_z = (self.Init_z - self.z_min)/(self.z_max - self.z_min)
        return train_theta, train_z

    def ScaleMapping(self):
            # Scales the mapping based on STD deviation
        
        if self.ARBO or self.n_cons == 1:
            self.mean = torch.mean(self.Y)
            self.std = torch.std(self.Y)
            self.var = torch.var(self.Y)                
            y_train = (self.Y - self.mean)/self.std
            y_train = y_train.squeeze(1)
            
        else: 
            
            pre_trained = False
            
            if pre_trained:
               print('Warning: Fixed scaling specifically for Heat X case') 
               self.mean = torch.tensor([[ -27.4220], [1.6236], [-146.7566],[ 13.3505], [ -23.3528]], dtype=torch.float64)
               self.std = torch.tensor([[19.7392],[15.1664],[29.9613],[30.7966],[31.3151]], dtype=torch.float64)
               self.var = torch.tensor([[389.6359],[230.0204],[897.6774],[948.4336],[980.6361]], dtype=torch.float64)
            else:
            
                self.mean = (torch.mean(self.Y,dim = 1)).unsqueeze(1)                   
                                
                self.std = (torch.std(self.Y,dim = 1)).unsqueeze(1) 
                
                self.var = (torch.var(self.Y,dim = 1)).unsqueeze(1)             
            
            y_train = (self.Y - self.mean)/self.std
            
        return y_train

    def update(self,new_theta,new_z,new_vals):
        
        """
        Updates the object with new data points
        """

    # Update the features
        self.Init_theta = torch.cat((self.Init_theta,new_theta),dim = 0)
        self.Init_z = torch.cat((self.Init_z,new_z),dim = 0)

        # Scale features
        self.train_theta, self.train_z = self.ScaleFeatures()
        self.train_x = torch.cat((self.train_theta,self.train_z),dim = 1)

        # Update the mapping
        if self.ARBO:
            self.Y = torch.vstack((self.Y,new_vals))
        else:
            self.Y = torch.cat((self.Y,new_vals),dim = 1)

        # Scale Mapping
        self.train_y = self.ScaleMapping()
        return None

    def train(self,train_hyp = True):
        # Train the GP model            
            self.model,self.likelihood = trainGPModel(self.train_x,self.train_y,train_hyp)
            return None
        
    def train_ARBO(self):
            self.model,self.likelihood = trainGPModel_ARBO(self.train_x,self.train_y)
            return None

    def eval(self):
            # Eval mode for the GP
            self.model.eval()
            
    
    def update_Beta(self):
        
        """
        TODO: develop this method to update the beta value according to the theory
        
        """
        
        self.l = []
        self.sigma_f = []
        self.noise = []
        
        with torch.no_grad():
            for i in range(self.n_cons):
                self.l.append(self.model.models[i].covar_module.base_kernel.lengthscale)
                self.sigma_f.append(self.model.models[i].covar_module.outputscale)
                self.noise.append(self.model.models[i].likelihood.noise)

    def Gen_UCB_LCB(self):
        
        """
        Generates the UCB and LCB values
        """

        # Create a mean list
        MeanLst = torch.zeros(self.nz_grid*self.ntheta_grid,self.n_cons)

        # Create a std list
        StdList = torch.zeros(self.nz_grid*self.ntheta_grid,self.n_cons)
        ## For upper quantile levels
        #UCB = [None]*self.n_cons # We will be selecting a low

        ## For lower quantile level
        #LCB = [None]*self.n_cons

        test_x_tup = ((self.test_x.double(), ) * self.n_cons)

        with torch.no_grad():
            # This contains predictions for both outcomes as a list
            predictions = self.likelihood(*self.model(*test_x_tup))

        i = 0
        for submodel, prediction in zip(self.model.models, predictions):
            # Unpack the mean and std function
            mean = prediction.mean*self.std[i] + self.mean[i]
            std = torch.sqrt(prediction.variance*self.var[i])
            # Add to the list of mean and std
            StdList[:,i] = std
            MeanLst[:,i] = mean
            # Increase iter count
            i += 1

        UCB = MeanLst + self.sqrt_beta*StdList
        LCB = MeanLst - self.sqrt_beta*StdList
        # Use the below function to generate a descritization of values
        return UCB, LCB

    def MaxMin(self):
        
        """
        Pessimistic selection of the worst case constraints selection of theta
        
        """

        UCB,LCB = self.Gen_UCB_LCB()

        max_UCB = torch.max(UCB,dim = 1).values
        max_LCB = torch.max(LCB,dim = 1).values

        # Same procedure as the ARBO case study
        z_list1 = torch.zeros(self.ntheta_grid,self.nz)
        val_list1 = torch.zeros(self.ntheta_grid)

        # Solve the UCB Problem
        #Solve the inner min problem
        for i in range(self.ntheta_grid):
            val = torch.min(max_UCB[i*self.nz_grid:(i+1)*self.nz_grid],dim = 0)
            index_val = int(val.indices)
            z_list1[i,:] = self.z_grid[index_val,:]
            val_list1[i] = val.values


        # Solve the outer max problem
        val = torch.max(val_list1,0)
        argmax_theta = self.theta_grid[val.indices,:]
        if self.theta_min_scale:
            argmax_theta = self.theta_min_scale +  (self.theta_max_scale - self.theta_min_scale)*argmax_theta
        else:
            argmax_theta = self.theta_min + (self.theta_max - self.theta_min)*argmax_theta
        argmin_z = z_list1[val.indices,:]
        argmin_z = self.z_min + (self.z_max - self.z_min)*argmin_z
        val = val.values

        maxmin_dict = {'MaxMin_UCB': val,'max_theta_UCB':argmax_theta,'min_z_UCB':argmin_z}
        
        # Save values for external plot
        with torch.no_grad():
            self.UCB_val_proj = val_list1
        
        
        
        ## Solve the LCB Problem
        
        z_list2 = torch.zeros(self.ntheta_grid,self.nz)
        val_list2 = torch.zeros(self.ntheta_grid)
        # Solve the inner min problem
        for i in range(self.ntheta_grid):
            val = torch.min(max_LCB[i*self.nz_grid:(i+1)*self.nz_grid],dim = 0)
            index_val = int(val.indices)
            z_list2[i,:] = self.z_grid[index_val,:]
            val_list2[i] = val.values

        # Solve the outer max problem
        val = torch.max(val_list2,0)
        argmax_theta = self.theta_grid[val.indices,:]
        if self.theta_min_scale:
            argmax_theta = self.theta_min_scale + + (self.theta_max_scale - self.theta_min_scale)*argmax_theta
        else:
            argmax_theta = self.theta_min + (self.theta_max - self.theta_min)*argmax_theta
        argmin_z = z_list2[val.indices,:]
        argmin_z = self.z_min + (self.z_max - self.z_min)*argmin_z
        val = val.values

        maxmin_dict['MaxMin_LCB'] = val
        maxmin_dict['max_theta_LCB'] = argmax_theta
        maxmin_dict['min_z_LCB'] = argmin_z
        
        # Save values for external plot
        with torch.no_grad():
            self.LCB_val_proj = val_list2
        

        self.max_LCB = max_LCB
        self.max_UCB = max_UCB
        
    

        return maxmin_dict

    def Min_LCB(self,theta_t,Min_UCB = None):
        
        """
        Select best possible 
        
        """
        with torch.no_grad():
            # Solves the Quantile function
            if self.theta_min_scale:
                theta_ts = (theta_t - self.theta_min_scale)/(self.theta_max_scale - self.theta_min_scale)
            else:
                theta_ts = (theta_t - self.theta_min)/(self.theta_max - self.theta_min)
            

            # Check where the best value lies on theta grid
            val = torch.min(torch.square(torch.sum((self.theta_grid - theta_ts),1)),0)
            num_index = int(val.indices)

            # Numerical errors like truncation/round-off error occuring here
            #bool_index = theta_ts == self.theta_grid.squeeze()
            #num_index = int(torch.nonzero(bool_index))
            
            if Min_UCB:
                LCB_pred = self.max_UCB[(num_index)*self.nz_grid:(num_index+1)*self.nz_grid]
            else:
                LCB_pred = self.max_LCB[(num_index)*self.nz_grid:(num_index+1)*self.nz_grid]            
            
            
            val = torch.min(LCB_pred,0)
            z_t = self.z_grid[val.indices,:]
            z_t = self.z_min + (self.z_max - self.z_min)*z_t
            
        return z_t 
    
    def plot(self):
        """
    

        Returns
        -------
        plots UCB and LCB of worst case constraints -- only for 2D problems

        """
        #
        n = 1000
        x_axis = torch.linspace(0,1,n)
        y_axis = torch.linspace(0,1,n)
        
        x,y = torch.meshgrid(x_axis,y_axis)
        
        # reshape x and y to match the input shape of fun
        xy = torch.stack([x.flatten(), y.flatten()], axis=1)
        
        # Create a mean list
        MeanLst2 = torch.zeros(n*n,self.n_cons)

        # Create a std list
        StdList2 = torch.zeros(n*n,self.n_cons)
        
        test_x_tup2 = ((xy, ) * self.n_cons)

        with torch.no_grad():
            # This contaplt.show()ins predictions for both outcomes as a list
            predictions = self.likelihood(*self.model(*test_x_tup2))

        i = 0
        for submodel, prediction in zip(self.model.models, predictions):
            # Unpack the mean and std function
            mean = prediction.mean*self.std[i] + self.mean[i]
            std = torch.sqrt(prediction.variance*self.var[i])
            # Add to the list of mean and std
            StdList2[:,i] = std
            MeanLst2[:,i] = mean
            # Increase iter count
            i += 1
        
        # Find the LCB and UCB of each GP Kernel
        UCB_plot = MeanLst2 + self.sqrt_beta*StdList2
        LCB_plot = MeanLst2 - self.sqrt_beta*StdList2
        
        UCB_max = torch.max(UCB_plot,axis = 1).values
        LCB_max = torch.max(LCB_plot,axis = 1).values
        
        
        
        UCB_max = torch.reshape(UCB_max,x.size())
        LCB_max = torch.reshape(LCB_max,x.size())
        
        x_reshape = self.theta_min + (self.theta_max - self.theta_min)*x
        y_reshape = self.z_min + (self.z_max - self.z_min)*y
        
        vmin = -2
        vmax = 5      #UCB_max.max()
        levels = torch.linspace(vmin,vmax,50)
        
        fig, ax = plt.subplots(1, 1) 
        
        color_map = "rainbow"
        cmap = plt.set_cmap(color_map)
        contour_plot = ax.contourf(x_reshape,y_reshape,UCB_max, levels = levels, cmap = cmap,extend = 'both')
        ax.scatter(self.Init_theta,self.Init_z, c = 'k', marker = 'o',s = 100)
        ax.set_xlabel(r'$\theta$', fontsize = 20)
        ax.set_ylabel('z', fontsize = 20)
        
        #fig.colorbar(contour_plot, format = '%.1f')
        plt.show()
        
        
        fig, ax = plt.subplots(1, 1) 
        cmap = plt.set_cmap(color_map)
        contour_plot = ax.contourf(x_reshape,y_reshape,LCB_max, levels = levels, cmap = cmap,extend = 'both')
        ax.scatter(self.Init_theta,self.Init_z, c = 'k', marker = 'o',s = 100)
        ax.set_xlabel(r'$\theta$', fontsize = 20)
        ax.set_ylabel('z', fontsize = 20)
        #fig.colorbar(contour_plot, format = '%.1f')
        
        plt.show()
        
        a = 0
        return 0 
    
    ################ Does exactly same as above counter parts but does not use decomposed structure ############################
    def MaxMin_ARBO(self):
        f_preds = self.model(self.test_x)
        f_mean = f_preds.mean 
        f_var = f_preds.variance
        
        with torch.no_grad():
            # UCB
            UCB_pred = f_mean*self.std + self.mean + self.sqrt_beta*torch.sqrt(f_var*torch.square(self.std))
        
            # LCB
            LCB_pred = f_mean*self.std + self.mean - self.sqrt_beta*torch.sqrt(f_var*torch.square(self.std))
        
            #theta_list1 = torch.zeros(self.ntheta_grid,ntheta)
            z_list1 = torch.zeros(self.ntheta_grid,self.nz)
            val_list1 = torch.zeros(self.ntheta_grid)
        
            ## Solve the UCB Problem
            # Solve the inner min problem
            for i in range(self.ntheta_grid):
                val = torch.min(UCB_pred[i*self.nz_grid:(i+1)*self.nz_grid],0)
                index_val = int(val.indices)
                z_list1[i,:] = self.z_grid[index_val,:]
                val_list1[i] = val.values
        
            # Solve the outer max problem
            val = torch.max(val_list1,0)
            argmax_theta = self.theta_grid[val.indices,:]
            argmax_theta = self.theta_min + (self.theta_max - self.theta_min)*argmax_theta
            argmin_z = z_list1[val.indices,:]
            argmin_z = self.z_min + (self.z_max - self.z_min)*argmin_z
            val = val.values
        
            maxmin_dict = {'MaxMin_UCB': val,'max_theta_UCB':argmax_theta,'min_z_UCB':argmin_z}
            
            # Save values for external plot
            self.UCB_val_proj = copy.deepcopy(val_list1)
        
            ## Solve the LCB Problem
            # Solve the inner min problem
            for i in range(self.ntheta_grid):
                val = torch.min(LCB_pred[i*self.nz_grid:(i+1)*self.nz_grid],0)
                index_val = int(val.indices)
                z_list1[i,:] = self.z_grid[index_val,:]
                val_list1[i] = val.values
        
            # Solve the outer max problem
            val = torch.max(val_list1,0)
            argmax_theta = self.theta_grid[val.indices,:]
            argmax_theta = self.theta_min + (self.theta_max - self.theta_min)*argmax_theta
            argmin_z = z_list1[val.indices,:]
            argmin_z = self.z_min + (self.z_max - self.z_min)*argmin_z
            val = val.values
        
            maxmin_dict['MaxMin_LCB'] = val
            maxmin_dict['max_theta_LCB'] = argmax_theta
            maxmin_dict['min_z_LCB'] = argmin_z
            
            # Save values for external plot
            self.LCB_val_proj = copy.deepcopy(val_list1) 
            
        return maxmin_dict

    def Min_LCB_ARBO(self,theta_t):
        theta_ts = (theta_t - self.theta_min)/(self.theta_max - self.theta_min)
        test_theta = theta_ts.repeat(self.nz_grid,1)
        test_x2 = torch.cat((test_theta,self.z_grid),dim = 1)
    
        # Prediction based on test_x2
        f_preds = self.model(test_x2)
        f_mean = f_preds.mean
        f_var = f_preds.variance
    
        with torch.no_grad():
            # LCB
            LCB_pred = f_mean*self.std + self.mean - self.sqrt_beta*torch.sqrt(f_var*torch.square(self.std))
            val = torch.min(LCB_pred,0)
            z_t = self.z_grid[val.indices,:]
            z_t = self.z_min + (self.z_max - self.z_min)*z_t
        return z_t

        
        
        
        
        
        
        
        

        



        
       