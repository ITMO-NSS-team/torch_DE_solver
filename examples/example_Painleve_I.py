# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:33:44 2021

@author: user
"""
import torch
import torchtext
import SALib
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import sys
import time


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# sys.path.append('../')
sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.input_preprocessing import Equation
from tedeous.solver import Solver
from tedeous.solution import Solution
from tedeous.device import solver_device, check_device

solver_device('cpu')



def p_I_exp(grid_res,nruns,CACHE):
    
    exp_dict_list=[]
    
    t = torch.from_numpy(np.linspace(0, 1, grid_res+1))
    
    grid = t.reshape(-1, 1).float()


    """
    Preparing boundary conditions (BC)
    
    For every boundary we define three items
    
    bnd=torch.Tensor of a boundary n-D points where n is the problem
    dimensionality
    
    bop=dict in form {'term1':term1,'term2':term2}-> term1+term2+...=0
    
    NB! dictionary keys at the current time serve only for user-frienly 
    description/comments and are not used in model directly thus order of
    items must be preserved as (coeff,op,pow)
    
    term is a dict term={coefficient:c1,[sterm1,sterm2],'pow': power}
    
    Meaning c1*u*d2u/dx2 has the form
    
    {'coefficient':c1,
     'u*d2u/dx2': [[None],[0,0]],
     'pow':[1,1]}
    
    None is for function without derivatives
    
    
    bval=torch.Tensor prescribed values at every point in the boundary
    """
    
    # point t=0
    bnd1 = torch.from_numpy(np.array([[0]], dtype=np.float64)).float()
    
    
    #  So u(0)=-1/2
    bndval1 = torch.from_numpy(np.array([[0]], dtype=np.float64))
    
    # point t=1
    bnd2 = torch.from_numpy(np.array([[float(t[0])]], dtype=np.float64)).float()
    
    # d/dt
    bop2 ={
            '1*du/dt**1':
                {
                    'coeff': 1,
                    'du/dt': [0],
                    'pow': 1
                    }
                
        }
        
        
    
    # So, du/dt |_{x=1}=3
    bndval2 = torch.from_numpy(np.array([[0]], dtype=np.float64))
    
    # Putting all bconds together
    bconds = [[bnd1, bndval1, 'dirichlet'],
              [bnd2, bop2, bndval2, 'operator']]
    
    """
    Defining Legendre polynomials generating equations
    
    Operator has the form
    
    op=dict in form {'term1':term1,'term2':term2}-> term1+term2+...=0
    
    NB! dictionary keys at the current time serve only for user-frienly 
    description/comments and are not used in model directly thus order of
    items must be preserved as (coeff,op,pow)
    
    
    
    term is a dict term={coefficient:c1,[sterm1,sterm2],'pow': power}
    
    c1 may be integer, function of grid or tensor of dimension of grid
    
    Meaning c1*u*d2u/dx2 has the form
    
    {'coefficient':c1,
     'u*d2u/dx2': [[None],[0,0]],
     'pow':[1,1]}
    
    None is for function without derivatives
    
    
    """
    
    
    # t
    def p1_c1(grid):
        return grid
    
    
    
 
    # P_I operator is  d2u/dt2-6*u^2-t=0 
    p_1= {
        '1*d2u/dt2**1':
            {
                'coeff': 1, #coefficient is a torch.Tensor
                'du/dt': [0, 0],
                'pow': 1
            },
        '-6*u**2':
            {
                'coeff': -6,
                'u':  [None],
                'pow': 2
            },
        '-t':
            {
                'coeff': -p1_c1(grid),
                'u':  [None],
                'pow': 0
            }
    }

    
    
  
    for _ in range(nruns):
        
        sln=np.genfromtxt(os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'wolfram_sln/P_I_sln_'+str(grid_res)+'.csv')),delimiter=',')
        sln_torch=torch.from_numpy(sln)
        sln_torch1=sln_torch.reshape(-1,1)
        
                
        
        model = torch.nn.Sequential(
        torch.nn.Linear(1, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 1)
        )

        start = time.time()

        equation = Equation(grid, p_1, bconds).set_strategy('NN')

        img_dir=os.path.join(os.path.dirname( __file__ ), 'PI_NN_img')


        model = Solver(grid, equation, model, 'NN').solve(lambda_bound=100, verbose=True, learning_rate=1e-4,
                                        eps=1e-7, tmin=1000, tmax=1e5,use_cache=CACHE,cache_dir='../cache/',cache_verbose=False
                                        ,save_always=False,print_every=None,model_randomize_parameter=1e-6,step_plot_print=False,step_plot_save=True,image_save_dir=img_dir)
        end = time.time()


        error_rmse=torch.sqrt(torch.mean((sln_torch1-model(check_device(grid)).detach().cpu())**2))
        
  
        _, end_loss = Solution(grid=grid, equal_cls=equation, model=model,
             mode='NN', weak_form=None, lambda_bound=100,lambda_operator=1).evaluate()

        exp_dict_list.append({'grid_res':grid_res,'time':end - start,'RMSE':error_rmse.detach().cpu().numpy(),'loss':end_loss.detach().cpu().numpy(),'type':'PI','cache':CACHE})
        
        print('Time taken {}= {}'.format(grid_res, end - start))
        print('RMSE {}= {}'.format(grid_res, error_rmse))
        print('loss {}= {}'.format(grid_res, end_loss))
    return exp_dict_list


nruns=10

exp_dict_list=[]

CACHE=True


for grid_res in range(10,100,10):
    exp_dict_list.append(p_I_exp(grid_res, nruns,CACHE))


for grid_res in range(100,501,100):
    exp_dict_list.append(p_I_exp(grid_res, nruns,CACHE))


import pandas as pd

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df=pd.DataFrame(exp_dict_list_flatten)
df.boxplot(by='grid_res',column='time',fontsize=42,figsize=(20,10))
df.boxplot(by='grid_res',column='RMSE',fontsize=42,figsize=(20,10),showfliers=False)
df.to_csv('benchmarking_data/PI_experiment_10_500_cache={}.csv'.format(str(CACHE)))


#exp_dict_list=[]

#CACHE=True


#for grid_res in range(10,100,10):
#    exp_dict_list.append(p_I_exp(grid_res, nruns,CACHE))


#for grid_res in range(100,501,100):
#    exp_dict_list.append(p_I_exp(grid_res, nruns,CACHE))


#exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
#df=pd.DataFrame(exp_dict_list_flatten)
#df.boxplot(by='grid_res',column='time',fontsize=42,figsize=(20,10))
#df.boxplot(by='grid_res',column='RMSE',fontsize=42,figsize=(20,10),showfliers=False)
#df.to_csv('benchmarking_data/PI_experiment_10_500_cache={}.csv'.format(str(CACHE)))


