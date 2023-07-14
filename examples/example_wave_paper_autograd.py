# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:25:13 2022

@author: user
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.append('../')


sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.input_preprocessing import Equation
from tedeous.solver import Solver
from tedeous.metrics import Solution
from tedeous.device import solver_device,check_device


solver_device('cuda')




exp_dict_list=[]

for grid_res in range(20, 110, 10):
    x = torch.from_numpy(np.linspace(0, 1, grid_res + 1))
    t = torch.from_numpy(np.linspace(0, 1, grid_res + 1))

    coord_list = []
    coord_list.append(x)
    coord_list.append(t)

    grid = torch.cartesian_prod(x, t).float()


    
    """
    Preparing boundary conditions (BC)
    
    Unlike KdV example there is optional possibility to define only two items
    when boundary operator is not needed
    
    bnd=torch.Tensor of a boundary n-D points where n is the problem
    dimensionality
    
    bval=torch.Tensor prescribed values at every point in the boundary
    
    """
    
    # Initial conditions at t=0
    bnd1 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()
    
    # u(0,x)=sin(pi*x)
    bndval1 = torch.sin(np.pi * bnd1[:, 0])
    
    # Initial conditions at t=1
    bnd2 = torch.cartesian_prod(x, torch.from_numpy(np.array([1], dtype=np.float64))).float()
    
    # u(1,x)=sin(pi*x)
    bndval2 = torch.sin(np.pi * bnd2[:, 0])
    
    # Boundary conditions at x=0
    bnd3 = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), t).float()
    
    # u(0,t)=0
    bndval3 = torch.from_numpy(np.zeros(len(bnd3), dtype=np.float64))
    
    # Boundary conditions at x=1
    bnd4 = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)), t).float()
    
    # u(1,t)=0
    bndval4 = torch.from_numpy(np.zeros(len(bnd4), dtype=np.float64))
    
    # Putting all bconds together
    bconds = [[bnd1, bndval1, 'dirichlet'],
              [bnd2, bndval2, 'dirichlet'],
              [bnd3, bndval3, 'dirichlet'],
              [bnd4, bndval4, 'dirichlet']]
    
    """
    Defining wave equation
    
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
    # operator is 4*d2u/dx2-1*d2u/dt2=0
    wave_eq = {
        '4*d2u/dx2**1':
            {
                'coeff': 4,
                'd2u/dx2': [0, 0],
                'pow': 1
            },
        '-d2u/dt2**1':
            {
                'coeff': -1,
                'd2u/dt2': [1,1],
                'pow':1
            }
    }
    
    
  
    for _ in range(10):
        sln=np.genfromtxt(os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'wolfram_sln/wave_sln_'+str(grid_res)+'.csv')),delimiter=',')
        
        model= torch.rand(grid[0].shape)
        
        start = time.time()
        
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 1)
        )

        equation = Equation(grid, wave_eq, bconds).set_strategy('autograd')

        img_dir=os.path.join(os.path.dirname( __file__ ), 'wave_autograd_img')

        if not(os.path.isdir(img_dir)):
            os.mkdir(img_dir)

        model=Solver(grid, equation, model, 'autograd').solve(use_cache=True, verbose=True, print_every=None,
                                                              cache_verbose=True, abs_loss=0.001, step_plot_print=False,
                                                              step_plot_save=True,image_save_dir=img_dir)


    
        end = time.time()

        error_rmse = np.sqrt(np.mean((sln.reshape(-1) - model(check_device(grid)).detach().cpu().numpy().reshape(-1)) ** 2))



        end_loss = Solution(grid=grid, equal_cls=equation, model=model,
             mode='autograd', weak_form=None, lambda_bound=100).evaluate()

        exp_dict_list.append({'grid_res':grid_res,'time':end - start,'RMSE':error_rmse,'loss':end_loss.detach().cpu().numpy(),'type':'wave_eqn'})
        
        print('Time taken {}= {}'.format(grid_res, end - start))
        print('RMSE {}= {}'.format(grid_res, error_rmse))
        print('loss {}= {}'.format(grid_res, end_loss))


#import pandas as pd
#CACHE=True

#result_assessment=pd.DataFrame(exp_dict_list)
#result_assessment.to_csv('examples/benchmarking_data/wave_experiment_2_20_100_cache={}.csv'.format(CACHE))

#result_assessment.boxplot(by='grid_res',column='time',showfliers=False,figsize=(20,10),fontsize=42)

#result_assessment.boxplot(by='grid_res',column='RMSE',figsize=(20,10),fontsize=42)