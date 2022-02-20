# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:33:44 2021

@author: user
"""
import torch
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys

sys.path.append('../')

from solver import *
from cache import *
import time

"""
Preparing grid

Grid is an essentially torch.Tensor  of a n-D points where n is the problem
dimensionality
"""

device = torch.device('cpu')



A = 2
C = np.sqrt(10)

def func(grid):
    x, t = grid[:,0],grid[:,1]
    return torch.sin(np.pi * x) * torch.cos(C * np.pi * t) + torch.sin(A * np.pi * x) * torch.cos(
        A * C * np.pi * t
    )

def wave_experiment(grid_res,CACHE):
    
    exp_dict_list=[]
    

    x_grid=np.linspace(0,1,grid_res+1)
    t_grid=np.linspace(0,1,grid_res+1)
    
    x = torch.from_numpy(x_grid)
    t = torch.from_numpy(t_grid)
    
    grid = torch.cartesian_prod(x, t).float()
    
    grid.to(device)

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
    bconds = [[bnd1, bndval1], [bnd2, bndval2], [bnd3, bndval3], [bnd4, bndval4]]
    
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

    sln=np.genfromtxt('wolfram_sln/wave_sln_'+str(grid_res)+'.csv',delimiter=',')
    sln_torch=torch.from_numpy(sln)
    sln_torch1=sln_torch.reshape(-1,1)

    # model = torch.nn.Sequential(
    #     torch.nn.Linear(2, 100),
    #     torch.nn.Tanh(),
    #     torch.nn.Linear(100, 100),
    #     torch.nn.Tanh(),
    #     torch.nn.Linear(100, 100),
    #     torch.nn.Tanh(),
    #     torch.nn.Linear(100, 100),
    #     torch.nn.Tanh(),
    #     torch.nn.Linear(100, 1)
    # )

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 100),
        torch.nn.ReLU(),
        # torch.nn.Linear(100, 100),
        # torch.nn.ReLU(),
        # torch.nn.Linear(100, 100),
        # torch.nn.ReLU(),
        torch.nn.Linear(100, 1)
    )


    lp_par={'operator_p':2,
        'operator_weighted':False,
        'operator_normalized':False,
        'boundary_p':1,
        'boundary_weighted':False,
        'boundary_normalized':False}
    
    start = time.time()
    
    model = point_sort_shift_solver(grid, model, wave_eq, bconds, lambda_bound=10, verbose=2, learning_rate=1e-4, h=abs((t[1]-t[0]).item()),
                                    eps=1e-6, tmin=1000, tmax=1e6,use_cache=CACHE,cache_dir='../cache/',cache_verbose=True
                                    ,batch_size=None, save_always=True,lp_par=lp_par,print_every=None,
                                    model_randomize_parameter=1e-6)
    end = time.time()
        
    error_rmse=torch.sqrt(torch.mean((sln_torch1-model(grid))**2))
    
  
    
    prepared_grid,grid_dict,point_type = grid_prepare(grid)
    
    prepared_bconds = bnd_prepare(bconds, prepared_grid,grid_dict, h=abs((t[1]-t[0]).item()))
    prepared_operator = operator_prepare(wave_eq, grid_dict, subset=['central'], true_grid=grid, h=abs((t[1]-t[0]).item()))
    end_loss = point_sort_shift_loss(model, prepared_grid, prepared_operator, prepared_bconds, lambda_bound=100)
    exp_dict_list.append({'grid_res':grid_res,'time':end - start,'RMSE':error_rmse.detach().numpy(),'loss':end_loss.detach().numpy(),'type':'wave_eqn','cache':CACHE})
    
    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))
    print('loss {}= {}'.format(grid_res, end_loss))
    return exp_dict_list


nruns=10

exp_dict_list=[]

CACHE=True

for grid_res in range(10,101,10):
    for _ in range(nruns):
        exp_dict_list.append(wave_experiment(grid_res,CACHE))
   

        
import pandas as pd

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df=pd.DataFrame(exp_dict_list_flatten)
df.boxplot(by='grid_res',column='time',fontsize=42,figsize=(20,10))
df.boxplot(by='grid_res',column='RMSE',fontsize=42,figsize=(20,10),showfliers=False)
df.to_csv('benchmarking_data/wave_experiment_2_10_100_cache={}.csv'.format(str(CACHE)))