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
sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from input_preprocessing import Equation
from solver import Solver
from metrics import Solution
import time


device = torch.device('cpu')


def func(grid):
    x, t = grid[:,0],grid[:,1]
    sln=500+x
    for i in range(1,100):
        sln+=8*np.exp(-1/4*np.pi**2*t*(2*i-1)**2)*((-1)**i+250*np.pi*(1-2*i))*np.sin(1/2*np.pi*x*(2*i-1))/(np.pi-2*np.pi*i)**2
    return sln

def heat_experiment(grid_res,CACHE):
    
    exp_dict_list=[]
    

    x_grid=np.linspace(0,1,grid_res+1)
    t_grid=np.linspace(0,1,grid_res+1)
    
    x = torch.from_numpy(x_grid)
    t = torch.from_numpy(t_grid)
    
    grid = torch.cartesian_prod(x, t).float()
    
    grid.to(device)

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
    # Boundary conditions at x=0
    bnd1 = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), t).float()
    
    
    # u(0,t)=500
    bndval1 = torch.from_numpy(np.zeros(len(bnd1), dtype=np.float64)+500)
    
    
    # Boundary conditions at x=1
    bnd2 = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)), t).float()
    
    # u'(1,t)=1
    bop2= {
            'du/dx':
                {
                    'r1': 1,
                    'du/dx': [0],
                    'pow': 1
                }
        }
    
    # u(x,0)=sin(pi*x)
    bndval2 = torch.from_numpy(np.zeros(len(bnd1), dtype=np.float64)+1)
    
    # Initial conditions at t=0
    bnd3 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()
    
    
    # u(x,0)=0
    bndval3 = torch.from_numpy(np.zeros(len(bnd1), dtype=np.float64))
    

    bconds = [[bnd1, bndval1], [bnd2, bop2, bndval2], [bnd3, bndval3]]
     
        
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
    heat_eq = {
        'du/dt**1':
            {
                'coeff': 1,
                'du/dt': [1],
                'pow':1
            },
            '-d2u/dx2**1':
            {
                'coeff': -1,
                'd2u/dx2': [0, 0],
                'pow': 1
            }
    }


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
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 1)
    )

    
    h=abs((t[1]-t[0]).item())

    equation = Equation(grid, heat_eq, bconds, h=h).set_strategy('NN')

    start = time.time()
    
    model = Solver(grid, equation, model, 'NN').solve(lambda_bound=10, verbose=2, learning_rate=1e-4,
                                    eps=1e-6, tmin=1000, tmax=1e6,use_cache=CACHE,cache_dir='../cache/',cache_verbose=True
                                    ,save_always=True, print_every=None, model_randomize_parameter=1e-6)
    end = time.time()
    
    
    rmse_x_grid=np.linspace(0,1,grid_res+1)
    rmse_t_grid=np.linspace(0.1,1,grid_res+1)

    rmse_x = torch.from_numpy(rmse_x_grid)
    rmse_t = torch.from_numpy(rmse_t_grid)

    rmse_grid = torch.cartesian_prod(rmse_x, rmse_t).float()
    
    error_rmse=torch.sqrt(torch.mean(((func(rmse_grid)-model(rmse_grid))/500)**2))
    
  
    
    end_loss = Solution(grid, equation, model, 'NN').loss_evaluation(lambda_bound=100)
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
        exp_dict_list.append(heat_experiment(grid_res,CACHE))
   

        
import pandas as pd

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df=pd.DataFrame(exp_dict_list_flatten)
df.boxplot(by='grid_res',column='time',fontsize=42,figsize=(20,10))
df.boxplot(by='grid_res',column='RMSE',fontsize=42,figsize=(20,10),showfliers=False)
df.to_csv('benchmarking_data/heat_experiment_10_100_cache={}.csv'.format(str(CACHE)))



