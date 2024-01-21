# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:25:13 2022

@author: user
"""
import numpy as np
import torch
import sys
import os
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import early_stopping, plot, cache
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device, check_device


solver_device('cuda')

exp_dict_list=[]

for grid_res in range(20, 110, 10):
    domain = Domain()
    domain.variable('x', [0, 1], grid_res)
    domain.variable('t', [0, 1], grid_res)
    
    """
    Preparing boundary conditions (BC)
    
    Unlike KdV example there is optional possibility to define only two items
    when boundary operator is not needed
    
    bnd=torch.Tensor of a boundary n-D points where n is the problem
    dimensionality
    
    bval=torch.Tensor prescribed values at every point in the boundary
    
    """
    
    boundaries = Conditions()

    # Initial conditions at t=0
    x = domain.variable_dict['x']
    boundaries.dirichlet({'x': [0, 1], 't': 0}, value = torch.sin(np.pi*x))
    
    # Initial conditions at t=1
    boundaries.dirichlet({'x': [0, 1], 't': 1}, value = torch.sin(np.pi*x))
    
    # Boundary conditions at x=0
    boundaries.dirichlet({'x': 0, 't': [0, 1]}, value=0)
    
    # Boundary conditions at x=1
    boundaries.dirichlet({'x': 1, 't': [0, 1]}, value=0)
    
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
    equation = Equation()
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
    
    equation.add(wave_eq)
  
    for _ in range(10):
        sln=np.genfromtxt(os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'wolfram_sln/wave_sln_'+str(grid_res)+'.csv')),delimiter=',')
        
        start = time.time()
        
        net = torch.nn.Sequential(
            torch.nn.Linear(2, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 1)
        )

        model =  Model(net, domain, equation, boundaries)

        model.compile("autograd", lambda_operator=1, lambda_bound=1000)

        cb_es = early_stopping.EarlyStopping(eps=1e-6, abs_loss=0.001)

        img_dir = os.path.join(os.path.dirname( __file__ ), 'wave_eq_img')

        cb_plots = plot.Plots(save_every=1000, print_every=None, img_dir=img_dir)

        optimizer = Optimizer('Adam', {'lr': 1e-3})

        model.train(optimizer, 1e5, save_model=False, callbacks=[cb_es, cb_plots])

        end = time.time()

        grid = domain.build('autograd')

        error_rmse = np.sqrt(np.mean((sln.reshape(-1) - net(check_device(grid)).detach().cpu().numpy().reshape(-1)) ** 2))

        exp_dict_list.append({'grid_res':grid_res,'time':end - start,'RMSE':error_rmse,'type':'wave_eqn'})
        
        print('Time taken {}= {}'.format(grid_res, end - start))
        print('RMSE {}= {}'.format(grid_res, error_rmse))


#import pandas as pd
#CACHE=True

#result_assessment=pd.DataFrame(exp_dict_list)
#result_assessment.to_csv('examples/benchmarking_data/wave_experiment_2_20_100_cache={}.csv'.format(CACHE))

#result_assessment.boxplot(by='grid_res',column='time',showfliers=False,figsize=(20,10),fontsize=42)

#result_assessment.boxplot(by='grid_res',column='RMSE',figsize=(20,10),fontsize=42)