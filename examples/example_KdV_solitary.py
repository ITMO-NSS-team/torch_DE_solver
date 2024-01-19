# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:33:44 2021

@author: user
"""
import torch
import numpy as np
import time
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device, check_device, device_type


solver_device('gpu')

exp_dict_list=[]


def soliton(x,t):
    E=np.exp(1)
    s=-((18*torch.exp((1/125)*(t + 25*x))*(16*torch.exp(2*t) + 
       1000*torch.exp((126*t)/125 + (4*x)/5) + 9*torch.exp(2*x) + 576*torch.exp(t + x) + 
       90*torch.exp((124*t)/125 + (6*x)/5)))/(5*(40*torch.exp((126*t)/125) + 
        18*torch.exp(t + x/5) + 9*torch.exp((6*x)/5) + 45*torch.exp(t/125 + x))**2))
    return s

def soliton_x(x,t):
    E=np.exp(1)
    s1=(18*E**((1/125)*(t + 25*x))*(-640*E**((376*t)/125) + 
     288*E**(3*t + x/5) - 200000*E**((252*t)/125 + (4*x)/5) + 
     81*E**((16*x)/5) - 185760*E**((251*t)/125 + x) - 
     65088*E**(2*t + (6*x)/5) - 8100*E**((249*t)/125 + (7*x)/5) + 
     225000*E**((127*t)/125 + (9*x)/5) + 
         162720*E**((126*t)/125 + 2*x) + 41796*E**(t + (11*x)/5) - 
     405*E**(t/125 + 3*x) + 
     4050*E**((4/125)*(31*t + 75*x))))/(25*(40*E**((126*t)/125) + 
      18*E**(t + x/5) + 9*E**((6*x)/5) + 45*E**(t/125 + x))**3)
    return s1

for grid_res in [30,50,100]:
    
    """
    Preparing grid

    Grid is an essentially torch.Tensor of a n-D points where n is the problem
    dimensionality
    """

    domain = Domain()
    domain.variable('x', [-10, 10], grid_res)
    domain.variable('t', [0, 1], grid_res)

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
    boundaries = Conditions()
    x = domain.variable_dict['x']
    t = domain.variable_dict['t']
    """
    Boundary x=-10
    """

    boundaries.dirichlet({'x': -10, 't': [0, 1]}, value=soliton(torch.tensor([-10]), t))
    
    """
    Boundary x=10
    """
    boundaries.dirichlet({'x': 10, 't': [0, 1]}, value=soliton(torch.tensor([10]), t))
    
    """
    Another boundary x=-10
    """
    # operator r1*du/dx+r2*u
    bop3 = {
        'r1*du/dx':
            {
                'coeff': 1,
                'du/dx': [0],
                'pow': 1,
                'var':0
            }
            }
    
    boundaries.operator({'x': -10, 't': [0, 1]}, operator=bop3, value=soliton_x(torch.tensor([-10]), t))
    """
    Initial conditions at t=0
    """
    boundaries.dirichlet({'x': [-10, 10], 't': 0}, value=soliton(x, torch.tensor([0])))
    
    """
    Defining kdv equation
    
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
    
    # operator is du/dt+6u*(du/dx)+d3u/dx3-sin(x)*cos(t)=0
    kdv = {
        '1*du/dt**1':
            {
                'coeff': 1,
                'du/dt': [1],
                'pow': 1,
                'var': 0
            },
        '6*u**1*du/dx**1':
            {
                'coeff': 6,
                'u*du/dx': [[None], [0]],
                'pow': [1,1],
                'var':[0,0]
            },
        'd3u/dx3**1':
            {
                'coeff': 1,
                'd3u/dx3': [0, 0, 0],
                'pow': 1,
                'var':0
            }
    }
    
    equation.add(kdv)
    
    """
    Solving equation
    """
    for _ in range(10):

        net = torch.nn.Sequential(
            torch.nn.Linear(2, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 1)
        )
    
        start = time.time()

        model = Model(net, domain, equation, boundaries)
        
        model.compile('NN', lambda_operator=1, lambda_bound=100, h=0.01)

        img_dir = os.path.join(os.path.dirname( __file__ ), 'kdv_solitary_img')

        cb_cache = cache.Cache(cache_verbose=True, model_randomize_parameter=1e-6)

        cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                            loss_window=100,
                                            no_improvement_patience=1000,
                                            patience=5,
                                            randomize_parameter=1e-6)
        
        cb_plots = plot.Plots(save_every=1000, print_every=None, img_dir=img_dir)

        optimizer = Optimizer('Adam', {'lr': 1e-4})

        model.train(optimizer, 1e5, save_model=True, callbacks=[cb_es, cb_cache, cb_plots])

        end = time.time()
        
        grid = domain.build('NN')
        sln_torch = torch.Tensor([soliton(point[0],point[1]) for point in grid]).detach().cpu()
        sln_torch1 = sln_torch.reshape(-1,1)

        net = net.to(device=device_type())
        grid = check_device(grid)
        sln_torch1 = check_device(sln_torch1)

        error_rmse=torch.sqrt(torch.mean((sln_torch1-net(grid))**2))
    
        exp_dict_list.append({'grid_res':grid_res,'time':end - start,'RMSE':error_rmse.detach().cpu().numpy(),'type':'kdv_eqn_solitary','cache':True})
        
        print('Time taken {}= {}'.format(grid_res, end - start))
        print('RMSE {}= {}'.format(grid_res, error_rmse))


CACHE=True

import pandas as pd

result_assessment=pd.DataFrame(exp_dict_list)

result_assessment.boxplot(by='grid_res',column='time',showfliers=False,figsize=(20,10),fontsize=42)

result_assessment.boxplot(by='grid_res',column='RMSE',figsize=(20,10),fontsize=42)

result_assessment.to_csv('examples/benchmarking_data/kdv_solitary_experiment_30_100_cache={}.csv'.format(str(CACHE)))

