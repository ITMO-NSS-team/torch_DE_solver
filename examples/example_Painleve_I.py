# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:33:44 2021

@author: user
"""
import torch
import numpy as np
import os
import sys
import time


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device, check_device

solver_device('cpu')

def p_I_exp(grid_res, nruns, CACHE):
    
    exp_dict_list=[]
    
    domain = Domain()
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
    # point t=0
    boundaries.dirichlet({'t': 0}, value=0)
    
    # point t=0
    # d/dt
    bop2 ={
            '1*du/dt**1':
                {
                    'coeff': 1,
                    'du/dt': [0],
                    'pow': 1
                    }
                
        }
    boundaries.operator({'t': 0}, operator=bop2, value=0)    

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
        return -grid

    equation = Equation()

    # P_I operator is  d2u/dt2-6*u^2-t=0 
    p_1= {
        '1*d2u/dt2**1':
            {
                'coeff': 1,
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
                'coeff': p1_c1,
                'u':  [None],
                'pow': 0
            }
    }

    equation.add(p_1)


    for _ in range(nruns):
        
        sln=np.genfromtxt(os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'wolfram_sln/P_I_sln_'+str(grid_res)+'.csv')),delimiter=',')
        sln_torch=torch.from_numpy(sln)
        sln_torch1=sln_torch.reshape(-1,1)
        
                
        
        net = torch.nn.Sequential(
        torch.nn.Linear(1, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 1)
        )

        model =  Model(net, domain, equation, boundaries)

        model.compile("NN", lambda_operator=1, lambda_bound=100)

        img_dir=os.path.join(os.path.dirname( __file__ ), 'PI_NN_img')

        start = time.time()
        
        cb_cache = cache.Cache(cache_verbose=False, model_randomize_parameter=1e-6)

        cb_es = early_stopping.EarlyStopping(eps=1e-7,
                                            loss_window=100,
                                            no_improvement_patience=1000,
                                            patience=5,
                                            randomize_parameter=1e-6,
                                            info_string_every=1000)

        cb_plots = plot.Plots(save_every=1000, print_every=None, img_dir=img_dir)

        optimizer = Optimizer('Adam', {'lr': 1e-4})

        model.train(optimizer, 1e5, save_model=False, callbacks=[cb_cache, cb_es, cb_plots])

        end = time.time()

        grid = domain.build('NN')

        error_rmse=torch.sqrt(torch.mean((sln_torch1-net(check_device(grid)).detach().cpu())**2))

        exp_dict_list.append({'grid_res':grid_res,'time':end - start,'RMSE':error_rmse.detach().cpu().numpy(),'type':'PI','cache':CACHE})
        
        print('Time taken {}= {}'.format(grid_res, end - start))
        print('RMSE {}= {}'.format(grid_res, error_rmse))
    return exp_dict_list


nruns=10

exp_dict_list=[]

CACHE=True


for grid_res in range(10,100,10):
    exp_dict_list.append(p_I_exp(grid_res, nruns, CACHE))


for grid_res in range(100,501,100):
    exp_dict_list.append(p_I_exp(grid_res, nruns, CACHE))


import pandas as pd

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df=pd.DataFrame(exp_dict_list_flatten)
df.boxplot(by='grid_res',column='time',fontsize=42,figsize=(20,10))
df.boxplot(by='grid_res',column='RMSE',fontsize=42,figsize=(20,10),showfliers=False)
df.to_csv('benchmarking_data/PI_experiment_10_500_cache={}.csv'.format(str(CACHE)))


