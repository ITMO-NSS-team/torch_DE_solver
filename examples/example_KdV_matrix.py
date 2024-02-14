# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:33:44 2021

@author: user
"""
import os
import numpy as np
import torch
import time
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))


from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.models import mat_model
from tedeous.callbacks import cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.solution import Solution
import time



solver_device('cpu')

exp_dict_list=[]


for grid_res in [20,30]:
    
    """
    Preparing grid

    Grid is an essentially torch.Tensor of a n-D points where n is the problem
    dimensionality
    """
    domain = Domain()

    domain.variable('x', [0, 1], grid_res)
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
    
    # coefficients for BC
    
    a1, a2, a3 = [1, 2, 1]
    
    b1, b2, b3 = [2, 1, 3]
    
    r1, r2 = [5, 5]
    
    """
    Boundary x=0
    """

    boundaries = Conditions()

    # operator a1*d2u/dx2+a2*du/dx+a3*u
    bop1 = {
        'a1*d2u/dx2':
            {
                'coeff': a1,
                'd2u/dx2': [0, 0],
                'pow': 1
            },
        'a2*du/dx':
            {
                'coeff': a2,
                'du/dx': [0],
                'pow': 1
            },
        'a3*u':
            {
                'coeff': a3,
                'u': [None],
                'pow': 1
            }
    }
    
    boundaries.operator({'x': 0, 't': [0, 1]}, operator=bop1, value=0)
    
    """
    Boundary x=1
    """
    
    # operator b1*d2u/dx2+b2*du/dx+b3*u
    bop2 = {
        'b1*d2u/dx2':
            {
                'coeff': b1,
                'd2u/dx2': [0, 0],
                'pow': 1
            },
        'b2*du/dx':
            {
                'coeff': b2,
                'du/dx': [0],
                'pow': 1
            },
        'b3*u':
            {
                'coeff': b3,
                'u': [None],
                'pow': 1
            }
    }
    
    boundaries.operator({'x': 1, 't': [0, 1]}, operator=bop2, value=0)
    
    """
    Another boundary x=1
    """
    
    # operator r1*du/dx+r2*u
    bop3 = {
        'r1*du/dx':
            {
                'coeff': r1,
                'du/dx': [0],
                'pow': 1
            },
        'r2*u':
            {
                'coeff': r2,
                'u': [None],
                'pow': 1
            }
    }

    boundaries.operator({'x': 1, 't': [0, 1]}, operator=bop3, value=0)
    
    """
    Initial conditions at t=0
    """

    boundaries.dirichlet({'x': [0, 1], 't': 0}, value=0)

    
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
    
    ## domain grid for equation coeff
    grid = domain.build('mat')

    # -sin(x)cos(t)
    def c1(grid):
        return (-1) * torch.sin(grid[0]) * torch.cos(grid[1])
    
    # operator is du/dt+6u*(du/dx)+d3u/dx3-sin(x)*cos(t)=0
    kdv = {
        '1*du/dt**1':
            {
                'coeff': 1,
                'du/dt': [1],
                'pow': 1
            },
        '6*u**1*du/dx**1':
            {
                'coeff': 6,
                'u*du/dx': [[None], [0]],
                'pow': [1,1]
            },
        'd3u/dx3**1':
            {
                'coeff': 1,
                'd3u/dx3': [0, 0, 0],
                'pow': 1
            },
        '-sin(x)cos(t)':
            {
                'coeff': c1(grid),
                'u': [None],
                'pow': 0
            }
    }
    
    equation = Equation()
    equation.add(kdv)
    
    
    """
    Solving equation
    """
    for _ in range(10):

        sln=np.genfromtxt(os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'wolfram_sln/KdV_sln_'+str(grid_res)+'.csv')),delimiter=',')

        sln_torch=torch.from_numpy(sln)
    
        start = time.time()

        net = mat_model(domain, equation)*0

        model = Model(net, domain, equation, boundaries)    

        model.compile('mat', lambda_operator=1, lambda_bound=10, derivative_points=2)

        img_dir=os.path.join(os.path.dirname( __file__ ), 'kdv_img_mat')

        cb_cache = cache.Cache(cache_verbose=True, model_randomize_parameter=1e-5)

        cb_es = early_stopping.EarlyStopping(eps=1e-8,
                                     loss_window=100,
                                     no_improvement_patience=1000,
                                     patience=5,
                                     randomize_parameter=1e-5,
                                     info_string_every=100)

        cb_plots = plot.Plots(save_every=100, print_every=None, img_dir=img_dir)

        optimizer = Optimizer('LBFGS', {'lr': 0.5})

        model.train(optimizer, 5e6, save_model=True, callbacks=[cb_es, cb_plots, cb_cache])

        end = time.time()

        model1 = torch.transpose(net, 0, 1)
        error_rmse=np.sqrt(np.mean((sln_torch.cpu().numpy().reshape(-1)-model1.detach().cpu().numpy().reshape(-1))**2))
    
        exp_dict_list.append({'grid_res':grid_res, 'time':end - start, 'RMSE':error_rmse, 'type':'kdv_eqn'})
        
        print('Time taken {}= {}'.format(grid_res, end - start))
        print('RMSE {}= {}'.format(grid_res, error_rmse))

#result_assessment=pd.DataFrame(exp_dict_list)
#result_assessment.to_csv('results_kdv_matrix.csv')

#result_assessment.boxplot(by='grid_res',column='time',showfliers=False,figsize=(20,10),fontsize=42)

#result_assessment.boxplot(by='grid_res',column='RMSE',figsize=(20,10),fontsize=42)
