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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.append('../')

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tedeous.input_preprocessing import Equation
from tedeous.solver import Solver
from tedeous.solution import Solution
from tedeous.device import solver_device
from tedeous.optimizers import ZO_SignSGD, ZO_AdaMM
"""
Preparing grid

Grid is an essentially torch.Tensor  of a n-D points where n is the problem
dimensionality
"""

solver_device('gpu')


def func(grid):
    x, t = grid[:, 0], grid[:, 1]
    sln = np.cos(2 * np.pi * t) * np.sin(np.pi * x)
    return sln


def wave_experiment(grid_res, CACHE):
    exp_dict_list = []

    x_grid = np.linspace(0, 1, grid_res + 1)
    t_grid = np.linspace(0, 1, grid_res + 1)

    x = torch.from_numpy(x_grid)
    t = torch.from_numpy(t_grid)

    # h=abs((t[1]-t[0]).item())

    grid = torch.cartesian_prod(x, t).float()

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

    # Initial conditions at t=0
    bnd1 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()

    # u(0,x)=sin(pi*x)
    bndval1 = func(bnd1)

    ## Initial conditions at t=1
    # bnd2 = torch.cartesian_prod(x, torch.from_numpy(np.array([1], dtype=np.float64))).float()

    ## u(1,x)=sin(pi*x)
    # bndval2 = func(bnd2)

    bnd2 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()

    bop2 = {
        'du/dt':
            {
                'coeff': 1,
                'du/dx': [1],
                'pow': 1,
                'var': 0
            }
    }

    bndval2 = torch.from_numpy(np.zeros(len(bnd2), dtype=np.float64))

    # Boundary conditions at x=0
    bnd3 = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), t).float()

    # u(0,t)=0
    bndval3 = func(bnd3)

    # Boundary conditions at x=1
    bnd4 = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)), t).float()

    # u(1,t)=0
    bndval4 = func(bnd4)

    # Putting all bconds together
    bconds = [[bnd1, bndval1, 'dirichlet'],
              # [bnd2, bndval2, 'dirichlet'],
              [bnd2, bop2, bndval2, 'operator'],
              [bnd3, bndval3, 'dirichlet'],
              [bnd4, bndval4, 'dirichlet'],
              ]

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
        'd2u/dt2**1':
            {
                'coeff': 1.,
                'd2u/dt2': [1, 1],
                'pow': 1
            },
        '-C*d2u/dx2**1':
            {
                'coeff': -4,
                'd2u/dx2': [0, 0],
                'pow': 1
            }
    }

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 1)
    )

    start = time.time()

    equation = Equation(grid, wave_eq, bconds).set_strategy('NN')

    img_dir = os.path.join(os.path.dirname(__file__), 'wave_example_physics_img')

    if not (os.path.isdir(img_dir)):
        os.mkdir(img_dir)

    config = {
        "seed": 42,
        "batch_size": 128,
        "net": "scalable",
        "scale": 1.0 / 9,
        "opt_params": [1e-4, 0.9, 0.999, 1e-10],
        "optimizer": 'ZO-AdaMM',
        "epochs": 50,
        "dataset": "mnist",
        "zo_optim": True,
        "mu": 1e-4,
        "use_scheduler": True,
        "verbose": True
    }

    optimizer = ZO_AdaMM(model.parameters(), lr=config['opt_params'][0],
                         betas=(config['opt_params'][1], config['opt_params'][2]),
                         eps=config['opt_params'][3],
                         mu=config['mu'])

    model = Solver(grid, equation, model, 'NN').solve(lambda_bound=100, verbose=1, learning_rate=1e-4,
                                                      eps=1e-5, tmin=1000, tmax=1e5, use_cache=True, cache_verbose=True,
                                                      save_always=True, print_every=1,
                                                      model_randomize_parameter=1e-6,
                                                      optimizer_mode=optimizer, no_improvement_patience=1000,
                                                      step_plot_print=True, step_plot_save=True,
                                                      image_save_dir=img_dir)

    end = time.time()

    error_rmse = torch.sqrt(torch.mean((func(grid).reshape(-1, 1) - model(grid)) ** 2))

    end_loss, _ = Solution(grid, equation, model, 'NN', lambda_bound=100,lambda_operator=1, weak_form=None).evaluate()

    exp_dict_list.append({'grid_res': grid_res, 'time': end - start, 'RMSE': error_rmse.detach().cpu().numpy(),
                          'loss': end_loss.detach().cpu().numpy(), 'type': 'wave_eqn_physical', 'cache': True})

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))
    print('loss {}= {}'.format(grid_res, end_loss))
    return exp_dict_list


nruns = 10

exp_dict_list = []

CACHE = True

for grid_res in range(10, 101, 10):
    for _ in range(nruns):
        exp_dict_list.append(wave_experiment(grid_res, CACHE))

import pandas as pd

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)
# df.boxplot(by='grid_res',column='time',fontsize=42,figsize=(20,10))
# df.boxplot(by='grid_res',column='RMSE',fontsize=42,figsize=(20,10),showfliers=False)
df.to_csv('examples/benchmarking_data/wave_experiment_physical_10_100_cache={}.csv'.format(str(CACHE)))