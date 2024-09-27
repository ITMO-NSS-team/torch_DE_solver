# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:33:44 2021

@author: user
"""
import torch
import numpy as np
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
import time

solver_device('cpu')


def func(grid):
    x, y, t = grid[:, 0], grid[:, 1], grid[:, 2]

    temp = 100

    # # var 1
    # # Stationary part of solve
    # sln = temp + x + y
    #
    # # Non-stationary part of solve
    # for i in range(1, 100):
    #     eig_value = (2 * i - 1) ** 2
    #     exp_term = np.exp(-1 / (4 * np.pi ** 2) * t * eig_value)
    #     sin_x = np.sin(np.pi / 2 * x * (2 * i - 1))
    #     sin_y = np.sin(np.pi / 2 * y * (2 * i - 1))
    #
    #     sln += 8 * exp_term * ((-1) ** i + temp / 2 * np.pi * (1 - 2 * i)) * sin_x * sin_y / (np.pi - 2 * np.pi * i) ** 2

    # # var 2
    # sln = torch.sin(np.pi * x) * torch.sinh(np.pi * y) * torch.exp(-2 * np.pi ** 2 * t)

    # var 3
    sln = torch.sin(np.pi * x) * torch.sin(np.pi * y) * torch.exp(-2 * np.pi ** 2 * t)

    # # var 4
    # sln = 10 * torch.sin(np.pi * x) ** 10 * torch.sin(np.pi * y) ** 10 * torch.sin(np.pi * t) ** 10

    return sln


def heat_2d_experiment(grid_res, CACHE):
    exp_dict_list = []

    domain = Domain()

    domain.variable('x', [0, 1], grid_res)
    domain.variable('y', [0, 1], grid_res)
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

    # Initial conditions at t=0
    boundaries.dirichlet({'x': [0, 1], 'y': [0, 1], 't': 0}, value=0)
    bop = {
        'du/dt':
            {
                'coeff': 1,
                'du/dt': [2],
                'pow': 1,
                'var': 0
            }
    }
    boundaries.operator({'x': [0, 1], 'y': [0, 1], 't': 0}, operator=bop, value=0)

    # Boundary conditions at x=0
    boundaries.dirichlet({'x': 0, 'y': [0, 1], 't': [0, 1]}, value=0)

    # Boundary conditions at y=0
    boundaries.dirichlet({'x': [0, 1], 'y': 0, 't': [0, 1]}, value=0)

    # Boundary conditions at x=1
    boundaries.dirichlet({'x': 1, 'y': [0, 1], 't': [0, 1]}, value=0)

    # Boundary conditions at y=1
    boundaries.dirichlet({'x': [0, 1], 'y': 1, 't': [0, 1]}, value=100)

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

    # operator is 1*du/dt-1*d2u/dx2-1*d2u/dy2=0
    heat_eq = {
        'du/dt**1':
            {
                'coeff': 1,
                'du/dt': [2],
                'pow': 1,
                'var': 0
            },
        '-d2u/dx2**1':
            {
                'coeff': -1,
                'd2u/dx2': [0, 0],
                'pow': 1,
                'var': 0
            },
        '-d2u/dy2**1':
            {
                'coeff': -1,
                'd2u/dy2': [1, 1],
                'pow': 1,
                'var': 0
            }
    }

    equation.add(heat_eq)

    neurons = 50

    net = torch.nn.Sequential(
        torch.nn.Linear(3, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, 1)
    )

    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

    t = domain.variable_dict['t']

    h = abs((t[1] - t[0]).item())

    model = Model(net, domain, equation, boundaries)

    model.compile('NN', lambda_operator=1, lambda_bound=10, h=h)

    img_dir = os.path.join(os.path.dirname(__file__), 'heat_2d_NN_img')

    cb_cache = cache.Cache(cache_verbose=True, model_randomize_parameter=1e-6)

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=100,
                                         no_improvement_patience=1000,
                                         patience=5,
                                         randomize_parameter=1e-6,
                                         info_string_every=10)

    cb_plots = plot.Plots(save_every=20, print_every=None, img_dir=img_dir)

    optimizer = Optimizer('Adam', {'lr': 10e-4})
    # optimizer = Optimizer('LBFGS', {'lr': 10e-4})
    # optimizer = Optimizer('RMSprop', {'lr': 10e-4})

    if CACHE:
        callbacks = [cb_cache, cb_es, cb_plots]
    else:
        callbacks = [cb_es, cb_plots]

    start = time.time()

    model.train(optimizer, 1e6, save_model=CACHE, callbacks=callbacks)

    end = time.time()

    grid = domain.build('NN').to('cuda')

    error_rmse = torch.sqrt(torch.mean(((func(grid) - net(grid)) / 100) ** 2))

    exp_dict_list.append(
        {'grid_res': grid_res, 'time': end - start, 'RMSE': error_rmse.detach().numpy(), 'type': 'wave_eqn',
         'cache': CACHE})

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))
    return exp_dict_list


nruns = 10

exp_dict_list = []

CACHE = True

for grid_res in range(10, 101, 10):
    for _ in range(nruns):
        exp_dict_list.append(heat_2d_experiment(grid_res, CACHE))

import pandas as pd

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)
df.boxplot(by='grid_res', column='time', fontsize=42, figsize=(20, 10))
df.boxplot(by='grid_res', column='RMSE', fontsize=42, figsize=(20, 10), showfliers=False)
df.to_csv('benchmarking_data/heat_experiment_10_100_cache={}.csv'.format(str(CACHE)))






