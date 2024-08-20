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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model

from tedeous.callbacks import early_stopping, plot, cache
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device

import efficient_kan


"""
Preparing grid

Grid is an essentially torch.Tensor of a n-D points where n is the problem
dimensionality
"""

solver_device('gpu')


def func(grid):
    x, t = grid[:, 0], grid[:, 1]
    sln = torch.cos(2 * np.pi * t) * torch.sin(np.pi * x)
    return sln


def wave_experiment(grid_res):
    exp_dict_list = []

    domain = Domain()
    domain.variable('x', [0, 1], grid_res)
    domain.variable('t', [0, 1], grid_res)

    """
    Preparing boundary conditions (BC)

    For every boundary we define three items

    bnd=torch.Tensor of a boundary n-D points where n is the problem
    dimensionality

    bop=dict in form {'term1':term1,'term2':term2} -> term1+term2+...=0

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
    boundaries.dirichlet({'x': [0, 1], 't': 0}, value=func)

    # Initial conditions at t=1
    # u(1,x)=sin(pi*x)
    bop2 = {
        'du/dt':
            {
                'coeff': 1,
                'du/dx': [1],
                'pow': 1,
                'var': 0
            }
    }
    boundaries.operator({'x': [0, 1], 't': 0}, operator=bop2, value=0)

    # Boundary conditions at x=0
    boundaries.dirichlet({'x': 0, 't': [0, 1]}, value=func)

    # Boundary conditions at x=1
    boundaries.dirichlet({'x': 1, 't': [0, 1]}, value=func)


    """
    Defining wave equation

    Operator has the form

    op=dict in form {'term1':term1,'term2':term2}-> term1+term2+...=0

    NB! dictionary keys at the current time serve only for user-friendly
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
        'd2u/dt2**1':
            {
                'coeff': 1,
                'd2u/dt2': [1, 1],
                'pow': 1
            },
        '-C*d2u/dx2**1':
            {
                'coeff': -4.,
                'd2u/dx2': [0, 0],
                'pow': 1
            }
    }

    equation.add(wave_eq)

    net = efficient_kan.KAN(
        [2, 100, 100, 100, 1],
        grid_size=20,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.Tanh,
        grid_eps=0.02,
        grid_range=[-2, 2]
    )

    start = time.time()

    model = Model(net, domain, equation, boundaries)

    model.compile("autograd", lambda_operator=1, lambda_bound=100)

    cb_es = early_stopping.EarlyStopping(eps=1e-5, randomize_parameter=1e-6, info_string_every=10)

    cb_cache = cache.Cache(cache_verbose=True, model_randomize_parameter=1e-6)

    img_dir = os.path.join(os.path.dirname( __file__ ), 'wave_img_efficient_kan')

    cb_plots = plot.Plots(save_every=500, print_every=500, img_dir=img_dir)

    optimizer = Optimizer('Adam', {'lr': 1e-4})

    model.train(optimizer, 5e6, save_model=True, callbacks=[cb_es, cb_plots, cb_cache])

    end = time.time()

    grid = domain.build('NN').to('cuda')
    net = net.to('cuda')

    error_rmse = torch.sqrt(torch.mean((func(grid).reshape(-1, 1) - net(grid)) ** 2))

    exp_dict_list.append({'grid_res': grid_res, 'time': end - start, 'RMSE': error_rmse.detach().cpu().numpy(),
                          'type': 'wave_eqn_physical', 'cache': True})

    print('Time taken {} = {}'.format(grid_res, end - start))
    print('RMSE {} = {}'.format(grid_res, error_rmse))

    return exp_dict_list


nruns = 10

exp_dict_list = []

for grid_res in range(10, 101, 10):
    for _ in range(nruns):
        exp_dict_list.append(wave_experiment(grid_res))

import pandas as pd

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)
# df.boxplot(by='grid_res',column='time',fontsize=42,figsize=(20,10))
# df.boxplot(by='grid_res',column='RMSE',fontsize=42,figsize=(20,10),showfliers=False)
df.to_csv('examples/benchmarking_data/wave_experiment_physical_10_100_cache={}.csv'.format(str(True)))