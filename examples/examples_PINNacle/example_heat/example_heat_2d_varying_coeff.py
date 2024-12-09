# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:33:44 2021

@author: user
"""
import torch
import numpy as np
import os
import sys
from scipy import interpolate

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
import time

solver_device('cpu')

heat_2d_coef = np.loadtxt("heat_2d_coef_256.dat")


def func(grid, A=200, m1=1, m2=5, m3=1):
    x, y, t = grid[:, 0], grid[:, 1], grid[:, 2]
    sln = A * torch.sin(np.pi * m1 * x) * torch.sin(np.pi * m2 * y) * torch.exp(np.pi * m3 * t)
    return sln


def a_coeff(grid):
    a = -torch.tensor(
        interpolate.griddata(heat_2d_coef[:, 0:2],
                             heat_2d_coef[:, 2],
                             grid.detach().cpu().numpy()[:, 0:2],
                             method='nearest'),
        dtype=torch.float32
    ).unsqueeze(dim=1)
    return a


def heat_2d_experiment(grid_res, CACHE):
    exp_dict_list = []

    x_min, x_max = 0, 1
    y_min, y_max = 0, 1
    t_max = 5
    grid_res *= 1

    domain = Domain()

    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('y', [y_min, y_max], grid_res)
    domain.variable('t', [0, t_max], grid_res)

    boundaries = Conditions()

    # Initial condition: ###############################################################################################

    # u(x, y, 0)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': [y_min, y_max], 't': 0}, value=0)

    # Boundary conditions: #############################################################################################

    # u(0, y, t)
    boundaries.dirichlet({'x': x_min, 'y': [y_min, y_max], 't': [0, t_max]}, value=0)
    # u(x, 0, t)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': y_min, 't': [0, t_max]}, value=0)
    # u(1, y, t)
    boundaries.dirichlet({'x': x_max, 'y': [y_min, y_max], 't': [0, t_max]}, value=0)
    # u(x, 1, t)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': y_max, 't': [0, t_max]}, value=0)

    equation = Equation()

    # Operator: du/dt - ∇(a(x)∇u) = f(x, y, t)

    forcing_term = lambda grid: -func(grid)

    heat_VC = {
        'du/dt**1':
            {
                'coeff': 1,
                'term': [2],
                'pow': 1
            },
        '-a(x) * d2u/dx2**1':
            {
                'coeff': a_coeff,
                'term': [0, 0],
                'pow': 1
            },
        '-a(x) * d2u/dy2**1':
            {
                'coeff': a_coeff,
                'term': [1, 1],
                'pow': 1
            },
        '-f(x, y, t)':
            {
                'coeff': forcing_term,
                'term': [None],
                'pow': 1,
            }
    }

    equation.add(heat_VC)

    neurons = 100

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
        torch.nn.Linear(neurons, 1)
    )

    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    # t = domain.variable_dict['t']
    # h = abs((t[1] - t[0]).item())

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=1, lambda_bound=1)

    img_dir = os.path.join(os.path.dirname(__file__), 'heat_2d_varying_coeff_img')

    cb_cache = cache.Cache(cache_verbose=True, model_randomize_parameter=1e-6)

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=100,
                                         no_improvement_patience=1000,
                                         patience=5,
                                         randomize_parameter=1e-6,
                                         info_string_every=10)

    cb_plots = plot.Plots(save_every=500,
                          print_every=None,
                          img_dir=img_dir,
                          img_dim='2d')  # 3 image dimension options: 3d, 2d, 2d_scatter

    optimizer = Optimizer('Adam', {'lr': 5e-3})

    if CACHE:
        callbacks = [cb_cache, cb_es, cb_plots]
    else:
        callbacks = [cb_es, cb_plots]

    start = time.time()

    model.train(optimizer, 1e6, save_model=CACHE, callbacks=callbacks)

    end = time.time()

    grid = domain.build('NN').to('cuda')

    error_rmse = torch.sqrt(torch.mean((func(grid).reshape(-1, 1) - net(grid)) ** 2))

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
