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
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.utils import exact_solution_data

solver_device('gpu')

datapath = "heat_darcy.dat"
heat_2d_coef = np.loadtxt("heat_2d_coef_256.dat")

A = 200
m1, m2, m3 = 1, 5, 1


def func(grid):
    x, y, t = grid[:, 0], grid[:, 1], grid[:, 2]
    sln = A * torch.sin(np.pi * m1 * x) * torch.sin(np.pi * m2 * y) * torch.exp(np.pi * m3 * t)
    return sln


def a_coeff(grid):
    return -torch.tensor(
        interpolate.griddata(heat_2d_coef[:, :2],
                             heat_2d_coef[:, 2],
                             grid.detach().cpu().numpy()[:, :2],
                             method='nearest'),
        dtype=torch.float32
    ).unsqueeze(dim=1)


def heat_2d_varying_coeff_experiment(grid_res):
    exp_dict_list = []

    x_min, x_max = 0, 1
    y_min, y_max = 0, 1
    t_max = 5
    # grid_res = 10

    pde_dim_in = 3
    pde_dim_out = 1

    domain = Domain()

    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('y', [y_min, y_max], grid_res)
    domain.variable('t', [0, t_max], grid_res)

    boundaries = Conditions()

    # Initial condition: ###############################################################################################

    # u(x, y, 0)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': [y_min, y_max], 't': 0}, value=0)

    # Boundary conditions: #############################################################################################

    # u(x_min, y, t)
    boundaries.dirichlet({'x': x_min, 'y': [y_min, y_max], 't': [0, t_max]}, value=0)
    # u(x, y_min, t)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': y_min, 't': [0, t_max]}, value=0)
    # u(x_max, y, t)
    boundaries.dirichlet({'x': x_max, 'y': [y_min, y_max], 't': [0, t_max]}, value=0)
    # u(x, x_min, t)
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
                'pow': 0,
            }
    }

    equation.add(heat_VC)

    neurons = 100

    net = torch.nn.Sequential(
        torch.nn.Linear(pde_dim_in, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, pde_dim_out)
    )

    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    start = time.time()

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

    cb_plots = plot.Plots(save_every=100,
                          print_every=None,
                          img_dir=img_dir,
                          img_dim='2d')  # 3 image dimension options: 3d, 2d, 2d_scatter

    optimizer = Optimizer('Adam', {'lr': 1e-3})

    callbacks = [cb_cache, cb_es, cb_plots]

    model.train(optimizer, 5e5, save_model=True, callbacks=callbacks)

    end = time.time()

    grid = domain.build('NN').to('cuda')
    net = net.to('cuda')

    exact = exact_solution_data(grid, datapath, pde_dim_in, pde_dim_out, t_dim_flag=True).reshape(-1, 1)
    net_predicted = net(grid)

    error_rmse = torch.sqrt(torch.mean((exact - net_predicted) ** 2))

    exp_dict_list.append({
        'grid_res': grid_res,
        'time': end - start,
        'RMSE': error_rmse.detach().cpu().numpy(),
        'type': 'heat_2d_varying_coeff',
        'cache': True
    })

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))

    return exp_dict_list


nruns = 10

exp_dict_list = []

for grid_res in range(20, 201, 20):
    for _ in range(nruns):
        exp_dict_list.append(heat_2d_varying_coeff_experiment(grid_res))

import pandas as pd

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)
df.to_csv('examples/benchmarking_data/heat_2d_varying_coeff_experiment_20_200_cache={}.csv'.format(str(True)))
