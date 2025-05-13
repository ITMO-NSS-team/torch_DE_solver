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
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device

solver_device('gpu')


def exact_func(grid):
    x, y, t = grid[:, 0], grid[:, 1], grid[:, 2]

    sln = torch.sin(20 * np.pi * x) * torch.sin(np.pi * y) * \
          torch.exp(-(20 * np.pi ** 2 / (500 * np.pi) ** 2 + np.pi ** 2 * 1 / np.pi ** 2) * t)

    return sln


def heat_2d_multi_scale_experiment(grid_res):
    exp_dict_list = []

    x_min, x_max = 0, 1
    y_min, y_max = 0, 1
    t_max = 5

    pde_dim_in = 3
    pde_dim_out = 1

    domain = Domain()

    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('y', [y_min, y_max], grid_res)
    domain.variable('t', [0, t_max], grid_res)

    boundaries = Conditions()

    # Initial condition ################################################################################################

    # u(x, y, 0)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': [y_min, y_max], 't': 0}, value=exact_func)

    # # u_t(x, 0) = 0
    # bop = {
    #     'du/dt':
    #         {
    #             'coeff': 1,
    #             'term': [2],
    #             'pow': 1,
    #             'var': 0
    #         }
    # }
    # boundaries.operator({'x': [x_min, x_max], 'y': [y_min, y_max], 't': 0}, operator=bop, value=0)

    # Boundary conditions ##############################################################################################

    # u(x_min, y, t)
    boundaries.dirichlet({'x': x_min, 'y': [y_min, y_max], 't': [0, t_max]}, value=0)
    # u(x, y_min, t)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': y_min, 't': [0, t_max]}, value=0)
    # u(x_min, y, t)
    boundaries.dirichlet({'x': x_max, 'y': [y_min, y_max], 't': [0, t_max]}, value=0)
    # u(x, x_max, t)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': y_max, 't': [0, t_max]}, value=0)

    equation = Equation()

    # Operator: u_t - 1 / (500 * pi)**2 * u_xx - 1 / pi**2 * u_yy = 0

    heat_MS = {
        'du/dt**1':
            {
                'coeff': 1,
                'term': [2],
                'pow': 1
            },
        '-1 / (500 * pi) ** 2 * d2u/dx2**1':
            {
                'coeff': -1 / (500 * torch.pi) ** 2,
                'term': [0, 0],
                'pow': 1
            },
        '-1 / pi ** 2 * d2u/dy2**1':
            {
                'coeff': -1 / torch.pi ** 2,
                'term': [1, 1],
                'pow': 1
            }
    }

    equation.add(heat_MS)

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

    model.compile('autograd', lambda_operator=1, lambda_bound=100)

    img_dir = os.path.join(os.path.dirname(__file__), 'heat_2d_multi_scale_img')

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
                          img_dim='2d',
                          scatter_flag=False,
                          plot_axes=[0, 1],
                          fixed_axes=[2],
                          n_samples=4,
                          img_rows=2,
                          img_cols=2)

    optimizer = Optimizer('Adam', {'lr': 1e-3})

    callbacks = [cb_cache, cb_es, cb_plots]

    model.train(optimizer, 5e5, save_model=True, callbacks=callbacks)

    end = time.time()

    grid = domain.build('NN').to('cuda')
    net = net.to('cuda')

    exact = exact_func(grid).reshape(-1, 1)
    net_predicted = net(grid)

    error_rmse = torch.sqrt(torch.mean((exact - net_predicted) ** 2))

    exp_dict_list.append({
        'grid_res': grid_res,
        'time': end - start,
        'RMSE': error_rmse.detach().cpu().numpy(),
        'type': 'heat_2d_multi_scale',
        'cache': True
    })

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))

    return exp_dict_list


nruns = 10

exp_dict_list = []

for grid_res in range(20, 201, 20):
    for _ in range(nruns):
        exp_dict_list.append(heat_2d_multi_scale_experiment(grid_res))

import pandas as pd

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)
df.to_csv('examples/benchmarking_data/heat_2d_multi_scale_experiment_20_200_cache={}.csv'.format(str(True)))
