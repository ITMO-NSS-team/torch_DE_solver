# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:33:44 2021

@author: user
"""
import torch
import os
import sys
import time
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model

from tedeous.callbacks import early_stopping, plot, cache
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.utils import exact_solution_data

solver_device('gpu')

data_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../PINNacle_data/burgers1d.npy"))

mu = 0.01 / np.pi


def burgers_1d_experiment(grid_res):
    exp_dict_list = []

    x_min, x_max = -1, 1
    t_max = 1

    pde_dim_in = 2
    pde_dim_out = 1

    domain = Domain()
    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('t', [0, t_max], grid_res)

    boundaries = Conditions()

    # Initial conditions ###############################################################################################

    # u(x, 0) = -sin(pi * x)
    boundaries.dirichlet({'x': [x_min, x_max], 't': 0}, value=lambda grid: -torch.sin(np.pi * grid[:, 0]))

    # Boundary conditions ##############################################################################################

    # u(x_min, t) = 0
    boundaries.dirichlet({'x': x_min, 't': [0, t_max]}, value=0)

    # u(x_max, t) = 0
    boundaries.dirichlet({'x': x_max, 't': [0, t_max]}, value=0)

    equation = Equation()

    # Operator: u_t + u * u_x - mu * u_xx = 0

    burgers_eq = {
        'du/dt**1':
            {
                'coeff': 1.,
                'du/dt': [1],
                'pow': 1,
                'var': 0
            },
        '+u*du/dx':
            {
                'coeff': 1,
                'u*du/dx': [[None], [0]],
                'pow': [1, 1],
                'var': [0, 0]
            },
        '-mu*d2u/dx2':
            {
                'coeff': -mu,
                'd2u/dx2': [0, 0],
                'pow': 1,
                'var': 0
            }
    }

    equation.add(burgers_eq)

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
        torch.nn.Linear(neurons, pde_dim_out)
    )

    start = time.time()

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=1, lambda_bound=10)

    cb_cache = cache.Cache(cache_verbose=False, model_randomize_parameter=1e-5)

    cb_es = early_stopping.EarlyStopping(eps=1e-5,
                                         randomize_parameter=1e-5,
                                         info_string_every=50)

    img_dir = os.path.join(os.path.dirname(__file__), 'burgers_1d_img')

    cb_plots = plot.Plots(save_every=500,
                          print_every=None,
                          img_dir=img_dir,
                          img_dim='3d',
                          scatter_flag=False)

    optimizer = Optimizer('Adam', {'lr': 1e-4})

    model.train(optimizer, 5e5, save_model=True, callbacks=[cb_es, cb_plots, cb_cache])

    end = time.time()

    grid = domain.build('NN').to('cuda')
    net = net.to('cuda')

    exact = exact_solution_data(grid, data_file, pde_dim_in, pde_dim_out, t_dim_flag=True).reshape(-1, 1)
    net_predicted = net(grid)

    error_rmse = torch.sqrt(torch.mean((exact - net_predicted) ** 2))

    exp_dict_list.append({
        'grid_res': grid_res,
        'time': end - start,
        'RMSE': error_rmse.detach().cpu().numpy(),
        'type': 'burgers_1d',
        'cache': True
    })

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))

    return exp_dict_list


nruns = 10

exp_dict_list = []

for grid_res in range(50, 501, 50):
    for _ in range(nruns):
        exp_dict_list.append(burgers_1d_experiment(grid_res))

import pandas as pd

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)
df.to_csv('examples/benchmarking_data/burgers_1d_experiment_50_500_cache={}.csv'.format(str(True)))
