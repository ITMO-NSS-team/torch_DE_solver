# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:33:44 2021

@author: user
"""
import torch
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
from tedeous.utils import exact_solution_data

solver_device('gpu')
datapath = "grayscott.dat"

b = 0.04
d = 0.1
epsilon_1 = 1e-5
epsilon_2 = 5e-6


def DR_2d_gray_scott_experiment(grid_res):
    exp_dict_list_u, exp_dict_list_v = [], []

    x_min, x_max = -1, 1
    y_min, y_max = -1, 1
    t_max = 200
    # grid_res = 20

    pde_dim_in = 3
    pde_dim_out = 2

    domain = Domain()
    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('y', [y_min, y_max], grid_res)
    domain.variable('t', [0, t_max], grid_res)

    boundaries = Conditions()

    # Initial conditions ###############################################################################################

    u_init_func = lambda grid: 1 - torch.exp(-80 * ((grid[:, 0] + 0.05)**2 + (grid[:, 1] + 0.02)**2))
    v_init_func = lambda grid: torch.exp(-80 * ((grid[:, 0] + 0.05)**2 + (grid[:, 1] + 0.02)**2))

    # u(x, y, 0) = u_init_func(x, y, 0)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': [y_min, y_max], 't': 0}, value=u_init_func, var=0)

    # v(x, y, 0) = v_init_func(x, y, 0)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': [y_min, y_max], 't': 0}, value=v_init_func, var=1)

    equation = Equation()

    # Operator 1:  ut = ε1 * ∆u + b * (1 − u) − u * v**2

    diffusion_reaction_u = {
        'du/dt**1':
            {
                'coeff': 1,
                'term': [2],
                'pow': 1,
                'var': 0
            },
        '-epsilon_1 * d2u/dx2**1':
            {
                'coeff': -epsilon_1,
                'term': [0, 0],
                'pow': 1,
                'var': 0
            },
        '-epsilon_1 * d2u/dy2**1':
            {
                'coeff': -epsilon_1,
                'term': [1, 1],
                'pow': 1,
                'var': 0
            },
        '-b':
            {
                'coeff': -b,
                'term': [None],
                'pow': 0,
                'var': 0
            },
        'b * u':
            {
                'coeff': b,
                'term': [None],
                'pow': 1,
                'var': 0
            },
        'u * v ** 2':
            {
                'coeff': 1,
                'term': [[None], [None]],
                'pow': [1, 2],
                'var': [0, 1]
            }
    }

    # Operator 2:  ut = ε2 * ∆v - d * u + u * v**2

    diffusion_reaction_v = {
        'dv/dt**1':
            {
                'coeff': 1,
                'term': [2],
                'pow': 1,
                'var': 1
            },
        '-epsilon_2 * d2v/dx2**1':
            {
                'coeff': -epsilon_2,
                'term': [0, 0],
                'pow': 1,
                'var': 1
            },
        '-epsilon_2 * d2v/dy2**1':
            {
                'coeff': -epsilon_2,
                'term': [1, 1],
                'pow': 1,
                'var': 1
            },
        'd * v':
            {
                'coeff': d,
                'term': [None],
                'pow': 1,
                'var': 1
            },
        '-u * v ** 2':
            {
                'coeff': -1,
                'term': [[None], [None]],
                'pow': [1, 2],
                'var': [0, 1]
            }
    }

    equation.add(diffusion_reaction_u)
    equation.add(diffusion_reaction_v)

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

    img_dir = os.path.join(os.path.dirname(__file__), 'diffusion_reaction_2d_gray_scott_img')

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

    optimizer = Optimizer('Adam', {'lr': 1e-4})

    model.train(optimizer, 5e5, save_model=True, callbacks=[cb_es, cb_plots, cb_cache])

    end = time.time()

    grid = domain.build('NN').to('cuda')
    net = net.to('cuda')

    predicted_u, predicted_v = net(grid)[:, 0], net(grid)[:, 1]
    exact_u, exact_v = exact_solution_data(grid, datapath, pde_dim_in, pde_dim_out)

    error_rmse_u = torch.sqrt(torch.mean((exact_u - predicted_u) ** 2))
    error_rmse_v = torch.sqrt(torch.mean((exact_v - predicted_v) ** 2))

    exp_dict_list_u.append({
        'grid_res': grid_res,
        'time': end - start,
        'RMSE_u_func': error_rmse_u.detach().cpu().numpy(),
        'type': 'diffusion_reaction_2d_gray_scott',
        'cache': True
    })
    exp_dict_list_v.append({
        'grid_res': grid_res,
        'time': end - start,
        'RMSE_v_func': error_rmse_v.detach().cpu().numpy(),
        'type': 'diffusion_reaction_2d_gray_scott',
        'cache': True
    })

    print('Time taken {}= {}'.format(grid_res, end - start))

    print('RMSE_u_func {}= {}'.format(grid_res, error_rmse_u))
    print('RMSE_v_func {}= {}'.format(grid_res, error_rmse_v))

    return exp_dict_list_u, exp_dict_list_v


nruns = 1

exp_dict_list_u, exp_dict_list_v = [], []

for grid_res in range(20, 21, 20):
    for _ in range(nruns):
        list_u, list_v = DR_2d_gray_scott_experiment(grid_res)
        exp_dict_list_u.append(list_u)
        exp_dict_list_v.append(list_v)

import pandas as pd

exp_dict_list_u_flatten = [item for sublist in exp_dict_list_u for item in sublist]
exp_dict_list_v_flatten = [item for sublist in exp_dict_list_v for item in sublist]

df_u = pd.DataFrame(exp_dict_list_u_flatten)
df_v = pd.DataFrame(exp_dict_list_v_flatten)

df_u.to_csv('examples/benchmarking_data/DR_2d_gray_scott_experiment_20_200_cache_u_func={}.csv'.format(str(True)))
df_v.to_csv('examples/benchmarking_data/DR_2d_gray_scott_experiment_20_200_cache_v_func={}.csv'.format(str(True)))
