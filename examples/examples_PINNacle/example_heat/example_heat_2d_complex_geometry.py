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

solver_device('gpu')


def func(grid):
    x, y, t = grid[:, 0], grid[:, 1], grid[:, 2]
    sln = torch.sin(np.pi * x) * torch.sin(np.pi * y) * torch.exp(np.pi * t)
    return sln


def heat_2d_experiment(grid_res, CACHE):
    exp_dict_list = []

    x_min, x_max = -8, 8
    y_min, y_max = -12, 12
    t_max = 3
    grid_res = 160

    domain = Domain()

    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('y', [y_min, y_max], grid_res)
    domain.variable('t', [0, t_max], grid_res)

    x = domain.variable_dict['x']
    y = domain.variable_dict['y']

    # Removed domains initialization ###################################################################################

    removed_domains_lst = [
        {'circle': {'center': (-4, 3), 'radius': 1}},
        {'circle': {'center': (-4, 9), 'radius': 1}},
        {'circle': {'center': (-4, -3), 'radius': 1}},
        {'circle': {'center': (-4, -9), 'radius': 1}},
        {'circle': {'center': (4, 3), 'radius': 1}},
        {'circle': {'center': (4, 9), 'radius': 1}},
        {'circle': {'center': (4, -3), 'radius': 1}},
        {'circle': {'center': (4, -9), 'radius': 1}},
        {'circle': {'center': (0, 0), 'radius': 1}},
        {'circle': {'center': (0, -6), 'radius': 1}},
        {'circle': {'center': (0, 6), 'radius': 1}},
        {'circle': {'center': (-3.2, 6), 'radius': 0.4}},
        {'circle': {'center': (-3.2, -6), 'radius': 0.4}},
        {'circle': {'center': (3.2, 6), 'radius': 0.4}},
        {'circle': {'center': (3.2, -6), 'radius': 0.4}},
        {'circle': {'center': (-3.2, 0), 'radius': 0.4}},
        {'circle': {'center': (3.2, 0), 'radius': 0.4}}
    ]

    boundaries = Conditions()

    # Initial condition: ###############################################################################################

    # u(x, y, 0)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': [y_min, y_max], 't': 0},
                         value=torch.sin(20 * np.pi * x) * torch.sin(np.pi * y))

    # Boundary conditions: −n*(−c∇u) = g−qu, c = 1, g = 0.1, q = 1 #####################################################

    def bop_generation(alpha, beta, grid_i):
        bop = {
            'alpha * u':
                {
                    'coeff': alpha,
                    'term': [None],
                    'pow': 1
                },
            'beta * du/dx':
                {
                    'coeff': beta,
                    'term': [grid_i],
                    'pow': 1
                }
        }
        return bop

    c_rec, g_rec, q_rec = 1, 0.1, 1

    bop_x_min = bop_generation(-1, q_rec, 0)
    boundaries.robin({'x': x_min, 'y': [y_min, y_max], 't': [0, t_max]}, operator=bop_x_min, value=g_rec)

    bop_x_max = bop_generation(1, q_rec, 0)
    boundaries.robin({'x': x_max, 'y': [y_min, y_max], 't': [0, t_max]}, operator=bop_x_max, value=g_rec)

    bop_y_min = bop_generation(-1, q_rec, 1)
    boundaries.robin({'x': [x_min, x_max], 'y': y_min, 't': [0, t_max]}, operator=bop_y_min, value=g_rec)

    bop_y_max = bop_generation(1, q_rec, 1)
    boundaries.robin({'x': [x_min, x_max], 'y': y_max, 't': [0, t_max]}, operator=bop_y_max, value=g_rec)

    # CSG boundaries ###################################################################################################

    def bopCSG_generation(x0, y0, r, c, q):
        coeff_x = lambda bnd: c * (bnd[:, 0] - x0) / r
        coeff_y = lambda bnd: c * (bnd[:, 1] - y0) / r

        bop = {
            'q * u':
                {
                    'coeff': q,
                    'term': [None],
                    'pow': 1
                },
            'c * nx * du/dx':
                {
                    'coeff': coeff_x,
                    'term': [0],
                    'pow': 1
                },
            'c * ny * du/dy':
                {
                    'coeff': coeff_y,
                    'term': [1],
                    'pow': 1
                }
        }
        return bop

    def bounds_generation(domains, c, g, q):
        for i, rd in enumerate(domains):
            if list(rd.keys())[0] == 'circle':
                x0, y0 = rd['circle']['center']
                r = rd['circle']['radius']

                bop = bopCSG_generation(x0, y0, r, c, q)
                boundaries.robin(rd, operator=bop, value=g)

    # Large circles
    c_big_circ, g_big_circ, q_big_circ = 1, 5, 1
    bounds_generation(removed_domains_lst[:11], c_big_circ, g_big_circ, q_big_circ)

    # Small circles
    c_small_circ, g_small_circ, q_small_circ = 1, 1, 1
    bounds_generation(removed_domains_lst[11:], c_small_circ, g_small_circ, q_small_circ)

    equation = Equation()

    # Operator: du/dt - ∇(a(x)∇u) = 0

    heat_CG = {
        'du/dt**1':
            {
                'coeff': 1,
                'term': [2],
                'pow': 1
            },
        '-d2u/dx2**1':
            {
                'coeff': -1,
                'term': [0, 0],
                'pow': 1
            },
        '-d2u/dy2**1':
            {
                'coeff': -1,
                'term': [1, 1],
                'pow': 1
            }
    }

    equation.add(heat_CG)

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

    model.compile('autograd', lambda_operator=1, lambda_bound=10, removed_domains=removed_domains_lst)

    img_dir = os.path.join(os.path.dirname(__file__), 'heat_2d_complex_geometry_img')

    cb_cache = cache.Cache(cache_verbose=True, model_randomize_parameter=1e-6)

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=100,
                                         no_improvement_patience=1000,
                                         patience=5,
                                         randomize_parameter=1e-6,
                                         info_string_every=5)

    cb_plots = plot.Plots(save_every=5,
                          print_every=None,
                          img_dir=img_dir,
                          img_dim='2d_scatter')  # 3 image dimension options: 3d, 2d, 2d_scatter

    optimizer = Optimizer('Adam', {'lr': 1e-3})

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
