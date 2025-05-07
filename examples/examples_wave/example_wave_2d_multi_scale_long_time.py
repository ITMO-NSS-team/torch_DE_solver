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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../examples_wave')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model

from tedeous.callbacks import early_stopping, plot, cache
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device

solver_device('gpu')

a = 2 ** 0.5
m1, m2 = 1, 3
n1, n2 = 1, 2
p1, p2 = 1, 1
c1, c2 = 1, 1


def exact_func(grid):
    x, y, t = grid[:, 0], grid[:, 1], grid[:, 2]
    sln = c1 * torch.sin(m1 * torch.pi * x) * torch.sinh(n1 * torch.pi * y) * torch.cos(p1 * torch.pi * t) + \
          c2 * torch.sinh(m2 * torch.pi * x) * torch.sin(n2 * torch.pi * y) * torch.cos(p2 * torch.pi * t)
    return sln


def wave2d_multi_scale_long_time_experiment(grid_res):
    exp_dict_list = []

    x_min, x_max = 0, 1
    y_min, y_max = 0, 1
    t_max = 100

    pde_dim_in = 3
    pde_dim_out = 1

    domain = Domain()
    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('y', [y_min, y_max], grid_res)
    domain.variable('t', [0, t_max], grid_res)

    x, y, t = [domain.variable_dict[axis] for axis in list(domain.variable_dict.keys())]

    boundaries = Conditions()

    # Initial conditions ###############################################################################################

    # u_t(x, 0) = 0
    bop = {
        'du/dt':
            {
                'coeff': 1,
                'du/dx': [2],
                'pow': 1,
                'var': 0
            }
    }
    boundaries.operator({'x': [x_min, x_max], 'y': [y_min, y_max], 't': 0}, operator=bop, value=0)

    # Boundary conditions ##############################################################################################

    # u(x_min, y, t) = 0
    boundaries.dirichlet({'x': x_min, 'y': [y_min, y_max], 't': [0, t_max]}, value=0)

    # u(x_max, y, t) = 0
    bnd_right_func = c2 * torch.sin(m2 * torch.pi * x) * torch.sinh(n2 * torch.pi * y) * torch.cos(p2 * torch.pi * t)
    boundaries.dirichlet({'x': x_max, 'y': [y_min, y_max], 't': [0, t_max]}, value=bnd_right_func)

    # u(x, y_min, t) = 0
    boundaries.dirichlet({'x': [x_min, x_max], 'y': y_min, 't': [0, t_max]},
                         value=0)

    # u(x, y_max, t) = 0
    bnd_top_func = c1 * torch.sin(m1 * torch.pi * x) * torch.sinh(n1 * torch.pi * y) * torch.cos(p1 * torch.pi * t)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': y_max, 't': [0, t_max]},
                         value=bnd_top_func)

    equation = Equation()

    # Operator: d2u/dt2 − (d2u/dx2 + a ** 2 * d2u/dy2) = 0

    wave_eq = {
        'd2u/dt2**1':
            {
                'coeff': 1,
                'd2u/dt2': [2, 2],
                'pow': 1
            },
        'd2u/dx2**1':
            {
                'coeff': 1,
                'd2u/dx2': [0, 0],
                'pow': 1
            },
        'a ** 2 * d2u/dy2**1':
            {
                'coeff': a,
                'd2u/dx2': [1, 1],
                'pow': 1
            }
    }

    equation.add(wave_eq)

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

    img_dir = os.path.join(os.path.dirname(__file__), 'wave_2d_multi_scale_long_time_img')

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

    optimizer = Optimizer('Adam', {'lr': 1e-4})

    model.train(optimizer, 5e5, save_model=True, callbacks=[cb_es, cb_plots, cb_cache])

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
        'type': 'wave2d_multi_scale_long_time',
        'cache': True
    })

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))

    return exp_dict_list


nruns = 10

exp_dict_list = []

for grid_res in range(20, 201, 20):
    for _ in range(nruns):
        exp_dict_list.append(wave2d_multi_scale_long_time_experiment(grid_res))

import pandas as pd

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)
df.to_csv('examples/benchmarking_data/wave2d_multi_scale_long_time_experiment_physical_50_500_cache={}.csv'
          .format(str(True)))
