# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:33:44 2021

@author: user
"""
import torch
import numpy as np
import os
import sys
import scipy

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
import time

solver_device('cpu')


# Function u(x, y, t)
def u_func(grid):
    x, y, t = grid[:, 0], grid[:, 1], grid[:, 2]
    sln = torch.sin(np.pi * x) * torch.sin(np.pi * y) * torch.exp(-t)
    return sln


# Function a(x, y)
def a_diffusion_coeff(grid):
    x, y = grid[:, 0], grid[:, 1]
    return 2 + torch.sin(np.pi * x) * torch.sin(np.pi * y)


# Source function f(x, y, t)
def f_right_hand(grid):
    x, y, t = grid[:, 0], grid[:, 1], grid[:, 2]
    sin, cos, pi = torch.sin, torch.cos, np.pi

    term1 = (4 * pi ** 2 - 1) * sin(pi * x) * sin(pi * y)
    term2 = pi ** 2 * (2 * sin(pi * x) ** 2 * sin(pi * y) ** 2 -
                       cos(pi * x) ** 2 * sin(pi * y) ** 2 -
                       sin(pi * x) ** 2 * cos(pi * y) ** 2)

    return torch.exp(-t) * (term1 + term2)


x_min, x_max = -1, 1
y_min, y_max = -1, 1
T = 1
grid_res = 10
N_samples = 2500

domain = Domain()

domain.variable('x', [x_min, x_max], grid_res)
domain.variable('y', [y_min, y_max], grid_res)
domain.variable('t', [0, T], grid_res)

data = np.loadtxt(os.path.abspath(os.path.join(os.path.dirname(__file__), 'heatinv_points.dat')))

x_data = torch.tensor(data[:, 0]).reshape(-1)
y_data = torch.tensor(data[:, 1]).reshape(-1)
t_data = torch.tensor(data[:, 2]).reshape(-1)

boundaries = Conditions()

data_grid = torch.stack([x_data, y_data, t_data], dim=1)
u_bnd_val = u_func(data_grid).reshape(-1, 1) + torch.normal(0, 0.1, size=(2500, 1))

ind_bnd = np.random.choice(len(data_grid), N_samples, replace=False)

bnd_data = data_grid[ind_bnd]
u_bnd_val = u_bnd_val[ind_bnd]

boundaries.data(bnd=bnd_data, operator=None, value=u_bnd_val)

equation = Equation()

# Operator: du/dt - a * (d2u/dx2 + d2u/dy2) = f

heat_inverse = {
    'du/dt**1':
        {
            'coeff': 1,
            'du/dt': [2],
            'pow': 1,
            'var': 0
        },
    '-a * d2u/dx2**1':
        {
            'coeff': a_diffusion_coeff,
            'd2u/dx2': [0, 0],
            'pow': 1,
            'var': 0
        },
    '-a * d2u/dy2**1':
        {
            'coeff': a_diffusion_coeff,
            'd2u/dy2': [1, 1],
            'pow': 1,
            'var': 0
        },
    'f(x, y, t)':
        {
            'coeff': f_right_hand,
            'term': [None],
            'pow': 1
        }
}

equation.add(heat_inverse)

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
    torch.nn.Linear(neurons, neurons),
    torch.nn.Tanh(),
    torch.nn.Linear(neurons, 1)
)

for m in net.modules():
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)

model = Model(net, domain, equation, boundaries)

model.compile('autograd', lambda_operator=1, lambda_bound=100)

img_dir = os.path.join(os.path.dirname(__file__), 'heat_2d_inverse_problem_img')

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
                      img_dim='2d')

optimizer = Optimizer('Adam', {'lr': 1e-3})  # 3 image dimension options: 3d, 2d, 2d_scatter

callbacks = [cb_cache, cb_es, cb_plots]

start = time.time()

model.train(optimizer, 1e6, save_model=True, callbacks=callbacks)

# end = time.time()
#
# grid = domain.build('NN').to('cuda')
#
# error_rmse = torch.sqrt(torch.mean((func(grid).reshape(-1, 1) - net(grid)) ** 2))
#
# exp_dict_list.append(
#     {'grid_res': grid_res, 'time': end - start, 'RMSE': error_rmse.detach().numpy(), 'type': 'wave_eqn',
#      'cache': True})
#
# print('Time taken {}= {}'.format(grid_res, end - start))
# print('RMSE {}= {}'.format(grid_res, error_rmse))
#
# nruns = 10
#
# exp_dict_list = []
#
# CACHE = True
#
# for grid_res in range(10, 101, 10):
#     for _ in range(nruns):
#         exp_dict_list.append(heat_2d_experiment(grid_res, CACHE))
#
# import pandas as pd
#
# exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
# df = pd.DataFrame(exp_dict_list_flatten)
# df.boxplot(by='grid_res', column='time', fontsize=42, figsize=(20, 10))
# df.boxplot(by='grid_res', column='RMSE', fontsize=42, figsize=(20, 10), showfliers=False)
# df.to_csv('benchmarking_data/heat_experiment_10_100_cache={}.csv'.format(str(CACHE)))

