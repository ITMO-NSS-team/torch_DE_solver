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
from tedeous.utils import exact_solution_data


solver_device('gpu')

datapath = "Kuramoto_Sivashinsky.dat"

alpha = 100 / 16
beta = 100 / 16**2
gamma = 100 / 16**4


def kuramoto_sivashinsky_experiment(grid_res):
    exp_dict_list = []

    x_min, x_max = 0, 2 * np.pi
    t_max = 1
    # grid_res = 20

    pde_dim_in = 2
    pde_dim_out = 1

    domain = Domain()
    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('t', [0, t_max], grid_res)

    boundaries = Conditions()

    # Initial conditions ###############################################################################################

    init_func = lambda grid: torch.cos(grid[:, 0]) * (1 + torch.sin(grid[:, 0]))

    # u(x, 0) = cos(x) * (1 + sin(x))
    boundaries.dirichlet({'x': [x_min, x_max], 't': 0}, value=init_func)

    equation = Equation()

    # Operator: u_t + alpha * u * u_x + beta * u_xx + gamma * u_xxx = 0

    KS_equation = {
        'du/dt**1':
            {
                'coeff': 1,
                'term': [1],
                'pow': 1
            },
        'alpha * u * du/dx**1':
            {
                'coeff': alpha,
                'term': [[None], [0]],
                'pow': [1, 1]
            },
        'beta * d2u/dx2**1':
            {
                'coeff': beta,
                'term': [0, 0],
                'pow': 1
            },
        'gamma * d4u/dx4**1':
            {
                'coeff': gamma,
                'term': [0, 0, 0, 0],
                'pow': 1
            },
    }

    equation.add(KS_equation)

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

    img_dir = os.path.join(os.path.dirname(__file__), 'kuramoto_sivashinsky_img')

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

    exact = exact_solution_data(grid, datapath, pde_dim_in, pde_dim_out).reshape(-1, 1)
    net_predicted = net(grid)

    error_rmse = torch.sqrt(torch.mean((exact - net_predicted) ** 2))

    exp_dict_list.append({
        'grid_res': grid_res,
        'time': end - start,
        'RMSE': error_rmse.detach().cpu().numpy(),
        'type': 'kuramoto_sivashinsky',
        'cache': True
    })

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))

    return exp_dict_list


nruns = 10

exp_dict_list = []

for grid_res in range(20, 201, 20):
    for _ in range(nruns):
        exp_dict_list.append(kuramoto_sivashinsky_experiment(grid_res))

import pandas as pd

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)
df.to_csv('examples/benchmarking_data/kuramoto_sivashinsky_experiment_physical_20_200_cache={}.csv'.format(str(True)))

