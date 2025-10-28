import torch
import numpy as np
import os
import sys
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

eps = 1
N = 10
k = torch.arange(N)


def exact_func(grid):
    x, y, t = grid[:, 0], grid[:, 1], grid[:, 2]
    sln = torch.sum((torch.sin(k * x[:, None]) + torch.sin(k * y[:, None])) * torch.exp(-k ** 2 * t[:, None]))
    return sln


def heat_2d_long_time_experiment(grid_res):
    exp_dict_list = []

    x_min, x_max = 0, 2 * torch.pi
    y_min, y_max = 0, 2 * torch.pi
    t_max = 0.01

    pde_dim_in = 3
    pde_dim_out = 1

    domain = Domain()

    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('y', [y_min, y_max], grid_res)
    domain.variable('t', [0, t_max], 10)

    boundaries = Conditions()

    # Initial condition: ###############################################################################################

    # u(x, y, 0)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': [y_min, y_max], 't': 0}, value=lambda grid: torch.sum(
        torch.sin(k * grid[:, 0][:, None]) + torch.sin(k * grid[:, 1][:, None])))

    # Boundary conditions (periodic): ##################################################################################

    # u(0, y, t) = u(2*pi, y, t)
    boundaries.periodic([{'x': x_min, 'y': [y_min, y_max], 't': [0, t_max]},
                        {'x': x_max, 'y': [y_min, y_max], 't': [0, t_max]}])

    # u(x, 0, t) = u(x, 2*pi, t)
    boundaries.periodic([{'x': [x_min, x_max], 'y': y_min, 't': [0, t_max]},
                        {'x': [x_min, x_max], 'y': y_max, 't': [0, t_max]}])

    equation = Equation()

    # Operator: du/dt -  epsilon * (u_xx + u_yy) = 0

    heat_LT = {
        'du/dt**1':
            {
                'coeff': 1,
                'term': [2],
                'pow': 1,
                'var': 0
            },
        '-epsilon * d2u/dx2**1':
            {
                'coeff': -eps,
                'term': [0, 0],
                'pow': 1,
                'var': 0
            },
        '-epsilon * d2u/dy2**1':
            {
                'coeff': -eps,
                'term': [1, 1],
                'pow': 1,
                'var': 0
            }
    }

    equation.add(heat_LT)

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

    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    start = time.time()

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=1, lambda_bound=100)

    img_dir = os.path.join(os.path.dirname(__file__), 'heat_2d_long_time_img')

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

    error_rmse = torch.sqrt(torch.mean((exact_func(grid).reshape(-1, 1) - net(grid)) ** 2))

    exp_dict_list.append({
        'grid_res': grid_res,
        'time': end - start,
        'RMSE': error_rmse.detach().cpu().numpy(),
        'type': 'heat_2d_linear_problem',
        'cache': True
    })

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))

    return exp_dict_list


nruns = 10

exp_dict_list = []

for grid_res in range(100, 1001, 100):
    for _ in range(nruns):
        exp_dict_list.append(heat_2d_long_time_experiment(grid_res))

import pandas as pd

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)
df.to_csv('examples/benchmarking_data/heat_2d_linear_problem_experiment_20_200_cache={}.csv'.format(str(True)))
