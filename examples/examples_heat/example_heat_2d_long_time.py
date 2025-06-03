import torch
import numpy as np
import os
import sys
import time

torch.manual_seed(42)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.utils import exact_solution_data

solver_device('cuda')

data_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../PINNacle_data/heat_longtime.npy"))


m1, m2, k = 4, 2, 1


def heat_2d_long_time_experiment(grid_res):
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

    boundaries = Conditions()

    # Initial condition: ###############################################################################################

    # u(x, y, 0)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': [y_min, y_max], 't': 0},
                         value=lambda grid: torch.sin(4 * np.pi * grid[:, 0]) * torch.sin(3 * np.pi * grid[:, 1]))

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

    def coeff_u(grid):
        x, y, t = grid[:, 0], grid[:, 1], grid[:, 2]
        return -5 * (1 + 2 * torch.sin(torch.pi * t / 4)) * \
               torch.sin(m1 * torch.pi * x) * torch.sin(m2 * torch.pi * y)

    # Operator: du/dt -  0.001 * ∆u + 5sin(k * u**2) * (1 + 2sin(pi * t / 4)) * sin(m1 * pi * x) * sin(m2 * pi * y)

    heat_LT = {
        'du/dt**1':
            {
                'coeff': 1,
                'term': [2],
                'pow': 1,
                'var': 0
            },
        '-0.001 * d2u/dx2**1':
            {
                'coeff': -0.001,
                'term': [0, 0],
                'pow': 1,
                'var': 0
            },
        '-0.001 * d2u/dy2**1':
            {
                'coeff': -0.001,
                'term': [1, 1],
                'pow': 1,
                'var': 0
            },
        'coeff_u * sin(k * u ** 2)':
            {
                'coeff': coeff_u,
                'term': [None],
                'pow': lambda u: torch.sin(k * u ** 2),
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
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, pde_dim_out)
    )

    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    start = time.time()

    grid = domain.build('autograd')
    grid_test = torch.cartesian_prod(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100))

    t_dim_flag = 't' in list(domain.variable_dict.keys())

    grid_params = domain.build('NN').to('cuda')
    exact = exact_solution_data(grid_params, data_file, pde_dim_in, pde_dim_out,
                                t_dim_flag=t_dim_flag).reshape(-1, 1)

    model_layers = [pde_dim_in, neurons, neurons, neurons, neurons, neurons, pde_dim_out]

    equation_params = [exact, grid_test, grid, domain, equation, boundaries, model_layers]

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

    model.train(optimizer, 5e5, save_model=True, callbacks=[cb_es, cb_plots, cb_cache])

    end = time.time()

    grid = domain.build('NN').to('cuda')
    net = net.to('cuda')

    exact = exact_solution_data(grid, data_file, pde_dim_in, pde_dim_out, t_dim_flag=t_dim_flag).reshape(-1, 1)
    net_predicted = net(grid)

    error_rmse = torch.sqrt(torch.mean((exact - net_predicted) ** 2))

    exp_dict_list.append({
        'grid_res': grid_res,
        'time': end - start,
        'RMSE': error_rmse.detach().cpu().numpy(),
        'type': 'heat_2d_long_time',
        'cache': True
    })

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))

    return exp_dict_list


nruns = 10

exp_dict_list = []

for grid_res in range(20, 201, 20):
    for _ in range(nruns):
        exp_dict_list.append(heat_2d_long_time_experiment(grid_res))

import pandas as pd

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)
df.to_csv('examples/benchmarking_data/heat_2d_long_time_experiment_20_200_cache={}.csv'.format(str(True)))
