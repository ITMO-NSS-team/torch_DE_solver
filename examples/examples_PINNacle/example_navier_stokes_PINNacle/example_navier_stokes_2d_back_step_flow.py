import torch
import os
import sys
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.utils import exact_solution_data

solver_device('gpu')

datapath = "../../PINNacle_data/ns_0_obstacle.npy"

ro = 1
mu = 0.01


def navier_stokes_2d_back_step_flow_experiment(grid_res):
    exp_dict_list_u, exp_dict_list_v, exp_dict_list_p = [], [], []

    x_min, x_max = 0, 4
    y_min, y_max = 0, 2
    # grid_res = 20

    pde_dim_in = 2
    pde_dim_out = 3

    # Rectangle type of removed domains
    removed_domains_lst = [
        {'rectangle': {'coords_min': [0, 1], 'coords_max': [2, 2]}}
    ]

    domain = Domain()
    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('y', [y_min, y_max], grid_res)

    y = domain.variable_dict['y']
    first_one_index = int(torch.where(y == 1.0)[0][0])
    y = y[:first_one_index + 1]

    boundaries = Conditions()

    # Inlet boundary condition u_in = 4y(1 - y) ########################################################################

    # u(x_min, y) = f(y)
    boundaries.dirichlet({'x': x_min, 'y': [y_min, 1]}, value=4 * y * (1 - y), var=0)

    # No-slip boundary condition u = 0 #################################################################################

    # u(x_max, y) = 0
    boundaries.dirichlet({'x': x_max, 'y': [y_min, y_max]}, value=0, var=0)
    # u(x, y_min) = 0
    boundaries.dirichlet({'x': [x_min, x_max], 'y': y_min}, value=0, var=0)
    # u(x, y_max) = 0
    boundaries.dirichlet({'x': [2, x_max], 'y': y_max}, value=0, var=0)
    # u(x_rec, y_rec_min) = 0
    boundaries.dirichlet({'x': [x_min, 2], 'y': 1}, value=0, var=0)
    # u(x_rec_max, y_rec) = 0
    boundaries.dirichlet({'x': 2, 'y': [1, y_max]}, value=0, var=0)

    # Outlet pressure condition (p = 0 at outlet) ######################################################################

    # p(x_max, y) = 0
    boundaries.dirichlet({'x': x_max, 'y': [y_min, y_max]}, value=0, var=2)

    equation = Equation()

    # Operator 1: # operator: u_x + v_y = 0

    NS_1 = {
        'du/dx':
            {
                'coeff': 1,
                'term': [0],
                'pow': 1,
                'var': 0
            },
        'dv/dy':
            {
                'coeff': 1,
                'term': [1],
                'pow': 1,
                'var': 1
            }
    }

    # Operator 2: u * u_x + v * u_y + 1 / ro * p_x - mu * (u_xx + u_yy) = 0

    NS_2 = {
        'u * du/dx':
            {
                'coeff': 1,
                'term': [[None], [0]],
                'pow': [1, 1],
                'var': [0, 0]
            },
        'v * du/dy':
            {
                'coeff': 1,
                'term': [[None], [1]],
                'pow': [1, 1],
                'var': [1, 0]
            },
        '1/ro * dp/dx':
            {
                'coeff': 1 / ro,
                'term': [0],
                'pow': 1,
                'var': 2
            },
        '-mu * d2u/dx2':
            {
                'coeff': -mu,
                'term': [0, 0],
                'pow': 1,
                'var': 0
            },
        '-mu * d2u/dy2':
            {
                'coeff': -mu,
                'term': [1, 1],
                'pow': 1,
                'var': 0
            }
    }

    # Operator 3: u * v_x + v * v_y + 1 / ro * p_y - mu * (v_xx + v_yy) = 0

    NS_3 = {
        'u * dv/dx':
            {
                'coeff': 1,
                'term': [[None], [0]],
                'pow': [1, 1],
                'var': [0, 1]
            },
        'v * dv/dy':
            {
                'coeff': 1,
                'term': [[None], [1]],
                'pow': [1, 1],
                'var': [1, 1]
            },
        '1/ro * dp/dy':
            {
                'coeff': 1 / ro,
                'term': [1],
                'pow': 1,
                'var': 2
            },
        '-mu * d2v/dx2':
            {
                'coeff': -mu,
                'term': [0, 0],
                'pow': 1,
                'var': 1
            },
        '-mu * d2v/dy2':
            {
                'coeff': -mu,
                'term': [1, 1],
                'pow': 1,
                'var': 1
            }
    }

    equation.add(NS_1)
    equation.add(NS_2)
    equation.add(NS_3)

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

    model.compile('autograd', lambda_operator=1, lambda_bound=100, removed_domains=removed_domains_lst)

    cb_es = early_stopping.EarlyStopping(eps=1e-5,
                                         loss_window=100,
                                         no_improvement_patience=1000,
                                         patience=10,
                                         info_string_every=10,
                                         randomize_parameter=1e-5)

    img_dir = os.path.join(os.path.dirname(__file__), 'navier_stokes_2d_back_step_flow_img')

    cb_plots = plot.Plots(save_every=500,
                          print_every=None,
                          img_dir=img_dir,
                          img_dim='2d',
                          scatter_flag=True)

    optimizer = Optimizer('Adam', {'lr': 1e-4})

    model.train(optimizer, 5e5, save_model=True, callbacks=[cb_es, cb_plots])

    end = time.time()

    grid = domain.build('NN').to('cuda')
    net = net.to('cuda')

    predicted_u, predicted_v, predicted_p = net(grid)[:, 0], net(grid)[:, 1], net(grid)[:, 2]
    exact_u, exact_v, exact_p = exact_solution_data(grid, datapath, pde_dim_in, pde_dim_out)

    error_rmse_u = torch.sqrt(torch.mean((exact_u - predicted_u) ** 2))
    error_rmse_v = torch.sqrt(torch.mean((exact_v - predicted_v) ** 2))
    error_rmse_p = torch.sqrt(torch.mean((exact_p - predicted_p) ** 2))

    exp_dict_list_u.append({
        'grid_res': grid_res,
        'time': end - start,
        'RMSE_u_func': error_rmse_u.detach().cpu().numpy(),
        'type': 'navier_stokes_2d_back_step_flow',
        'cache': True
    })
    exp_dict_list_v.append({
        'grid_res': grid_res,
        'time': end - start,
        'RMSE_v_func': error_rmse_v.detach().cpu().numpy(),
        'type': 'navier_stokes_2d_back_step_flow',
        'cache': True
    })
    exp_dict_list_p.append({
        'grid_res': grid_res,
        'time': end - start,
        'RMSE_p_func': error_rmse_p.detach().cpu().numpy(),
        'type': 'navier_stokes_2d_back_step_flow',
        'cache': True
    })

    print('Time taken {}= {}'.format(grid_res, end - start))

    print('RMSE_u_func {}= {}'.format(grid_res, error_rmse_u))
    print('RMSE_v_func {}= {}'.format(grid_res, error_rmse_v))
    print('RMSE_p_func {}= {}'.format(grid_res, error_rmse_p))

    return exp_dict_list_u, exp_dict_list_v, exp_dict_list_p


nruns = 10

exp_dict_list_u, exp_dict_list_v, exp_dict_list_p = [], [], []

for grid_res in range(20, 201, 20):
    for _ in range(nruns):
        list_u, list_v, list_p = navier_stokes_2d_back_step_flow_experiment(grid_res)
        exp_dict_list_u.append(list_u)
        exp_dict_list_v.append(list_v)
        exp_dict_list_p.append(list_p)

import pandas as pd

exp_dict_list_u_flatten = [item for sublist in exp_dict_list_u for item in sublist]
exp_dict_list_v_flatten = [item for sublist in exp_dict_list_v for item in sublist]
exp_dict_list_p_flatten = [item for sublist in exp_dict_list_p for item in sublist]

df_u = pd.DataFrame(exp_dict_list_u_flatten)
df_v = pd.DataFrame(exp_dict_list_v_flatten)
df_p = pd.DataFrame(exp_dict_list_p_flatten)

df_u.to_csv('examples/benchmarking_data/navier_stokes_2d_back_step_flow_experiment_20_200_cache_u_func={}.csv'
            .format(str(True)))
df_v.to_csv('examples/benchmarking_data/navier_stokes_2d_back_step_flow_experiment_20_200_cache_v_func={}.csv'
            .format(str(True)))
df_p.to_csv('examples/benchmarking_data/navier_stokes_2d_back_step_flow_experiment_20_200_cache_p_func={}.csv'
            .format(str(True)))
