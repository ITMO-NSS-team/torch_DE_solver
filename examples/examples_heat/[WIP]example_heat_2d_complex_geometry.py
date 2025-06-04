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
from tedeous.utils import exact_solution_data

solver_device('gpu')

data_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../PINNacle_data/heat_complex.npy"))

def heat_2d_complex_geometry_experiment(grid_res):
    exp_dict_list = []

    x_min, x_max = -8, 8
    y_min, y_max = -12, 12
    t_max = 3

    pde_dim_in = 3
    pde_dim_out = 1

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
        for rd in domains:
            key = list(rd.keys())[0]
            if key == 'circle':
                x0, y0 = rd[key]['center']
                r = rd[key]['radius']

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

    model.compile('autograd', lambda_operator=1, lambda_bound=10, removed_domains=removed_domains_lst)

    img_dir = os.path.join(os.path.dirname(__file__), 'heat_2d_complex_geometry_img')

    cb_cache = cache.Cache(cache_verbose=True, model_randomize_parameter=1e-6)

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=100,
                                         no_improvement_patience=1000,
                                         patience=5,
                                         randomize_parameter=1e-6,
                                         info_string_every=1)

    cb_plots = plot.Plots(save_every=100,
                          print_every=None,
                          img_dir=img_dir,
                          img_dim='2d',
                          scatter_flag=True,
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

    exact = exact_solution_data(grid, data_file, pde_dim_in, pde_dim_out, t_dim_flag=True).reshape(-1, 1)
    net_predicted = net(grid)

    error_rmse = torch.sqrt(torch.mean((exact - net_predicted) ** 2))

    exp_dict_list.append({
        'grid_res': grid_res,
        'time': end - start,
        'RMSE': error_rmse.detach().cpu().numpy(),
        'type': 'heat_2d_complex_geometry',
        'cache': True
    })

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))

    return exp_dict_list


nruns = 10

exp_dict_list = []

for grid_res in range(120, 1201, 120):
    for _ in range(nruns):
        exp_dict_list.append(heat_2d_complex_geometry_experiment(grid_res))

import pandas as pd

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)
df.to_csv('examples/benchmarking_data/heat_2d_complex_geometry_experiment_120_1200_cache={}.csv'.format(str(True)))
