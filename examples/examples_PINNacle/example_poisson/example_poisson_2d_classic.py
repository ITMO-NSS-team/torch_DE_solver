import time
import torch
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import adaptive_lambda, cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.utils import exact_solution_data

solver_device('cuda')
datapath = "poisson1_cg_data.dat"


def poisson_2d_classic_experiment(grid_res):
    exp_dict_list = []

    x_min, x_max = -0.5, 0.5
    y_min, y_max = -0.5, 0.5
    # grid_res = 30

    pde_dim_in = 2
    pde_dim_out = 1

    domain = Domain()
    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('y', [y_min, y_max], grid_res)

    boundaries = Conditions()

    # Circle type of removed domains ###################################################################################

    removed_domains_lst = [
        {'circle': {'center': (0.3, 0.3), 'radius': 0.1}},
        {'circle': {'center': (-0.3, 0.3), 'radius': 0.1}},
        {'circle': {'center': (0.3, -0.3), 'radius': 0.1}},
        {'circle': {'center': (-0.3, -0.3), 'radius': 0.1}}
    ]

    # CSG boundaries ###################################################################################################

    boundaries.dirichlet({'circle': {'center': (0.3, 0.3), 'radius': 0.1}}, value=0)
    boundaries.dirichlet({'circle': {'center': (-0.3, 0.3), 'radius': 0.1}}, value=0)
    boundaries.dirichlet({'circle': {'center': (0.3, -0.3), 'radius': 0.1}}, value=0)
    boundaries.dirichlet({'circle': {'center': (-0.3, -0.3), 'radius': 0.1}}, value=0)

    # Non CSG boundaries ###############################################################################################

    boundaries.dirichlet({'x': x_min, 'y': [y_min, y_max]}, value=1)
    boundaries.dirichlet({'x': x_max, 'y': [y_min, y_max]}, value=1)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': y_min}, value=1)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': y_max}, value=1)

    equation = Equation()

    # Operator: -d2u/dx2 - d2u/dy2 = 0

    poisson = {
        '-d2u/dx2':
            {
                'coeff': -1.,
                'term': [0, 0],
                'pow': 1,
                'var': 0
            },
        '-d2u/dy2':
            {
                'coeff': -1.,
                'term': [1, 1],
                'pow': 1,
                'var': 0
            }
    }

    equation.add(poisson)

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

    model.compile('autograd', lambda_operator=1, lambda_bound=100, removed_domains=removed_domains_lst)

    img_dir = os.path.join(os.path.dirname(__file__), 'poisson_2d_classic_img')

    cb_cache = cache.Cache(cache_verbose=True, model_randomize_parameter=1e-5)

    cb_es = early_stopping.EarlyStopping(eps=1e-9,
                                         loss_window=100,
                                         no_improvement_patience=1000,
                                         patience=5,
                                         info_string_every=10,
                                         randomize_parameter=1e-5)

    cb_plots = plot.Plots(save_every=500,
                          print_every=None,
                          img_dir=img_dir,
                          img_dim='2d_scatter')  # 3 image dimension options: 3d, 2d, 2d_scatter

    optimizer = Optimizer('Adam', {'lr': 1e-3})

    model.train(optimizer, 5e5, save_model=True, callbacks=[cb_cache, cb_es, cb_plots])

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
        'type': 'poisson_2d_classic',
        'cache': True
    })

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))

    return exp_dict_list


nruns = 10

exp_dict_list = []

for grid_res in range(30, 301, 30):
    for _ in range(nruns):
        exp_dict_list.append(poisson_2d_classic_experiment(grid_res))


import pandas as pd

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)
df.to_csv('examples/benchmarking_data/poisson_2d_classic_experiment_30_300_cache={}.csv'.format(str(True)))


