import torch
import os
import sys
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../examples_diffusion')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model

from tedeous.callbacks import early_stopping, plot, cache
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device

solver_device('gpu')

epsilon = 0.01
a = 0.1
N = 6
k = torch.arange(N)


def exact_func(grid):
    x, t = grid[:, 0], grid[:, 1]
    sln = torch.sum(torch.sin(k * x[:, None] - k * a * t[:, None]) *
                    torch.exp(-epsilon * k ** 2 * t[:, None]))
    return sln


def convection_diffusion_1d_experiment(grid_res):
    exp_dict_list = []

    x_min, x_max = 0, 2 * torch.pi
    t_max = 0.1

    pde_dim_in = 2
    pde_dim_out = 1

    domain = Domain()
    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('t', [0, t_max], 10)

    boundaries = Conditions()

    # Initial condition: ###############################################################################################

    # u(x, 0)
    boundaries.dirichlet({'x': [x_min, x_max], 't': 0},
                         value=lambda grid: torch.sum(torch.sin(k * grid[:, 0][:, None])))

    # Boundary conditions (periodic): ##################################################################################

    # u(0, t) = u(2*pi, t)
    boundaries.periodic([{'x': x_min, 't': [0, t_max]},
                         {'x': x_max, 't': [0, t_max]}])

    equation = Equation()

    # Operator 1:  ut + a * u_x - ε1 * u_xx = 0

    diffusion_1d = {
        'du/dt**1':
            {
                'coeff': 1,
                'term': [1],
                'pow': 1,
                'var': 0
            },
        'a * du/dx**1':
            {
                'coeff': a,
                'term': [0],
                'pow': 1,
                'var': 0
            },
        '-epsilon * d2u/dx2**1':
            {
                'coeff': -epsilon,
                'term': [0, 0],
                'pow': 1,
                'var': 0
            }
    }

    equation.add(diffusion_1d)

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

    img_dir = os.path.join(os.path.dirname(__file__), 'convection_diffusion_1d_img')

    cb_cache = cache.Cache(cache_verbose=True, model_randomize_parameter=1e-6)

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=100,
                                         no_improvement_patience=1000,
                                         patience=5,
                                         randomize_parameter=1e-6,
                                         info_string_every=10)

    cb_plots = plot.Plots(save_every=None,
                          print_every=None,
                          img_dir=img_dir,
                          img_dim='2d',
                          scatter_flag=False,
                          plot_axes=[0, 1],
                          fixed_axes=[2],
                          n_samples=4,
                          var_transpose=False)

    optimizer = Optimizer('Adam', {'lr': 1e-4})

    model.train(optimizer, 5e5, save_model=True, callbacks=[cb_es, cb_plots, cb_cache])

    end = time.time()

    grid = domain.build('NN').to('cuda')
    net = net.to('cuda')

    error_rmse_u = torch.sqrt(torch.mean((exact_func(grid).reshape(-1, 1) - net(grid)) ** 2))

    exp_dict_list.append({
        'grid_res': grid_res,
        'time': end - start,
        'RMSE': error_rmse_u.detach().cpu().numpy(),
        'type': 'convection_diffusion_1d',
        'cache': True
    })

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse_u))

    return exp_dict_list


nruns = 10

exp_dict_list = []

for grid_res in range(100, 1001, 100):
    for _ in range(nruns):
        exp_dict_list.append(convection_diffusion_1d_experiment(grid_res))

import pandas as pd

exp_dict_list = [item for sublist in exp_dict_list for item in sublist]

df_u = pd.DataFrame(exp_dict_list)

df_u.to_csv('examples/benchmarking_data/convection_diffusion_1d_experiment_100_1000_cache_u_func={}.csv'.format(str(True)))
