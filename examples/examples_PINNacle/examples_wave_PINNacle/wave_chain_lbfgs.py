import torch
import numpy as np
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
from tedeous.models import mat_model

solver_device('cpu')


def exact_func(grid, beta=5):
    x, t = grid[:, 0], grid[:, 1]
    sln = torch.sin(np.pi * x) * torch.cos(2 * np.pi * t) + 0.5 * \
          torch.sin(beta * np.pi * x) * torch.cos(2 * beta * np.pi * t)
    return sln


def wave_1d_basic_experiment(x_res, t_res, optimizer, beta=5):
    exp_dict_list = []

    x_min, x_max = 0, 1
    t_max = 1

    domain = Domain()
    domain.variable('x', [x_min, x_max], x_res)
    domain.variable('t', [0, t_max], t_res)

    x = domain.variable_dict['x']
    t = domain.variable_dict['t']

    boundaries = Conditions()

    # Initial conditions ###############################################################################################

    init_func = torch.sin(torch.pi * x) + (1 / 2) * torch.sin(beta * torch.pi * t)

    # u(x, 0) = f_init(x, 0)
    boundaries.dirichlet({'x': [x_min, x_max], 't': 0}, value=init_func)

    # u_t(x, 0) = 0
    bop = {
        'du/dt':
            {
                'coeff': 1,
                'term': [1],
                'pow': 1,
                'var': 0
            }
    }
    boundaries.operator({'x': [x_min, x_max], 't': 0}, operator=bop, value=0)

    # Boundary conditions ##############################################################################################

    # u(0, t) = f_bnd(x, t)
    boundaries.dirichlet({'x': x_min, 't': [0, t_max]}, value=0)

    # u(1, t) = f_bnd(x, t)
    boundaries.dirichlet({'x': x_max, 't': [0, t_max]}, value=0)

    equation = Equation()

    # Operator: d2u/dt2 - 4 * d2u/dx2 = 0

    wave_eq = {
        'd2u/dt2**1':
            {
                'coeff': 1,
                'd2u/dt2': [1, 1],
                'pow': 1
            },
        '-C*d2u/dx2**1':
            {
                'coeff': -4,
                'd2u/dx2': [0, 0],
                'pow': 1
            }
    }

    equation.add(wave_eq)

    neurons = 200

    net = torch.nn.Sequential(
        torch.nn.Linear(2, neurons),
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

    start = time.time()

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=1, lambda_bound=100)

    img_dir = os.path.join(os.path.dirname(__file__), 'wave_1d_basic_img')

    cb_cache = cache.Cache(cache_verbose=True, model_randomize_parameter=1e-6)

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=100,
                                         no_improvement_patience=1000,
                                         patience=5,
                                         randomize_parameter=1e-6,
                                         info_string_every=100)

    cb_plots = plot.Plots(save_every=500,
                          print_every=None,
                          img_dir=img_dir,
                          scatter_flag=False
                          )

    model.train(optimizer, 5e5, save_model=True, callbacks=[cb_es, cb_plots, cb_cache])

    end = time.time()

    grid = domain.build('NN').to('cuda')
    net = net.to('cuda')

    error_rmse = torch.sqrt(torch.mean((exact_func(grid).reshape(-1, 1) - net(grid)) ** 2))

    exp_dict_list.append({
        'x_res': x_res,
        't_res': t_res,
        'time': end - start,
        'RMSE': error_rmse.detach().cpu().numpy(),
        'type': 'wave_1d_basic',
        'cache': True
    })

    error_l2re = torch.sqrt(torch.sum(
        (exact_func(grid).reshape(-1, 1) - net(grid)) ** 2) / torch.sum(exact_func(grid).reshape(-1, 1) ** 2))

    exp_dict_list.append({
        'x_res': x_res,
        't_res': t_res,
        'time': end - start,
        'L2RE': error_l2re.detach().cpu().numpy(),
        'type': 'wave_1d_basic',
        'cache': True
    })

    print(f'x_res={x_res}, t_res={t_res}, time={end - start}')
    print(f'x_res={x_res}, t_res={t_res}, RMSE={error_rmse}')
    print(f'x_res={x_res}, t_res={t_res}, L2RE={error_l2re}')

    return exp_dict_list


x_res = 257
t_res = 101
beta = 5

optimizer = [
    {
        "name": "LBFGS",
        "params": {
            "lr": 1,
            "max_iter": 20,
            "max_eval": None,
            "tolerance_grad": 1e-05,
            "tolerance_change": 1e-07,
            "history_size": 50,
            "line_search_fn": "strong_wolfe"
        }, "epochs": 2050
    }
]

nruns = 1

exp_dict_list = []

for _ in range(nruns):
    exp_dict_list.append(wave_1d_basic_experiment(x_res, t_res, optimizer, beta=beta))

import pandas as pd

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)
df.to_csv(
    f'examples/benchmarking_data/wave_1d_basic_experiment_physical_{x_res}_{t_res}.csv'
)
