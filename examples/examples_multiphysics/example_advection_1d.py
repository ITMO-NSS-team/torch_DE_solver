import torch
import os
import sys
import argparse

import numpy as np

import h5py
from datetime import datetime

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model

from tedeous.callbacks import early_stopping, cache
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.utils import exact_solution_data

solver_device('cuda')

data_file = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..examples/PINNacle_data/grayscott.npy")
)


def exact_solution(grid, c=1):
    x, t = grid[:, 0], grid[:, 1]
    sln = torch.sin(torch.pi * (x - c * t))
    return sln


def advection_1d_experiment(grid_res, n_run: int, file_to_save: h5py.File, fune_tune_flag: bool = False):
    x_min, x_max = 0, 1
    t_max = 1

    pde_dim_in = 2
    pde_dim_out = 1

    domain = Domain()
    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('t', [0, t_max], grid_res)

    boundaries = Conditions()

    # Initial conditions ###############################################################################################

    init_func = lambda grid: torch.sin(torch.pi * grid[:, 0])

    # u(x, 0) = sin(pi * x)
    boundaries.dirichlet({'x': [x_min, x_max], 't': 0}, value=init_func, var=0)

    equation = Equation()

    v_field = lambda grid: 1 + 0.5 * torch.sin(2 * np.pi * grid[:, 0])

    advection = {
        'du/dt**1':
            {
                'coeff': 1,
                'term': [1],
                'pow': 1,
                'var': 0
            },
        'du/dx**1':
            {
                'coeff': v_field,
                'term': [0],
                'pow': 1,
                'var': 0
            },
    }

    equation.add(advection)

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

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=1, lambda_bound=100)

    cb_cache = cache.Cache(cache_verbose=False, model_randomize_parameter=1e-6)

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=100,
                                         no_improvement_patience=1000,
                                         patience=5,
                                         randomize_parameter=1e-6,
                                         info_string_every=10)

    optimizer = Optimizer('Adam', {'lr': 1e-4})

    model.train(optimizer, 6e4, save_model=False, callbacks=[cb_es, cb_cache])

    grid = domain.build('NN').to('cuda')

    net = model.net
    net = net.to('cuda')

    predicted_u = net(grid)[:, 0]
    exact_u = exact_solution(grid).reshape(-1, 1)

    error_rmse_u = torch.sqrt(torch.mean((exact_u - predicted_u) ** 2))

    with h5py.File(file_to_save, 'a') as f:
        run_results_subgrp = f.create_group(str(n_run))

        info_subgrp = run_results_subgrp.create_group('info')
        info_subgrp['has_fune_tune_flag'] = fune_tune_flag
        info_subgrp['pinn_loss'] = error_rmse_u.cpu().detach().numpy()

        output_function_subgrp = run_results_subgrp.create_group('outputs')
        output_function_subgrp['U'] = predicted_u.cpu().detach().numpy()

        input_function_subgrp = run_results_subgrp.create_group('inputs')
        input_function_subgrp['initial condition'] = init_func(grid).cpu().detach().numpy()
        input_function_subgrp['v_field'] = v_field(grid).cpu().detach().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fine_tune_data", default=0, type=int)
    parser.add_argument("--grid_size", default=51, type=int)
    parser.add_argument("--n_run", default=10, type=int)

    args = parser.parse_args()

    nruns = args.n_run
    grid_res = args.grid_size

    exp_dict_list_u, exp_dict_list_v = [], []

    now = datetime.now()
    filename = f'advection_1d_{now.day}_{now.hour}_{now.minute}.hdf5'

    for n_run in range(nruns):
        advection_1d_experiment(grid_res, n_run=n_run, file_to_save=filename, fune_tune_flag=args.fine_tune_data)
