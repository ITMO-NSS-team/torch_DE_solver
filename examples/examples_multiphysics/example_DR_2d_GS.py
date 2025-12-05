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
    os.path.join(os.path.dirname(__file__), "../PINNacle_data/grayscott.npy")
)


def DR_2d_gray_scott_experiment(grid_res, n_run: int, file_to_save: h5py.File, adv: bool = False):
    b = 0.04
    d = 0.1
    epsilon_1 = 1e-5
    epsilon_2 = 5e-6

    x_min, x_max = -1, 1
    y_min, y_max = -1, 1
    t_max = 10

    pde_dim_in = 3
    pde_dim_out = 2

    domain = Domain()
    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('y', [y_min, y_max], grid_res)
    domain.variable('t', [0, t_max], grid_res)

    boundaries = Conditions()

    # Initial conditions ###############################################################################################

    def u_init_func(grid):
        with torch.no_grad():
            magn = 1e-2
            output = 1 - torch.exp(-80 * ((grid[:, 0] + 0.05) ** 2 + (grid[:, 1] + 0.02) ** 2)) + 5
            output = magn * torch.rand_like(output) * output + output
        return output

    def v_init_func(grid):
        with torch.no_grad():
            magn = 1e-2
            output = torch.exp(-80 * ((grid[:, 0] + 0.05) ** 2 + (grid[:, 1] + 0.02) ** 2)) + 5
            output = magn * torch.rand_like(output) * output + output
        return output

    # u(x, y, 0) = u_init_func(x, y, 0)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': [y_min, y_max], 't': 0}, value=u_init_func, var=0)

    # v(x, y, 0) = v_init_func(x, y, 0)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': [y_min, y_max], 't': 0}, value=v_init_func, var=1)

    equation = Equation()

    if adv:
        gridXX = domain.build('nn')[0, ...]
        gridYY = domain.build('nn')[1, ...]
        print(f'gridXX shape is {gridXX.shape}')
        vortex_intensity_mult = np.random.uniform(-10, 10)
        with torch.no_grad():
            uu = vortex_intensity_mult * torch.where(gridXX ** 2 + gridYY ** 2 < 1,
                                                     -gridYY * torch.sin((gridXX ** 2 + gridYY ** 2) * (torch.pi)),
                                                     0.)
            vv = vortex_intensity_mult * torch.where(gridXX ** 2 + gridYY ** 2 < 1,
                                                     gridXX * torch.sin((gridXX ** 2 + gridYY ** 2) * (torch.pi)), 0.)

            uu = uu.view(size=[uu.numel(), ])
            vv = vv.view(size=[vv.numel(), ])

        diffusion_reaction_u = {
            'du/dt**1':
                {
                    'coeff': 1,
                    'term': [2],
                    'pow': 1,
                    'var': 0
                },
            'w_x * dudx**1':
                {
                    'coeff': uu,
                    'term': [0],
                    'pow': 1,
                    'var': 0
                },
            'w_y * dudy**1':
                {
                    'coeff': vv,
                    'term': [1],
                    'pow': 1,
                    'var': 0
                },
            '-epsilon_1 * d2u/dx2**1':
                {
                    'coeff': -epsilon_1,
                    'term': [0, 0],
                    'pow': 1,
                    'var': 0
                },
            '-epsilon_1 * d2u/dy2**1':
                {
                    'coeff': -epsilon_1,
                    'term': [1, 1],
                    'pow': 1,
                    'var': 0
                },
            '-b':
                {
                    'coeff': -b,
                    'term': [None],
                    'pow': 0,
                    'var': 0
                },
            'b * u':
                {
                    'coeff': b,
                    'term': [None],
                    'pow': 1,
                    'var': 0
                },
            'u * v ** 2':
                {
                    'coeff': 1,
                    'term': [[None], [None]],
                    'pow': [1, 2],
                    'var': [0, 1]
                }
        }

        diffusion_reaction_v = {
            'dv/dt**1':
                {
                    'coeff': 1,
                    'term': [2],
                    'pow': 1,
                    'var': 1
                },
            'w_x * dvdx**1':
                {
                    'coeff': uu,
                    'term': [0],
                    'pow': 1,
                    'var': 1
                },
            'w_y * dvdy**1':
                {
                    'coeff': vv,
                    'term': [1],
                    'pow': 1,
                    'var': 1
                },
            '-epsilon_2 * d2v/dx2**1':
                {
                    'coeff': -epsilon_2,
                    'term': [0, 0],
                    'pow': 1,
                    'var': 1
                },
            '-epsilon_2 * d2v/dy2**1':
                {
                    'coeff': -epsilon_2,
                    'term': [1, 1],
                    'pow': 1,
                    'var': 1
                },
            'd * v':
                {
                    'coeff': d,
                    'term': [None],
                    'pow': 1,
                    'var': 1
                },
            '-u * v ** 2':
                {
                    'coeff': -1,
                    'term': [[None], [None]],
                    'pow': [1, 2],
                    'var': [0, 1]
                }
        }
    else:
        diffusion_reaction_u = {
            'du/dt**1':
                {
                    'coeff': 1,
                    'term': [2],
                    'pow': 1,
                    'var': 0
                },
            '-epsilon_1 * d2u/dx2**1':
                {
                    'coeff': -epsilon_1,
                    'term': [0, 0],
                    'pow': 1,
                    'var': 0
                },
            '-epsilon_1 * d2u/dy2**1':
                {
                    'coeff': -epsilon_1,
                    'term': [1, 1],
                    'pow': 1,
                    'var': 0
                },
            '-b':
                {
                    'coeff': -b,
                    'term': [None],
                    'pow': 0,
                    'var': 0
                },
            'b * u':
                {
                    'coeff': b,
                    'term': [None],
                    'pow': 1,
                    'var': 0
                },
            'u * v ** 2':
                {
                    'coeff': 1,
                    'term': [[None], [None]],
                    'pow': [1, 2],
                    'var': [0, 1]
                }
        }

        diffusion_reaction_v = {
            'dv/dt**1':
                {
                    'coeff': 1,
                    'term': [2],
                    'pow': 1,
                    'var': 1
                },
            '-epsilon_2 * d2v/dx2**1':
                {
                    'coeff': -epsilon_2,
                    'term': [0, 0],
                    'pow': 1,
                    'var': 1
                },
            '-epsilon_2 * d2v/dy2**1':
                {
                    'coeff': -epsilon_2,
                    'term': [1, 1],
                    'pow': 1,
                    'var': 1
                },
            'd * v':
                {
                    'coeff': d,
                    'term': [None],
                    'pow': 1,
                    'var': 1
                },
            '-u * v ** 2':
                {
                    'coeff': -1,
                    'term': [[None], [None]],
                    'pow': [1, 2],
                    'var': [0, 1]
                }
        }

    equation.add(diffusion_reaction_u)
    equation.add(diffusion_reaction_v)

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

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=1, lambda_bound=100)

    cb_cache = cache.Cache(cache_verbose=True, model_randomize_parameter=1e-6)

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=100,
                                         no_improvement_patience=1000,
                                         patience=5,
                                         randomize_parameter=1e-6,
                                         info_string_every=10)

    optimizer = Optimizer('Adam', {'lr': 1e-4})

    model.train(optimizer, 6e4, save_model=True, callbacks=[cb_es, cb_cache])

    grid = domain.build('NN').to('cuda')

    net = model.net
    net = net.to('cuda')

    predicted_u, predicted_v = net(grid)[:, 0], net(grid)[:, 1]
    exact_u, exact_v = exact_solution_data(grid, data_file, pde_dim_in, pde_dim_out)

    error_rmse_u = torch.sqrt(torch.mean((exact_u - predicted_u) ** 2))
    error_rmse_v = torch.sqrt(torch.mean((exact_v - predicted_v) ** 2))

    with h5py.File(file_to_save, 'a') as f:
        run_results_subgrp = f.create_group(str(n_run))

        info_subgrp = run_results_subgrp.create_group('info')
        info_subgrp['has_adv'] = adv
        info_subgrp['pinn_loss'] = error_rmse_u.cpu().detach().numpy() + error_rmse_v.cpu().detach().numpy()

        output_function_subgrp = run_results_subgrp.create_group('outputs')
        output_function_subgrp['U'] = predicted_u.cpu().detach().numpy()
        output_function_subgrp['V'] = predicted_v.cpu().detach().numpy()

        input_function_subgrp = run_results_subgrp.create_group('inputs')
        input_function_subgrp['b'] = torch.full_like(predicted_v, b).cpu().detach().numpy()
        input_function_subgrp['d'] = torch.full_like(predicted_v, b).cpu().detach().numpy()
        input_function_subgrp['epsilon_1'] = torch.full_like(predicted_v, epsilon_1).cpu().detach().numpy()
        input_function_subgrp['epsilon_2'] = torch.full_like(predicted_v, epsilon_2).cpu().detach().numpy()

        if adv:
            input_function_subgrp['velocity_uu'] = uu.cpu().detach().numpy()
            input_function_subgrp['velocity_vv'] = vv.cpu().detach().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fine_tune_data", default=0, type=int)
    parser.add_argument("--grid_size", default=31, type=int)
    parser.add_argument("--n_run", default=10, type=int)

    args = parser.parse_args()

    nruns = args.n_run
    grid_res = args.grid_size

    exp_dict_list_u, exp_dict_list_v = [], []

    now = datetime.now()
    filename = f'reaction_diff_{now.day}_{now.hour}_{now.minute}.hdf5'

    for n_run in range(nruns):
        DR_2d_gray_scott_experiment(grid_res, n_run=n_run, file_to_save=filename, adv=args.fine_tune_data)
