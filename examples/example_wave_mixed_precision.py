import torch
import numpy as np
import pandas as pd
import sys
import os
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import SALib
import scipy

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.solver import Solver
from tedeous.input_preprocessing import Equation
from tedeous.device import solver_device


devices = ['cpu', 'cuda']
mixed_precision = [True, False]

result = {
    'grid_res': [],
    'speedup': [],
    'RMSE': [],
    'device': [],
}


def experiment(device):
    solver_device(device)
    grid_res = 50

    # Grid
    x_grid = np.linspace(0, 1, grid_res + 1)
    t_grid = np.linspace(0, 1, grid_res + 1)

    x = torch.from_numpy(x_grid)
    t = torch.from_numpy(t_grid)

    grid = torch.cartesian_prod(x, t).float()

    # u(x,0)=1e4*sin^2(x(x-1)/10)

    func_bnd1 = lambda x: 10 ** 4 * np.sin((1 / 10) * x * (x - 1)) ** 2
    bnd1 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()
    bndval1 = func_bnd1(bnd1[:, 0])

    # du/dx (x,0) = 1e3*sin^2(x(x-1)/10)
    func_bnd2 = lambda x: 10 ** 3 * np.sin((1 / 10) * x * (x - 1)) ** 2
    bnd2 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()
    bop2 = {
        'du/dt':
            {
                'coeff': 1,
                'du/dt': [1],
                'pow': 1,
                'var': 0
            }
    }
    bndval2 = func_bnd2(bnd2[:, 0])

    # u(0,t) = u(1,t)
    bnd3_left = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), t).float()
    bnd3_right = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)), t).float()
    bnd3 = [bnd3_left, bnd3_right]

    # du/dt(0,t) = du/dt(1,t)
    bnd4_left = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), t).float()
    bnd4_right = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)), t).float()
    bnd4 = [bnd4_left, bnd4_right]

    bop4 = {
        'du/dx':
            {
                'coeff': 1,
                'du/dx': [0],
                'pow': 1,
                'var': 0
            }
    }
    bcond_type = 'periodic'

    bconds = [[bnd1, bndval1, 'dirichlet'],
              [bnd2, bop2, bndval2, 'operator'],
              [bnd3, bcond_type],
              [bnd4, bop4, bcond_type]]

    # wave equation is d2u/dt2-(1/4)*d2u/dx2=0
    C = 4
    wave_eq = {
        'd2u/dt2':
            {
                'coeff': 1,
                'd2u/dt2': [1, 1],
                'pow': 1,
                'var': 0
            },
        '-1/C*d2u/dx2':
            {
                'coeff': -1 / C,
                'd2u/dx2': [0, 0],
                'pow': 1,
                'var': 0
            }
    }

    # NN
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 1))

    model_1 = torch.nn.Sequential(
        torch.nn.Linear(2, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 1))

    equation = Equation(grid, wave_eq, bconds, h=0.01).set_strategy('NN')

    start = time.time()

    model = Solver(grid, equation, model, 'NN').solve(lambda_bound=1000, verbose=True, learning_rate=1e-2,
                                                      eps=1e-6, tmin=1000, tmax=1e5, use_cache=True,
                                                      cache_dir='../cache/', cache_verbose=True,
                                                      save_always=False, no_improvement_patience=500, print_every=500,
                                                      step_plot_print=False,
                                                      step_plot_save=True, mixed_precision=True)

    end = time.time()

    start_1 = time.time()

    model_1 = Solver(grid, equation, model_1, 'NN').solve(lambda_bound=1000, verbose=True, learning_rate=1e-2,
                                                          eps=1e-6, tmin=1000, tmax=1e5, use_cache=True,
                                                          cache_dir='../cache/', cache_verbose=True,
                                                          save_always=False, no_improvement_patience=500,
                                                          print_every=500, step_plot_print=False,
                                                          step_plot_save=True, mixed_precision=False)
    end_1 = time.time()

    model = model.cpu()
    model_1 = model_1.cpu()

    mp_true = model(grid).detach().cpu().numpy().flatten()
    mp_false = model_1(grid).detach().cpu().numpy().flatten()

    rmse = np.mean(np.square(mp_true - mp_false))

    result['grid_res'].append(grid_res)
    result['speedup'].append((end_1 - start_1) / (end - start))
    result['RMSE'].append(rmse)
    result['device'].append(device)
    print('Time taken = ', end - start)


for _ in range(10):
    experiment(devices[1])

df = pd.DataFrame(result)

df.to_csv('benchmarking_data/wave_exp_AMP_speedup.csv', index=False)