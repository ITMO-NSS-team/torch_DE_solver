import functools
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd

# import tensorflow as tf

# tf.config.set_visible_devices([], 'GPU')

from scipy.optimize import minimize
# from scipy.optimize import differential_evolution
# from scipy.optimize import dual_annealing
import numba
import time
import matplotlib.pyplot as plt

import torch
from scipy.ndimage import gaussian_filter
from PDE_solver import string_reshape, plot_3D_surface, lbfgs_solution, solution_interp_nn, solution_interp_RBF, solution_interp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams["figure.max_open_warning"] = 1000

norm_bond = float(20)


# @ray.remote
def init_field_expetiment(nrun=1, grid_scenario=[[10, 10]], interp_type='random', diff_scheme=[1, 2]):
    np.random.seed()
    experiment = []

    x = np.linspace(0, 1, 10)
    t = np.linspace(0, 1, 10)

    old_grid = numba.typed.List()

    old_grid.append(x)
    old_grid.append(t)

    arr = np.random.random((10, 10))

    params = gaussian_filter(arr, 1)

    params = torch.from_numpy(params).to(device)

    for grid_res in grid_scenario:

        x = np.linspace(0, 1, grid_res[0])
        t = np.linspace(0, 1, grid_res[1])

        new_grid = numba.typed.List()

        new_grid.append(x)
        new_grid.append(t)

        grid = np.meshgrid(*new_grid)
        grid = torch.tensor(grid, device=device)

        bcond = [{'boundary': 0, 'axis': 0, 'string': torch.zeros(len(new_grid[0]), device=device)},
                 {'boundary': len(x) - 1, 'axis': 0, 'string': torch.zeros(len(new_grid[0]), device=device)},
                 {'boundary': 0, 'axis': 1, 'string': torch.from_numpy(np.sin(np.pi * new_grid[1])).to(device)},
                 {'boundary': len(t) - 1, 'axis': 1,
                  'string': torch.from_numpy(np.sin(np.pi * new_grid[1])).to(device)}]

        if interp_type == 'scikit_interpn':
            arr = params.cpu().detach().numpy()
            arr_interpn = solution_interp(old_grid, arr, new_grid)
            params = torch.from_numpy(arr_interpn).to(device)
            params.requires_grad_() 
        elif interp_type == 'interp_RBF':
            arr = solution_interp_RBF(old_grid, arr, new_grid, method='linear', smooth=10)
        elif interp_type == 'interp_nn':
            arr = solution_interp_nn(old_grid, arr, new_grid)
        else:
            np.random.seed(grid_res)
            arr = np.random.random((len(x), len(t)))

        exact_sln = f'exact_sln/wave_{grid_res[0]}.csv'

        wolfram_interp = np.genfromtxt(exact_sln, delimiter=',')

        # params = torch.zeros(grid[0].shape, device=device)
        # params.requires_grad_() 

        operator = [[(1, 0, 2, 1)], [(-1 / 4, 1, 2, 1)]]

        start_time = time.time()

        params = lbfgs_solution(params, grid, operator, norm_bond, bcond)

        elapsed_time = time.time() - start_time

        print(f'[{datetime.datetime.now()}] grid_x = {grid_res[0]} grid_t={grid_res[1]} time = {elapsed_time}')

        full_sln_interp = string_reshape(params, new_grid)
        # print(full_sln_interp)
        error = np.abs(full_sln_interp.detach().cpu().numpy() - wolfram_interp)
        max_error = np.max(error)
        wolfram_MAE = np.mean(error)
        print(max_error)
        print(wolfram_MAE)
        params = full_sln_interp

        plot_3D_surface(full_sln_interp.detach().cpu().numpy(), wolfram_interp, new_grid)

        experiment.append(
            {'grid_x': len(x), 'grid_t': len(t), 'time': elapsed_time, 'MAE': wolfram_MAE, 'max_err': max_error,
             'interp_type': interp_type, 'scheme_order': diff_scheme[0], 'boundary_order': diff_scheme[1]})

        old_grid = new_grid

    return experiment


if os.path.isfile("interp_v3.csv"):
    df = pd.read_csv("interp_v3.csv", index_col=0)
else:
    df = pd.DataFrame(
        columns=['grid_x', 'grid_t', 'time', 'MAE', 'max_err', 'interp_type', 'scheme_order', 'boundary_order'])

# ray.init(ignore_reinit_error=True)

# warm up
# results = ray.get([init_field_expetiment.remote(run,interp_type='random') for run in range(1)])
results = init_field_expetiment(grid_scenario=[[20, 20]], interp_type='scikit_interpn')

# for ds in [[1,1],[2,2],[1,2]]:
#     for interp_type in ['random','scikit_interpn','interp_RBF','interp_nn']:
#         results = ray.get([init_field_expetiment.remote(run,interp_type=interp_type,grid_scenario=[[10,10],[15,15],[20,20],[25,25],[30,30],[35,35],[40,40],[45,45],[50,50]],diff_scheme=ds) for run in range(20)])
#         for result in results:
#             df=df.append(result,ignore_index=True)
#         df.to_csv(f'interp_v3_{datetime.datetime.now().timestamp()}.csv',index=False