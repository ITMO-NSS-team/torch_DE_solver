# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:33:44 2021

@author: user
"""
import torch
import numpy as np
import os
import sys
import time
from scipy import interpolate

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model

from tedeous.callbacks import early_stopping, plot, cache
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device

solver_device('gpu')

wave_datapath = "wave_darcy.dat"
darcy_2d_coef_data = np.loadtxt("darcy_2d_coef_256.dat")


def func_data(datapath):

    t_transpose = True

    input_dim = 3
    output_dim = 2

    def trans_time_data_to_dataset(datapath, ref_data):
        data = ref_data
        slice = (data.shape[1] - input_dim + 1) // output_dim
        assert slice * output_dim == data.shape[
            1] - input_dim + 1, "Data shape is not multiple of pde.output_dim"

        with open(datapath, "r", encoding='utf-8') as f:
            def extract_time(string):
                index = string.find("t=")
                if index == -1:
                    return None
                return float(string[index + 2:].split(' ')[0])

            t = None
            for line in f.readlines():
                if line.startswith('%') and line.count('@') == slice * output_dim:
                    t = line.split('@')[1:]
                    t = list(map(extract_time, t))
            if t is None or None in t:
                raise ValueError("Reference Data not in Comsol format or does not contain time info")
            t = np.array(t[::output_dim])

        t, x0 = np.meshgrid(t, data[:, 0])
        list_x = [x0.reshape(-1)]
        for i in range(1, input_dim - 1):
            list_x.append(np.stack([data[:, i] for _ in range(slice)]).T.reshape(-1))
        list_x.append(t.reshape(-1))
        for i in range(output_dim):
            list_x.append(data[:, input_dim - 1 + i::output_dim].reshape(-1))
        return np.stack(list_x).T

    scale = 1

    def transform_fn(data):
        data[:, :input_dim] *= scale
        return data

    def load_ref_data(datapath, transform_fn=None, t_transpose=False):
        ref_data = np.loadtxt(datapath, comments="%", encoding='utf-8').astype(np.float32)
        if t_transpose:
            ref_data = trans_time_data_to_dataset(datapath, ref_data)
        if transform_fn is not None:
            ref_data = transform_fn(ref_data)
            return ref_data

    load_ref_data(datapath)
    load_ref_data(datapath, transform_fn, t_transpose)

    return load_ref_data


def func(grid):
    wave_data = func_data(wave_datapath)
    sln = torch.Tensor(
        interpolate.griddata(wave_data[:, 0:2], wave_data[:, 2],
                             (grid.detach().cpu().numpy()[:, 0:2] + 1) / 2)
    ).unsqueeze(dim=-1)
    return sln


def coef(grid):
    c = torch.Tensor(
        interpolate.griddata(darcy_2d_coef_data[:, 0:2], darcy_2d_coef_data[:, 2],
                             (grid.detach().cpu().numpy()[:, 0:2] + 1) / 2)
    ).unsqueeze(dim=-1)
    return c


def wave2d_heterogeneous_experiment(grid_res):
    exp_dict_list = []

    x_min, x_max = -1, 1
    y_min, y_max = -1, 1
    t_max = 5
    # grid_res *= 4

    mu_1, mu_2 = -0.5, 0
    sigma = 0.3

    domain = Domain()
    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('y', [y_min, y_max], grid_res)
    domain.variable('t', [0, t_max], grid_res)

    boundaries = Conditions()

    # Initial conditions ###############################################################################################

    def init_func(grid):
        x, y = grid[:, 0], grid[:, 1]
        return np.exp(-((x - mu_1) ** 2 + (y - mu_2) ** 2) / (2 * sigma ** 2))

    # u(x, 0) = f_init(x, 0)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': [y_min, y_max], 't': 0}, value=init_func)

    # u_t(x, 0) = 0
    bop = {
        'du/dt':
            {
                'coeff': 1,
                'du/dx': [2],
                'pow': 1,
                'var': 0
            }
    }
    boundaries.operator({'x': [x_min, x_max], 'y': [y_min, y_max], 't': 0}, operator=bop, value=0)

    # Boundary conditions ##############################################################################################

    def bop_generation(coeff, grid_i):
        bop = {
            'coeff * du/dx_i':
                {
                    'coeff': coeff,
                    'term': [grid_i],
                    'pow': 1
                }
        }
        return bop

    # u_x_min(x_min, y, t) = 0
    bop = bop_generation(-1, 0)
    boundaries.operator({'x': x_min, 'y': [y_min, y_max], 't': [0, t_max]}, operator=bop, value=0)

    # u_x_max(x_max, y, t) = 0
    bop = bop_generation(1, 0)
    boundaries.operator({'x': x_max, 'y': [y_min, y_max], 't': [0, t_max]}, operator=bop, value=0)

    # u_y_min(x, y_min, t) = 0
    bop = bop_generation(-1, 1)
    boundaries.operator({'x': [x_min, x_max], 'y': y_min, 't': [0, t_max]}, operator=bop, value=0)

    # u_y_max(x, y_max, t) = 0
    bop = bop_generation(1, 1)
    boundaries.operator({'x': [x_min, x_max], 'y': y_max, 't': [0, t_max]}, operator=bop, value=0)

    equation = Equation()

    # Operator:  d2u/dx2 + d2u/dy2 - (1 / coef) * d2u/dt2 = 0

    wave_eq = {
        'd2u/dx2**1':
            {
                'coeff': 1,
                'd2u/dx2': [0, 0],
                'pow': 1
            },
        'd2u/dy2**1':
            {
                'coeff': 1,
                'd2u/dx2': [1, 1],
                'pow': 1
            },
        '-(1 / coef) * d2u/dt2**1':
            {
                'coeff': lambda grid: -1 / coef(grid),
                'd2u/dt2': [2, 2],
                'pow': 1
            },
    }

    equation.add(wave_eq)

    neurons = 100

    net = torch.nn.Sequential(
        torch.nn.Linear(3, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
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

    img_dir = os.path.join(os.path.dirname(__file__), 'wave_2d_heterogeneous_medium_img')

    cb_cache = cache.Cache(cache_verbose=True, model_randomize_parameter=1e-6)

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=100,
                                         no_improvement_patience=1000,
                                         patience=5,
                                         randomize_parameter=1e-6,
                                         info_string_every=10)

    cb_plots = plot.Plots(save_every=50,
                          print_every=None,
                          img_dir=img_dir,
                          img_dim='2d')  # 3 image dimension options: 3d, 2d, 2d_scatter

    optimizer = Optimizer('Adam', {'lr': 1e-4})

    model.train(optimizer, 5e6, save_model=True, callbacks=[cb_es, cb_plots, cb_cache])

    end = time.time()

    grid = domain.build('NN').to('cuda')
    net = net.to('cuda')

    error_rmse = torch.sqrt(torch.mean((func(grid).reshape(-1, 1) - net(grid)) ** 2))

    exp_dict_list.append({'grid_res': grid_res, 'time': end - start, 'RMSE': error_rmse.detach().cpu().numpy(),
                          'type': 'wave_eqn_physical', 'cache': True})

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))

    return exp_dict_list


nruns = 10

exp_dict_list = []

for grid_res in range(10, 101, 10):
    for _ in range(nruns):
        exp_dict_list.append(wave2d_heterogeneous_experiment(grid_res))

import pandas as pd

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)
# df.boxplot(by='grid_res',column='time',fontsize=42,figsize=(20,10))
# df.boxplot(by='grid_res',column='RMSE',fontsize=42,figsize=(20,10),showfliers=False)
df.to_csv('examples/benchmarking_data/wave_experiment_physical_10_100_cache={}.csv'.format(str(True)))









