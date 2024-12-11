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

DR_datapath = "grayscott.dat"
b = 0.04
d = 0.1
epsilon_1 = 1e-5
epsilon_2 = 5e-6


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


def DR2d_heterogeneous_experiment(grid_res):
    exp_dict_list = []

    x_min, x_max = -1, 1
    y_min, y_max = -1, 1
    t_max = 200
    grid_res = 20

    domain = Domain()
    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('y', [y_min, y_max], grid_res)
    domain.variable('t', [0, t_max], grid_res)

    boundaries = Conditions()

    # Initial conditions ###############################################################################################

    u_init_func = lambda grid: 1 - torch.exp(-80 * ((grid[:, 0] + 0.05)**2 + (grid[:, 1] + 0.02)**2))
    v_init_func = lambda grid: torch.exp(-80 * ((grid[:, 0] + 0.05)**2 + (grid[:, 1] + 0.02)**2))

    # u(x, y, 0) = u_init_func(x, y, 0)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': [y_min, y_max], 't': 0}, value=u_init_func, var=0)

    # v(x, y, 0) = v_init_func(x, y, 0)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': [y_min, y_max], 't': 0}, value=v_init_func, var=1)

    equation = Equation()

    # Operator 1:  ut = ε1 * ∆u + b * (1 − u) − u * v**2

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

    # Operator 2:  ut = ε2 * ∆v - d * u + u * v**2

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
        torch.nn.Linear(neurons, 2)
    )

    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    start = time.time()

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=1, lambda_bound=100)

    img_dir = os.path.join(os.path.dirname(__file__), 'diffusion_reaction_2d_gray_scott_img')

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

    error_rmse = torch.sqrt(torch.mean((func_data(grid).reshape(-1, 1) - net(grid)) ** 2))

    exp_dict_list.append({'grid_res': grid_res, 'time': end - start, 'RMSE': error_rmse.detach().cpu().numpy(),
                          'type': 'wave_eqn_physical', 'cache': True})

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))

    return exp_dict_list


nruns = 10

exp_dict_list = []

for grid_res in range(10, 101, 10):
    for _ in range(nruns):
        exp_dict_list.append(DR2d_heterogeneous_experiment(grid_res))

import pandas as pd

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)
# df.boxplot(by='grid_res',column='time',fontsize=42,figsize=(20,10))
# df.boxplot(by='grid_res',column='RMSE',fontsize=42,figsize=(20,10),showfliers=False)
df.to_csv('examples/benchmarking_data/wave_experiment_physical_10_100_cache={}.csv'.format(str(True)))









