# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:33:44 2021

@author: user
"""
import torch
import numpy as np
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
import time

solver_device('cpu')

x_min, x_max = 0, 1
y_min, y_max = 0, 1
t_max = 100
grid_res = 40

m1, m2, k = 4, 2, 1

domain = Domain()

domain.variable('x', [x_min, x_max], grid_res)
domain.variable('y', [y_min, y_max], grid_res)
domain.variable('t', [0, t_max], grid_res)

x = domain.variable_dict['x']
y = domain.variable_dict['y']

boundaries = Conditions()

# Initial condition: ###############################################################################################

# u(x, y, 0)
boundaries.dirichlet({'x': [x_min, x_max], 'y': [y_min, y_max], 't': 0},
                     value=torch.sin(4 * np.pi * x) * torch.sin(3 * np.pi * y))

# Boundary conditions: #############################################################################################

# u(0, y, t)
boundaries.dirichlet({'x': x_min, 'y': [y_min, y_max], 't': [0, t_max]}, value=0)
# u(x, 0, t)
boundaries.dirichlet({'x': [x_min, x_max], 'y': y_min, 't': [0, t_max]}, value=0)
# u(1, y, t)
boundaries.dirichlet({'x': x_max, 'y': [y_min, y_max], 't': [0, t_max]}, value=0)
# u(x, 1, t)
boundaries.dirichlet({'x': [x_min, x_max], 'y': y_max, 't': [0, t_max]}, value=0)

equation = Equation()

def coeff_u(grid):
    x, y, t = grid[:, 0], grid[:, 1], grid[:, 2]
    return -5 * (1 + 2 * torch.sin(torch.pi * t / 4)) * \
           torch.sin(m1 * torch.pi * x) * torch.sin(m2 * torch.pi * y)

# Operator: du/dt -  0.001 * ∆u + 5sin(k * u**2) * (1 + 2sin(pi * t / 4)) * sin(m1 * pi * x) * sin(m2 * pi * y)

heat_LT = {
    'du/dt**1':
        {
            'coeff': 1,
            'term': [2],
            'pow': 1,
            'var': 0
        },
    '-0.001 * d2u/dx2**1':
        {
            'coeff': -0.001,
            'term': [0, 0],
            'pow': 1,
            'var': 0
        },
    '-0.001 * d2u/dy2**1':
        {
            'coeff': -0.001,
            'term': [1, 1],
            'pow': 1,
            'var': 0
        },
    'coeff_u * sin(k * u ** 2)':
        {
            'coeff': coeff_u,
            'term': [None],
            'pow': lambda u: torch.sin(k * u ** 2),
            'var': 0
        }
}

equation.add(heat_LT)

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

model = Model(net, domain, equation, boundaries)

model.compile('autograd', lambda_operator=1, lambda_bound=100)

img_dir = os.path.join(os.path.dirname(__file__), 'heat_2d_long_time_img')

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

optimizer = Optimizer('Adam', {'lr': 1e-3})

callbacks = [cb_cache, cb_es, cb_plots]

model.train(optimizer, 1e6, save_model=True, callbacks=callbacks)
