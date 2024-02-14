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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import early_stopping, plot, adaptive_lambda
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device

"""
Preparing grid

Grid is an essentially torch.Tensor  of a n-D points where n is the problem
dimensionality
"""

solver_device('cuda')

domain = Domain()
domain.variable('x', [0, 1], 20)
domain.variable('t', [0, 1], 20)

A = 0.5
C = 2

def func(grid):
    x, t = grid[:,1],grid[:,0]
    return torch.sin(np.pi * x) * torch.cos(C * np.pi * t) + \
            A * torch.sin(2 * C * np.pi * x) * torch.cos(4 * C * np.pi * t)

boundaries = Conditions()

# Initial conditions at t=0
boundaries.dirichlet({'t': [0, 1], 'x': 0}, value=func)

# Boundary conditions at x=1
boundaries.dirichlet({'t': [0, 1], 'x': 1}, value=func)

# Initial conditions at t=0
boundaries.dirichlet({'t': 0, 'x': [0, 1]}, value=func)

# Initial conditions (operator) at t=0
bop4= {
        'du/dt':
            {
                'coeff': 1,
                'du/dt': [0],
                'pow': 1,
            }
}
boundaries.operator({'t': 0, 'x': [0, 1]}, operator=bop4, value=func)

equation = Equation()

# operator is 4*d2u/dx2-1*d2u/dt2=0
wave_eq = {
    '-C*d2u/dx2**1':
        {
            'coeff': -4,
            'd2u/dx2': [1, 1],
            'pow': 1
        },
    'd2u/dt2**1':
        {
            'coeff': 1,
            'd2u/dt2': [0, 0],
            'pow':1
        }
}

equation.add(wave_eq)

net = torch.nn.Sequential(
    torch.nn.Linear(2, 256),
    torch.nn.Tanh(),
    torch.nn.Linear(256, 256),
    torch.nn.Tanh(),
    torch.nn.Linear(256, 256),
    torch.nn.Tanh(),
    torch.nn.Linear(256, 256),
    torch.nn.Tanh(),
    torch.nn.Linear(256, 1))

model =  Model(net, domain, equation, boundaries)

model.compile("autograd", lambda_operator=1, lambda_bound=100)

cb_es = early_stopping.EarlyStopping(eps=1e-7,
                                    loss_window=1000,
                                    no_improvement_patience=1000,
                                    patience=10,
                                    randomize_parameter=1e-5,
                                    abs_loss=0.1,
                                    info_string_every=500)

img_dir=os.path.join(os.path.dirname( __file__ ), 'wave_eq_img')

cb_plots = plot.Plots(save_every=500, print_every=None, img_dir=img_dir)

cb_lambda = adaptive_lambda.AdaptiveLambda()

optimizer = Optimizer('Adam', {'lr': 1e-3}, gamma=0.9, decay_every=1000)

model.train(optimizer, 1e5, save_model=False, callbacks=[cb_es, cb_plots, cb_lambda])
