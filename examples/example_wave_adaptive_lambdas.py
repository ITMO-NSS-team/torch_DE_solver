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


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.append('../')

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.input_preprocessing import Equation
from tedeous.solver import Solver
from tedeous.solution import Solution
from tedeous.device import solver_device

"""
Preparing grid

Grid is an essentially torch.Tensor  of a n-D points where n is the problem
dimensionality
"""

solver_device('cuda')

x = torch.from_numpy(np.linspace(0, 1, 21))
t = torch.from_numpy(np.linspace(0, 1, 21))

h = (x[1]-x[0]).item()

grid = torch.cartesian_prod(t,x).float()

A = 0.5
C = 2

def func(grid):
    x, t = grid[:,1],grid[:,0]
    return torch.sin(np.pi * x) * torch.cos(C * np.pi * t) + \
            A * torch.sin(2 * C * np.pi * x) * torch.cos(4 * C * np.pi * t)

# Initial conditions at t=0
bnd1 = torch.cartesian_prod(t, torch.from_numpy(np.array([0], dtype=np.float64))).float()
bndval1 = func(bnd1)

# Boundary conditions at x=1
bnd2 = torch.cartesian_prod(t, torch.from_numpy(np.array([1], dtype=np.float64))).float()
bndval2 = func(bnd2)

# Initial conditions at t=0
bnd3 = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), x).float()

# u(x,0)=sin(pi*x) + 1/2 * sin(4pi*x)
bndval3 = func(bnd3)

# Initial conditions (operator) at t=0
bnd4 = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), x).float()

# u(x,0)=0
bndval4 = func(bnd4)

bop4= {
        'du/dt':
            {
                'coeff': 1,
                'du/dt': [0],
                'pow': 1,
            }
}

bconds = [[bnd1, bndval1, 'dirichlet'],
          [bnd2, bndval2, 'dirichlet'],
          [bnd3, bndval3, 'dirichlet'],
          [bnd4, bop4, bndval4, 'operator']]



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

model = torch.nn.Sequential(
    torch.nn.Linear(2, 256),
    torch.nn.Tanh(),
    torch.nn.Linear(256, 256),
    torch.nn.Tanh(),
    torch.nn.Linear(256, 256),
    torch.nn.Tanh(),
    torch.nn.Linear(256, 256),
    torch.nn.Tanh(),
    torch.nn.Linear(256, 1))


img_dir=os.path.join(os.path.dirname( __file__ ), 'wave_eq_img')

if not(os.path.isdir(img_dir)):
    os.mkdir(img_dir)


equation = Equation(grid, wave_eq, bconds).set_strategy('autograd')

model = Solver(grid, equation, model, 'autograd').solve(update_every_lambdas= 500,
                              verbose=1, learning_rate=1e-3, eps=1e-7, tmin=1000, tmax=1e5, gamma=0.9,
                              use_cache=False, cache_dir='../cache/',cache_verbose=True,
                              save_always=True, print_every=500, patience=10,
                              loss_oscillation_window=1000, no_improvement_patience=1000,
                              model_randomize_parameter=0, optimizer_mode='Adam', cache_model=None,
                              step_plot_print=True, step_plot_save=False, abs_loss=0.1, image_save_dir=img_dir)