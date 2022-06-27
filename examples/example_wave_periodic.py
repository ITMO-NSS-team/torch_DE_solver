# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:33:44 2021

@author: user
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
# import torch_rbf as rbf
# sys.path.append('../')
sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))


from solver import *
from cache import *
import time

"""
Preparing grid

Grid is an essentially torch.Tensor  of a n-D points where n is the problem
dimensionality
"""

device = torch.device('cpu')

x_grid=np.linspace(0,1,21)
t_grid=np.linspace(0,1,21)

x = torch.from_numpy(x_grid)
t = torch.from_numpy(t_grid)

grid = torch.cartesian_prod(x, t).float()

grid.to(device)

"""
Preparing boundary conditions (BC)

Unlike KdV example there is optional possibility to define only two items
when boundary operator is not needed

bnd=torch.Tensor of a boundary n-D points where n is the problem
dimensionality

bval=torch.Tensor prescribed values at every point in the boundary

"""

# u(x,0) = 1e4 * sin^2(x (x - 1) / 10)
func_bnd1 = lambda x: 10 ** 4 * np.sin((1/10) * x * (x - 1))
bnd1 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()
bndval1 = func_bnd1(bnd1[:,0])

# du/dx (x,0) = 1e3 * sin^2(x (x - 1) / 10)
func_bnd2 = lambda x: 10 ** 3 * np.sin((1/10) * x * (x - 1))
bnd2 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()
bop2 = {
        'du/dx':
            {
                'coeff': 1,
                'du/dx': [0],
                'pow': 1,
                'var': 0
            }
}
bndval2 = func_bnd2(bnd2[:,0])

# u(0,t) = u(1,t)
bnd3_left = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)),t).float()
bnd3_right = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)),t).float()
bnd3 = [bnd3_left,bnd3_right]

# du/dt(0,t) = du/dt(1,t)
bnd4_left = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)),t).float()
bnd4_right = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)),t).float()
bnd4 = [bnd4_left,bnd4_right]

bop4= {
        'du/dt':
            {
                'coeff': 1,
                'du/dt': [1],
                'pow': 1,
                'var': 0
            }
}
bcond_type = 'periodic'

bconds = [[bnd1,bndval1],[bnd2,bop2,bndval2],[bnd3,bcond_type],[bnd4,bop4,bcond_type]]
"""
Defining wave equation

Operator has the form

op=dict in form {'term1':term1,'term2':term2}-> term1+term2+...=0

NB! dictionary keys at the current time serve only for user-frienly 
description/comments and are not used in model directly thus order of
items must be preserved as (coeff,op,pow)



term is a dict term={coefficient:c1,[sterm1,sterm2],'pow': power}

c1 may be integer, function of grid or tensor of dimension of grid

Meaning c1*u*d2u/dx2 has the form

{'coefficient':c1,
 'u*d2u/dx2': [[None],[0,0]],
 'pow':[1,1]}

None is for function without derivatives


"""
# operator is 4*d2u/dx2-1*d2u/dt2=0

wave_eq = {
    'd2u/dt2**1':
        {
            'coeff': -1/4,
            'd2u/dt2': [1,1],
            'pow':1,
            'var':0
        },
        '-C*d2u/dx2**1':
        {
            'coeff': 1,
            'd2u/dx2': [0, 0],
            'pow': 1,
            'var':0
        }
}

model = torch.nn.Sequential(
    torch.nn.Linear(2, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 1)
)

model = point_sort_shift_solver(grid, model, wave_eq, bconds, lambda_bound=1000, verbose=1, learning_rate=1e-4,
                                eps=1e-5, tmin=1000, tmax=1e5, use_cache=False, cache_dir='../cache/',
                                cache_verbose=True,
                                batch_size=None, save_always=True, print_every=500)
