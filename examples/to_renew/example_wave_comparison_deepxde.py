# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:33:44 2021

@author: user
"""

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
# import torch_rbf as rbf
# sys.path.append('../')
sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.cache import *
import time

"""
Preparing grid

Grid is an essentially torch.Tensor  of a n-D points where n is the problem
dimensionality
"""

device = torch.device('cpu')

x_grid=np.linspace(0,1,11)
t_grid=np.linspace(0,1,11)

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

A = 2
C = 10

def func(grid):
    x, t = grid[:,0],grid[:,1]
    return torch.sin(np.pi * x) * torch.cos(C * np.pi * t) + torch.sin(A * np.pi * x) * torch.cos(
        A * C * np.pi * t
    )


# Initial conditions at t=0
bnd1 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()

# u(0,x)=sin(pi*x)
bndval1 = func(bnd1)

# Initial conditions at t=1
bnd2 = torch.cartesian_prod(x, torch.from_numpy(np.array([1], dtype=np.float64))).float()

# u(1,x)=sin(pi*x)
bndval2 = func(bnd2)

# Boundary conditions at x=0
bnd3 = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), t).float()

# u(0,t)=0
bndval3 = func(bnd3)

# Boundary conditions at x=1
bnd4 = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)), t).float()

# u(1,t)=0
bndval4 = func(bnd4)

# Putting all bconds together
bconds = [[bnd1, bndval1], [bnd2, bndval2], [bnd3, bndval3], [bnd4, bndval4]]

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
            'coeff': 1,
            'd2u/dt2': [1,1],
            'pow':1,
            'var':0
        },
        '-C*d2u/dx2**1':
        {
            'coeff': -C,
            'd2u/dx2': [0, 0],
            'pow': 1,
            'var':0
        }
}



for _ in range(1):
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 1)
    )

    
    start = time.time()
    
    model = point_sort_shift_solver(grid, model, wave_eq , bconds, lambda_bound=1000, verbose=1, learning_rate=1e-4,
                                    eps=1e-5, tmin=1000, tmax=1e5,use_cache=True,cache_dir='../cache/',cache_verbose=True,
                                    batch_size=32, save_always=True,print_every = 500)


    end = time.time()
    print('Time taken 10= ', end - start)
