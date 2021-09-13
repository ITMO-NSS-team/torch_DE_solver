# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 20:54:57 2021

@author: Sashka
"""
import torch
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys

sys.path.append('../')

from solver import *
from cache import *
import time

"""
Preparing grid

Grid is an essentially torch.Tensor  of a n-D points where n is the problem
dimensionality
"""

device = torch.device('cpu')

x = torch.from_numpy(np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]))
t = torch.from_numpy(np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]))

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

# Initial conditions at t=0
bnd1 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()

# u(0,x)=sin(pi*x)
bndval1 = torch.sin(np.pi * bnd1[:, 0])

# Initial conditions at t=1
bnd2 = torch.cartesian_prod(x, torch.from_numpy(np.array([1], dtype=np.float64))).float()

# u(1,x)=sin(pi*x)
bndval2 = torch.sin(np.pi * bnd2[:, 0])

# Boundary conditions at x=0
bnd3 = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), t).float()

# u(0,t)=0
bndval3 = torch.from_numpy(np.zeros(len(bnd3), dtype=np.float64))

# Boundary conditions at x=1
bnd4 = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)), t).float()

# u(1,t)=0
bndval4 = torch.from_numpy(np.zeros(len(bnd4), dtype=np.float64))

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
    '4*d2u/dx2**1':
        {
            'coeff': 4,
            'd2u/dx2': [0, 0],
            'pow': 1
        },
    '-d2u/dt2**1':
        {
            'coeff': -1,
            'd2u/dt2': [1,1],
            'pow':1
        }
}


    
def batch_bconds_transform(batch_grid,bconds):
    batch_bconds=[]
    bconds = bnd_unify(bconds)
    for bcond in bconds:
        b_coord = bcond[0]
        bop = bcond[1]
        bval = bcond[2]
    for point in bcoords:
        pos = int(torch.where(torch.all(torch.isclose(grid, point), dim=1))[0])
        bndposlist.append(pos)
    
    return batch_bconds

for _ in range(1):
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 1)
    )

    
    # from input_preprocessing import grid_prepare, bnd_prepare, operator_prepare
    
    # start = time.time()
    
    # prepared_grid = grid_prepare(grid)
    # bconds = bnd_prepare(bconds, prepared_grid, h=0.001)
    # operator = operator_prepare(wave_eq, prepared_grid, subset=['central'], true_grid=grid, h=0.001)
    
    # loss = point_sort_shift_loss(model, prepared_grid, operator, bconds, lambda_bound=1000)
    
    # print(loss)
    # # model = point_sort_shift_solver(grid, model, wave_eq , bconds, 
    # #                                           lambda_bound=100, verbose=True, learning_rate=1e-4,
    # #                                 eps=1e-5, tmin=1000, tmax=1e5,use_cache=True,cache_dir='../cache/',cache_verbose=True)
    
    # end = time.time()
    # print('Time taken 10= ', end - start)
    
    batch_size=32

    permutation = torch.randperm(grid.size()[0])
    
    for i in range(0,grid.size()[0], batch_size):
    
        indices = permutation[i:i+batch_size]
        batch= grid[indices]
    
        prepared_grid = grid_prepare(batch)
        bconds = bnd_prepare(bconds, prepared_grid, h=0.001)
        operator = operator_prepare(wave_eq, prepared_grid, subset=None, true_grid=batch, h=0.001)
        
        loss = point_sort_shift_loss(model, prepared_grid, operator, bconds, lambda_bound=1000)

        print(loss)