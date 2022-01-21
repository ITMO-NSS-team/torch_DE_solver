# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:33:44 2021

@author: user
"""
import torch
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys

sys.path.append('../')

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from solver import *
import time

device = torch.device('cpu')

"""
Preparing grid

Grid is an essentially torch.Tensor of a n-D points where n is the problem
dimensionality
"""

t = torch.from_numpy(np.linspace(0, 4*np.pi, 10000))

grid = t.reshape(-1, 1).float()

grid.to(device)

"""
Preparing boundary conditions (BC)

For every boundary we define three items

bnd=torch.Tensor of a boundary n-D points where n is the problem
dimensionality

bop=dict in form {'term1':term1,'term2':term2}-> term1+term2+...=0

NB! dictionary keys at the current time serve only for user-frienly 
description/comments and are not used in model directly thus order of
items must be preserved as (coeff,op,pow)

term is a dict term={coefficient:c1,[sterm1,sterm2],'pow': power}

Meaning c1*u*d2u/dx2 has the form

{'coefficient':c1,
 'u*d2u/dx2': [[None],[0,0]],
 'pow':[1,1]}

None is for function without derivatives


bval=torch.Tensor prescribed values at every point in the boundary
"""

# point t=0
bnd1 = torch.from_numpy(np.array([[0]], dtype=np.float64))

bop1 = None

#  So u(0)=-1/2
bndval1 = torch.from_numpy(np.array([[1.3]], dtype=np.float64))



# Putting all bconds together
bconds = [[bnd1, bndval1]]

"""
Defining Legendre polynomials generating equations

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


# 1-t^2
def c1(grid):
    return torch.cos(grid)


# -2t
def c2(grid):
    return torch.sin(grid)



# this one is to show that coefficients may be a function of grid as well
op= {
    '(1-t^2)*d2u/dt2**1':
        {
            'coeff': c1(grid), #coefficient is a function
            'du/dt': [0],
            'pow': 1
        },
    'n*(n-1)*u**1':
        {
            'coeff': c2(grid),
            'u':  [None],
            'pow': 1
        },
    'n*(n-1)*u**1':
        {
            'coeff': -1,
            'u':  [None],
            'pow': 0
        }
}



for _ in range(1):

    # lp_par={'operator_p':2,
    #     'operator_weighted':False,
    #     'operator_normalized':False,
    #     'boundary_p':1,
    #     'boundary_weighted':False,
    #     'boundary_normalized':False}


    # start = time.time()
    # model = point_sort_shift_solver(grid, model, op, bconds, lambda_bound=100, verbose=True, learning_rate=1e-3,h=abs((t[1]-t[0]).item()),
    #                                 eps=1e-8, tmin=1000, tmax=3e5,use_cache=False,cache_dir='../cache/',cache_verbose=True
    #                                 ,batch_size=None, save_always=False,lp_par=lp_par)
    # end = time.time()

    # print('Time taken 10= ', end - start)

    model = torch.nn.Sequential(
        torch.nn.Linear(1, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 1)
    )

    start = time.time()
    model = point_sort_shift_solver(grid, model, op, bconds, lambda_bound=10, verbose=True, learning_rate=1e-3,
                                    eps=1e-7, tmin=1000, tmax=1e5,use_cache=False,cache_dir='../cache/',cache_verbose=True
                                    ,batch_size=None, save_always=False)
    end = time.time()

    print('Time taken 10= ', end - start)