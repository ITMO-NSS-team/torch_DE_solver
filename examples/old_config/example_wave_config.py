# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:33:44 2021

@author: user
"""
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
sys.path.append('../')

from tedeous.solver import *
# from cache import *
from tedeous.config import Config
import time

"""
Preparing grid

Grid is an essentially torch.Tensor  of a n-D points where n is the problem
dimensionality
"""

device = torch.device('cpu')

x = torch.from_numpy(np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]))
t = torch.from_numpy(np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]))

coord_list=[x,t]

# grid = torch.cartesian_prod(*coord_list).float()

# grid.to(device)

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

#default config used
config=Config()
    
for _ in range(1):
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 1)
    )

    
    
    start = time.time()
    
    lp_par={'operator_p':2,
            'operator_weighted':False,
            'operator_normalized':False,
            'boundary_p':2,
            'boundary_weighted':False,
            'boundary_normalized':False}
    
    # model = point_sort_shift_solver(grid, model, wave_eq , bconds, 
    #                                           lambda_bound=100, verbose=True, learning_rate=1e-4,
    #                                 eps=1e-5, tmin=1000, tmax=1e5,use_cache=False,cache_dir='../cache/',cache_verbose=True,
    #                                 batch_size=None, save_always=True,lp_par=lp_par)
    model=optimization_solver(coord_list, model, wave_eq,bconds,config,mode='mat')
    end = time.time()
    print('Time taken 10= ', end - start)
