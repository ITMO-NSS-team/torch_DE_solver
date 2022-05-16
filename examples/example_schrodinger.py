import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy

import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
from solver import *
from cache import *

device = torch.device('cpu')

x_grid=np.linspace(0,1,21)
t_grid=np.linspace(0,1,21)

x = torch.from_numpy(x_grid)
t = torch.from_numpy(t_grid)

grid = torch.cartesian_prod(x, t).float()

grid.to(device)

## BOUNDARY AND INITIAL CONDITIONS
def func(grid):
    x, t = grid[:,0],grid[:,1]
    return torch.sin(np.pi * x) * torch.cos(C * np.pi * t) + torch.sin(A * np.pi * x) * torch.cos(
        A * C * np.pi * t
    )
fun = lambda x: 2/np.cosh(x)

A = 2
C = 10
# Initial conditions at t=0
bnd1 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()

# u(0,x)=sin(pi*x)
bndval1_1 = func(bnd1)
bndval1_2 = fun(bnd1[:, 0])
bndval1 = torch.stack((bndval1_1,bndval1_2),dim=1)

# Initial conditions at t=1
bnd2 = torch.cartesian_prod(x, torch.from_numpy(np.array([1], dtype=np.float64))).float()

# u(1,x)=sin(pi*x)
bndval2_1 = func(bnd2)
bndval2_2 = fun(bnd2[:, 0])
bndval2 = torch.stack((bndval2_1,bndval2_2),dim=1)

# Boundary conditions at x=0
bnd3 = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), t).float()

# u(0,t)=0
bndval3_1 = func(bnd3)
bndval3_2 = fun(bnd3[:, 0])
bndval3 = torch.stack((bndval3_1,bndval3_2),dim=1)

# Boundary conditions at x=1
bnd4 = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)), t).float()

# u(1,t)=0
bndval4_1 = func(bnd4)
bndval4_2 = fun(bnd4[:, 0])
bndval4 = torch.stack((bndval4_1,bndval4_2),dim=1)

bcond_type = 'periodic'
# Putting all bconds together
bconds = [[bnd1, bndval1], [bnd2, bndval2], [bnd3, bndval3], [bnd4, bndval4]]


'''
schrodinger equation:
i * dh/dt + 1/2 * d2h/dx2 + abs(h)**2 * h = 0 

real part: 
du/dt + 1/2 * d2v/dx2 + (u**2 + v**2) * v

imag part:
dv/dt - 1/2 * d2u/dx2 - (u**2 + v**2) * v

u = var:0
v = var:1

'''

schrodinger_eq_real = {
    'du/dt':
        {
            'const': 1,
            'term': [0],
            'power': 1,
            'var': 0
        },
    '1/2*d2v/dx2':
        {
            'const': 1 / 2,
            'term': [1, 1],
            'power': 1,
            'var': 1
        },
    'v * u**2':
        {
            'const': 1,
            'term': [[None], [None]],
            'power': [1, 2],
            'var': [0, 1]
        },
    'v**3':
        {
            'const': 1,
            'term': [None],
            'power': 3,
            'var': 1
        }

}

model = torch.nn.Sequential(
        torch.nn.Linear(2, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 2)
    )

model = point_sort_shift_solver(grid, model, schrodinger_eq_real , bconds,
                                              lambda_bound=1000, verbose=1, learning_rate=1e-3,
                                    eps=1e-6, tmin=1000, tmax=1e5,use_cache=True,cache_dir='../cache/',cache_verbose=True,
                                    batch_size=None, save_always=True,no_improvement_patience=500,print_every = 500,show_plot=False)

