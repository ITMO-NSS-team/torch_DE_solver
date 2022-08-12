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
from input_preprocessing import *
import time
device = torch.device('cpu')

x_grid = np.linspace(0,1,21)
t_grid = np.linspace(0,1,21)

x = torch.from_numpy(x_grid)
t = torch.from_numpy(t_grid)

grid = torch.cartesian_prod(x, t).float()

grid.to(device)

# Boundary and initial conditions.

# u(x,0) = 0, v(x,0) = exp(50 * (x - 1/2) ** 2)

eq_v = lambda x: np.exp(50 * (x - 1/2) ** 2)

bnd1_u = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()
bnd1_v = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()
bnd1 = [bnd1_u, bnd1_v]

bndval1_u = torch.from_numpy(np.zeros_like(bnd1_u[:,0]))
bndval1_v = eq_v(bnd1_v[:,0])
bndval1 = [bndval1_u, bndval1_v]

# u(0,t) = u(1,t), v(0,t) = v(1,t)
bnd2_u_left = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), t).float()
bnd2_u_right = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)), t).float()
bnd2_u = [bnd2_u_left, bnd2_u_right]

bnd2_v_left = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), t).float()
bnd2_v_right = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)), t).float()
bnd2_v = [bnd2_v_left, bnd2_v_right]

bnd2 = [bnd2_u, bnd2_v]

bconds_type = 'periodic'

bconds = [[bnd1, bndval1], [bnd2, bconds_type]]

# Operator
'''
du/dt = -dv/dx
dv/dt = -du/dx
'''
equation_1 = {
    'du/dt':
        {
            'const': 1,
            'term': [1],
            'power': 1,
            'var': 0
        },
    'dv/dx':
        {
            'const': 1,
            'term': [0],
            'power': 1,
            'var': 1,
        }
    }
equation_2 = {
    'dv/dt':
        {
            'const': 1,
            'term': [1],
            'power': 1,
            'var': 1
        },
    'du/dx':
        {
            'const': 1,
            'term': [0],
            'power': 1,
            'var': 0,
        }
    }

equation = [equation_1, equation_2]

model = torch.nn.Sequential(
    torch.nn.Linear(2, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 2)
)

model = point_sort_shift_solver(grid, model, equation , bconds,
                                              lambda_bound=1000, verbose=1, learning_rate=1e-4,
                                    eps=1e-6, tmin=1000, tmax=1e5,use_cache=False,cache_dir='../cache/',cache_verbose=True,
                                    batch_size=None, save_always=True,no_improvement_patience=500,print_every = 500)