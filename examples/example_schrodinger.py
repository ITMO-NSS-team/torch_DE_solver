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

x_grid = np.linspace(-5,5,25)
t_grid = np.linspace(0,np.pi/2,21)

x = torch.from_numpy(x_grid)
t = torch.from_numpy(t_grid)

grid = torch.cartesian_prod(x, t).float()

grid.to(device)

## BOUNDARY AND INITIAL CONDITIONS
fun = lambda x: 2/np.cosh(x)
# u(x,0) = 2sech(x), v(x,0) = 0
bnd1 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()

bndval1_re = fun(bnd1)
bndval1_im = torch.from_numpy(np.zeros_like(bnd1))
bndval1 = [bndval1_re, bndval1_im]
# u(-5,t), u(5,t)
bnd_re_1 = torch.cartesian_prod(torch.from_numpy(np.array([-5], dtype=np.float64)), t).float()
bnd_re_2 = torch.cartesian_prod(torch.from_numpy(np.array([5], dtype=np.float64)), t).float()
bnd2 = [bnd_re_1,bnd_re_2]

# v(-5,t), v(5,t)
bnd_im_1 = torch.cartesian_prod(torch.from_numpy(np.array([-5], dtype=np.float64)), t).float()
bnd_im_2 = torch.cartesian_prod(torch.from_numpy(np.array([5], dtype=np.float64)), t).float()
bnd3 = [bnd_im_1,bnd_im_2]

bcond_type = 'periodic'
bconds = [[bnd1,bndval1],[bnd2,bcond_type],[bnd3,bcond_type]]

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
                                    batch_size=None, save_always=True,no_improvement_patience=500,print_every = 500,show_plot=True)


