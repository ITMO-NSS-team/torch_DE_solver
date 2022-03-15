import torch
import numpy as np
from solver import *
from cache import *

x_lower = -5
x_upper = 5
t_lower = 0
t_upper = np.pi / 2

device = torch.device('cpu')

x_grid = np.linspace(x_lower, x_upper, 256)
t_grid = np.linspace(t_lower, t_upper, 201)

x = torch.from_numpy(x_grid)
t = torch.from_numpy(t_grid)

grid = torch.cartesian_prod(x, t).float()

grid.to(device)

func = lambda x: 2/np.cosh(x)
# Initial conditions at t=0
bnd1 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()

# u(0,x)=2sech(x)
bndval1 = func(bnd1[:, 0])

# Initial conditions at t=pi/2
bnd2 = torch.cartesian_prod(x, torch.from_numpy(np.array([t_upper], dtype=np.float64))).float()

# u(pi/2,x)=2sech(x)
bndval2 = func(bnd2[:, 0])

# u(0,x)=u(pi/2,x)

# Boundary cond at x=0

bnd3 = bnd3 = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), t).float()

# u(0,t)=0
bndval3 = torch.from_numpy(np.zeros(len(bnd3), dtype=np.float64))

# Boundary conditions at x=1
bnd4 = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)), t).float()

# u(1,t)=0
bndval4 = torch.from_numpy(np.zeros(len(bnd4), dtype=np.float64))

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
                                    batch_size=32, save_always=True,no_improvement_patience=None)

