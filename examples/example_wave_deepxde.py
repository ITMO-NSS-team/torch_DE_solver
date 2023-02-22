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
from tedeous.nn import NN
from tedeous.device import solver_device


solver_device('cuda')
A = 2
C = 10

def func(grid):
    x, t = grid[:,0],grid[:,1]
    return torch.sin(np.pi * x) * torch.cos(C * np.pi * t) + torch.sin(A * np.pi * x) * torch.cos(
        A * C * np.pi * t
    )


grid_res = 10
x_grid = np.linspace(0, 1, grid_res + 1)
t_grid = np.linspace(0, 1, grid_res + 1)

x = torch.from_numpy(x_grid)
t = torch.from_numpy(t_grid)

h = abs((t[1] - t[0]).item())

grid = torch.cartesian_prod(x, t).float()

# Initial conditions at t=0
bnd1 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()

# u(x,0)=sin(pi*x) + sin(2pi*x)
bndval1 = func(bnd1)

# Initial conditions at t=1
bnd2 = torch.cartesian_prod(x, torch.from_numpy(np.array([1], dtype=np.float64))).float()

# u(x,1)=0
bndval2 = func(bnd2)

# Boundary conditions at x=0
bnd3 = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), t).float()

# u(x,0)=0
bndval3 = func(bnd3)

# Boundary conditions at x=0
bnd4 = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), t).float()

# u_t(x,0)=0
bndval4 = func(bnd4)

bop4= {
        'du/dt':
            {
                'coeff': 1,
                'du/dx': [1],
                'pow': 1,
            }
}

# Putting all bconds together
bconds = [[bnd1, bndval1, 'dirichlet'],
          [bnd2, bndval2, 'dirichlet'],
          [bnd3, bndval3, 'dirichlet'],
          [bnd4, bop4, bndval4, 'operator']]

wave_eq = {
        'd2u/dt2**1':
            {
                'coeff': 1.,
                'd2u/dt2': [1,1],
                'pow':1
            },
            '-C*d2u/dx2**1':
            {
                'coeff': -C**2,
                'd2u/dx2': [0, 0],
                'pow': 1
            }
    }

model = NN(2,[100] * 3, 1,activations='tanh', fourier_features = True, sigma = 10, mapping_size = 256)

equation = Equation(grid, wave_eq, bconds, h=h).set_strategy('NN')

img_dir = os.path.join(os.path.dirname(__file__), 'wave_example_paper_img')

model = Solver(grid, equation, model, 'NN').solve(lambda_bound=100, verbose=True, learning_rate=1e-4,
                                                  eps=1e-8, tmax=1e6, use_cache=False, cache_verbose=True,
                                                  save_always=True, print_every=500, model_randomize_parameter=1e-5,
                                                  optimizer_mode='Adam', no_improvement_patience=1000,
                                                  step_plot_print=True, step_plot_save=False, image_save_dir=img_dir)
