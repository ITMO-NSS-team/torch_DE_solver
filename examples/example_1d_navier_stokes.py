import torch
import math
import matplotlib.pyplot as plt
import scipy
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.input_preprocessing import Equation
from tedeous.solver import Solver
from tedeous.metrics import Solution
from tedeous.device import solver_device
from tedeous.models import Fourier_embedding

solver_device('cuda')
grid_res = 30

x = torch.linspace(0, 5, grid_res + 1)
t = torch.linspace(0, 2, grid_res + 1)

h = abs((t[1] - t[0]).item())

grid = torch.cartesian_prod(t, x)

def bconds_1():
    # Boundary conditions at x=0
    bnd1_u = torch.cartesian_prod(torch.Tensor([0]), t)
    # u(0,t) = 2
    bndval1_u = 2 * torch.ones_like(bnd1_u[:, 0])
    bnd_type_1_u = 'dirichlet'

    # Boundary conditions at x=5
    bnd2_u = torch.cartesian_prod(torch.Tensor([5]), t)
    # u(5,t) = 2
    bndval2_u = 2 * torch.ones_like(bnd2_u[:, 0])
    bnd_type_2_u = 'dirichlet'

    # Boundary conditions at x=0
    bnd1_p = torch.cartesian_prod(torch.Tensor([0]), t)
    # p(0,t) = 0
    bndval1_p = torch.zeros_like(bnd1_p[:, 0])
    bnd_type_1_p = 'dirichlet'

    # Boundary conditions at x=5
    bnd2_p = torch.cartesian_prod(torch.Tensor([5]), t)
    # p(5,t) = 0
    bndval2_p = torch.zeros_like(bnd2_p[:, 0])
    bnd_type_2_p = 'dirichlet'

    # Initial condition at t=0
    ics_u = torch.cartesian_prod(x, torch.Tensor([0]))
    icsval_u = torch.sin((2 * math.pi * x) / 5) + 2
    ics_type_u = 'dirichlet'

    bconds = [[bnd1_u, bndval1_u, 0, bnd_type_1_u],
              [bnd1_p, bndval1_p, 1, bnd_type_1_p],
              [bnd2_u, bndval2_u, 0, bnd_type_2_u],
              [bnd2_p, bndval2_p, 1, bnd_type_2_p],
              [ics_u, icsval_u, 0, ics_type_u]]
    return bconds

def bconds_2():
    # Boundary conditions at x=0
    bnd1_u = torch.cartesian_prod(t, torch.Tensor([0]))
    bop1_u = {
                'du/dt':
                    {
                    'coeff': 1,
                    'term': [0],
                    'pow': 1,
                    'var': 0
                    }
            }
    # u_t = t*sin(t)
    bval1_u = t * torch.sin(t)
    bnd_type_1_u = 'operator'

    # Boundary conditions at x=5
    bnd2_u = torch.cartesian_prod(t, torch.Tensor([5]))
    bop2_u = {
        'du/dt':
            {
                'coeff': 1,
                'term': [0],
                'pow': 1,
                'var': 0
            }
    }
    # u_t = t*sin(t)
    bval2_u = t * torch.sin(t)
    bnd_type_2_u = 'operator'

    # Boundary conditions at x=0
    bnd1_p = torch.cartesian_prod(t, torch.Tensor([0]))
    # p(0,t) = 0
    bndval1_p = torch.zeros_like(bnd1_p[:, 0])
    bnd_type_1_p = 'dirichlet'

    # Boundary conditions at x=5
    bnd2_p = torch.cartesian_prod(t, torch.Tensor([5]))
    # p(5,t) = 0
    bndval2_p = torch.zeros_like(bnd2_p[:, 0])
    bnd_type_2_p = 'dirichlet'

    # Initial condition at t=0
    ics_u = torch.cartesian_prod(torch.Tensor([0]), x)
    icsval_u = torch.sin((math.pi * x) / 5) + 1
    ics_type_u = 'dirichlet'

    bconds = [[bnd1_u, bop1_u,bval1_u, 0, bnd_type_1_u],
              [bnd1_p, bndval1_p, 1, bnd_type_1_p],
              [bnd2_u,bop2_u, bval2_u, 0, bnd_type_2_u],
              [bnd2_p, bndval2_p, 1, bnd_type_2_p],
              [ics_u, icsval_u, 0, ics_type_u]]
    return bconds

bconds = bconds_2()
ro = 1
mu = 1

NS_1 = {
    'du/dx':
        {
        'coeff': 1,
        'term': [1],
        'pow': 1,
        'var': 0
        }
}

NS_2 = {
    'du/dt':
        {
        'coeff': 1,
        'term': [0],
        'pow': 1,
        'var': 0
        },
    'u * du/dx':
        {
        'coeff': 1,
        'term': [[None], [1]],
        'pow': [1, 1],
        'var': [0, 0]
        },
    '1/ro * dp/dx':
        {
        'coeff': 1/ro,
        'term': [1],
        'pow': 1,
        'var': 1
        },
    '-mu * d2u/dx2':
        {'coeff': -mu,
        'term': [1, 1],
        'pow': 1,
        'var': 0
         }
}

navier_stokes = [NS_1, NS_2]
FFL = Fourier_embedding(L=[1, 1], M=[1, 1])
out = FFL.out_features

model = torch.nn.Sequential(
    torch.nn.Linear(2, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 2)
)

equation = Equation(grid, navier_stokes, bconds, h=h).set_strategy('autograd')

img_dir = os.path.join(os.path.dirname(__file__), 'navier_stokes_img')

model = Solver(grid, equation, model, 'autograd').solve(tol=0.1,lambda_bound=1000,verbose=True, learning_rate=1e-5,
                                                  eps=1e-5, tmax=1e6, use_cache=False, cache_verbose=True,
                                                  save_always=True, print_every=5000, model_randomize_parameter=1e-5,
                                                  optimizer_mode='Adam', no_improvement_patience=1000, patience=10,
                                                  step_plot_print=False, step_plot_save=True, image_save_dir=img_dir)

# model = Solver(grid, equation, model, 'autograd').solve(lambda_bound=1000,tol=0.1,verbose=True, learning_rate=1e-4,
#                                                   eps=1e-4, tmax=1e6, use_cache=False, cache_verbose=True,
#                                                   save_always=True, print_every=5000, model_randomize_parameter=1e-5,
#                                                   optimizer_mode='LBFGS', no_improvement_patience=1000,
#                                                   step_plot_print=500, step_plot_save=True, image_save_dir=img_dir)