# this script perfom comparing between TEDEOuS and DeepXDE algorithms for burgers equation solution
# It's required to install DeepXDE and tensorflow.

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
import pandas as pd
from scipy.integrate import quad
import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append('../')
sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.input_preprocessing import Equation
from tedeous.solver import Solver, PSO, Plots, grid_format_prepare
from tedeous.solution import Solution
from tedeous.device import solver_device, check_device
from tedeous.models import mat_model

solver_device('cuda')

mode = 'autograd'

def exact(grid):
    mu = 0.02 / np.pi

    def f(y):
        return np.exp(-np.cos(np.pi * y) / (2 * np.pi * mu))

    def integrand1(m, x, t):
        return np.sin(np.pi * (x - m)) * f(x - m) * np.exp(-m ** 2 / (4 * mu * t))

    def integrand2(m, x, t):
        return f(x - m) * np.exp(-m ** 2 / (4 * mu * t))

    def u(x, t):
        if t == 0:
            return -np.sin(np.pi * x)
        else:
            return -quad(integrand1, -np.inf, np.inf, args=(x, t))[0] / quad(integrand2, -np.inf, np.inf, args=(x, t))[
                0]

    solution = []
    for point in grid:
        solution.append(u(point[0].item(), point[1].item()))

    return torch.tensor(solution)


mu = 0.02 / np.pi
x = torch.from_numpy(np.linspace(-1, 1, 21))
t = torch.from_numpy(np.linspace(0, 1, 21))
coord_list = [x, t]
grid = grid_format_prepare(coord_list, mode=mode).float()

##initial cond
bnd1 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()
bndval1 = -torch.sin(np.pi * bnd1[:, 0])

##boundary cond
bnd2 = torch.cartesian_prod(torch.from_numpy(np.array([-1.], dtype=np.float64)), t).float()
bndval2 = torch.zeros_like(bnd2[:, 0])

##boundary cond
bnd3 = torch.cartesian_prod(torch.from_numpy(np.array([1.], dtype=np.float64)), t).float()
bndval3 = torch.zeros_like(bnd3[:, 0])

bconds = [[bnd1, bndval1, 'dirichlet'],
            [bnd2, bndval2, 'dirichlet'],
            [bnd3, bndval3, 'dirichlet']]

burgers_eq = {
    'du/dt**1':
        {
            'coeff': 1.,
            'du/dt': [1],
            'pow': 1,
            'var': 0
        },
    '+u*du/dx':
        {
            'coeff': 1,
            'u*du/dx': [[None], [0]],
            'pow': [1, 1],
            'var': [0, 0]
        },
    '-mu*d2u/dx2':
        {
            'coeff': -mu,
            'd2u/dx2': [0, 0],
            'pow': 1,
            'var': 0
        }
}

model = torch.nn.Sequential(
    torch.nn.Linear(2, 20),
    torch.nn.Tanh(),
    torch.nn.Linear(20, 20),
    torch.nn.Tanh(),
    torch.nn.Linear(20, 20),
    torch.nn.Tanh(),
    torch.nn.Linear(20, 20),
    torch.nn.Tanh(),
    torch.nn.Linear(20, 1)
)


img_dir=os.path.join(os.path.dirname( __file__ ), 'Burg_eq_img')

if not(os.path.isdir(img_dir)):
    os.mkdir(img_dir)

equation = Equation(grid, burgers_eq, bconds).set_strategy(mode)

# m = mat_model(grid, equation).requires_grad_()*0

pso = PSO(
    pop_size=100,
    b=0.9,
    c1=8e-2,
    c2=5e-1,
    variance=1)

model = Solver(grid, equation, model, mode).solve(lambda_bound=100, verbose=1, learning_rate=1e-3, derivative_points=3,
                                                                eps=1e-6, use_cache=False, print_every=100, step_plot_save=True,
                                                                cache_dir='../cache/', patience=2,
                                                                save_always=False, no_improvement_patience=100,
                                                                optimizer_mode=pso, image_save_dir=img_dir)

u_exact = exact(grid).to('cuda')

u_exact = check_device(u_exact).reshape(-1)

u_pred = check_device(model(grid)).reshape(-1)

error_rmse = torch.sqrt(torch.mean((u_exact - u_pred) ** 2))

print('RMSE_grad= ', error_rmse.item())



for j in range(10):

    u_pred = check_device(model(grid)).reshape(-1)

    error_rmse = torch.sqrt(torch.mean((u_exact - u_pred) ** 2))

    print('RMSE_pso= ', error_rmse.item())

    plot = Plots(model.to('cuda'), grid.to('cuda'), 'autograd')

    plot.solution_print(solution_save=True, save_dir=img_dir)