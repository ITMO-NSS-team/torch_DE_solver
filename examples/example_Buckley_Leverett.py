import torch
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys

sys.path.append('../')
sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from input_preprocessing import Equation
from solver import Solver
from metrics import Solution
import time

m = 0.2
L = 1
Q = -0.1
Sq = 1
mu_water = 0.89e-3
mu_o = 4.62e-3
Swi0 = 0.
Sk = 1.
t_end = 1.

x = torch.from_numpy(np.linspace(0, 1, 21))
t = torch.from_numpy(np.linspace(0, 1, 21))

h = (x[1]-x[0]).item()

grid = torch.cartesian_prod(x,t).float()

##initial cond
bnd1 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()
bndval1 = torch.zeros_like(x)+Swi0

##boundary cond
bnd2 = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), t).float()
bndval2 = torch.zeros_like(t)+Sk

bconds = [[bnd1, bndval1], [bnd2, bndval2]]

model = torch.nn.Sequential(
    torch.nn.Linear(2, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 1)
)

def k_oil(x):
    return (1-model(x))**2

def k_water(x):
    return (model(x))**2

def dk_water(x):
    return 2*model(x)

def dk_oil(x):
    return -2*(1-model(x))

def df(x):
    return (dk_water(x)*(k_water(x)+mu_water/mu_o*k_oil(x))-
            k_water(x)*(dk_water(x)+mu_water/mu_o*dk_oil(x)))/(k_water(x)+mu_water/mu_o*k_oil(x))**2

def coef_model(x):
    return -Q/Sq*df(x)

buckley_eq = {
    'm*ds/dt**1':
        {
            'coeff': m,
            'ds/dt': [1],
            'pow': 1
        },
    '-Q/Sq*df*ds/dx**1':
        {
            'coeff': coef_model,
            'ds/dx': [0],
            'pow':1
        }
}

equation = Equation(grid, buckley_eq, bconds).set_strategy('autograd')

img_dir=os.path.join(os.path.dirname( __file__ ), 'Buckley_NN_img')

if not(os.path.isdir(img_dir)):
    os.mkdir(img_dir)

model = Solver(grid, equation, model, 'autograd').solve(lambda_bound=1, verbose=1, learning_rate=1e-3,
                                    eps=1e-6, tmin=1000, tmax=1e6, use_cache=False, cache_dir='../cache/', cache_verbose=True,
                                    save_always=False, no_improvement_patience=500, print_every=500, optimizer_mode='Adam',
                                    step_plot_print=False, step_plot_save=True, image_save_dir=img_dir)