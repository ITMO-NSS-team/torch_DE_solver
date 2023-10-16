import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.input_preprocessing import Equation
from tedeous.solver import Solver
from tedeous.solution import Solution
from tedeous.device import solver_device
from tedeous.models import parameter_registr


solver_device('cuda')

x = torch.from_numpy(np.linspace(-1, 1, 61))
t = torch.from_numpy(np.linspace(0, 1, 61))

grid = torch.cartesian_prod(x, t).float()

data = scipy.io.loadmat(os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'wolfram_sln/Burgers.mat')))

x = torch.tensor(data['x']).reshape(-1)
t = torch.tensor(data['t']).reshape(-1)

usol = data['usol']

bnd1 = torch.cartesian_prod(x, t).float()
bndval1 = torch.tensor(usol).reshape(-1,1)

id_f = np.random.choice(len(bnd1), 2000, replace=False)

bnd1 = bnd1[id_f]
bndval1 = bndval1[id_f]


bconds = [[bnd1, bndval1, 'dirichlet']]

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

parameters = {'lam1': 2., 'lam2': 0.2} # true parameters: lam1 = 1, lam2 = -0.01*pi

parameter_registr(model, parameters)

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
            'coeff': model.lam1,
            'u*du/dx': [[None], [0]],
            'pow': [1, 1],
            'var': [0, 0]
        },
    '-mu*d2u/dx2':
        {
            'coeff': model.lam2,
            'd2u/dx2': [0, 0],
            'pow': 1,
            'var': 0
        }
}

equation = Equation(grid, burgers_eq, bconds).set_strategy('autograd')


model = Solver(grid, equation, model, 'autograd').solve(lambda_bound=100, verbose=1, learning_rate=1e-4, tmax=25e3,
                                                            eps=1e-6, print_every=5000, use_cache=False, cache_dir='../cache/',
                                                            patience=3, optimizer_mode='Adam', inverse_parameters=parameters)
