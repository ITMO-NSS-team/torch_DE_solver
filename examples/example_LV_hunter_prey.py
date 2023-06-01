from scipy import integrate
import time
import pandas as pd
import sys
import os
import torch
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append('../')
sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.input_preprocessing import Equation
from tedeous.solver import Solver
from tedeous.metrics import Solution
from tedeous.device import solver_device
from tedeous.models import Modified_MLP


x0 = 30.
y0 = 4.

solver_device('cpu')

t1 = np.linspace(0,20,1001)

t = torch.from_numpy(t1)

grid = t.reshape(-1, 1).float()

#initial conditions

bnd1_0 = torch.from_numpy(np.array([[0]], dtype=np.float64)).float()
bndval1_0 = torch.from_numpy(np.array([[x0]], dtype=np.float64))
bnd1_1 = torch.from_numpy(np.array([[0]], dtype=np.float64)).float()
bndval1_1  = torch.from_numpy(np.array([[y0]], dtype=np.float64))

bconds = [[bnd1_0, bndval1_0, 0, 'dirichlet'],
            [bnd1_1, bndval1_1, 1, 'dirichlet']]

#equation system
# eq1: dx/dt = x(alpha-beta*y)
# eq2: dy/dt = y(-delta+gamma*x)

# x var: 0
# y var:1

eq1 = {
    '1/x*dx/dt':{
        'coeff': 1,
        'term': [[None], [0]],
        'pow': [-1, 1],
        'var': [0, 0]
    },
    '-alpha':{
        'coeff': -0.55,
        'term': [None],
        'pow': 0,
        'var': [0]
    },
    '+beta*y':{
        'coeff': 0.028,
        'term': [None],
        'pow': 1,
        'var': [1]
    }
}

eq2 = {
    '1/y*dy/dt':{
        'coeff': 1,
        'term': [[None], [0]],
        'pow': [-1, 1],
        'var': [1, 1]
    },
    '+delta':{
        'coeff': 0.84,
        'term': [None],
        'pow': 0,
        'var': [1]
    },
    '-gamma*x':{
        'coeff': -0.026,
        'term': [None],
        'pow': 1,
        'var': [0]
    }
}

Lotka = [eq1, eq2]

model = Modified_MLP([512, 512, 512, 512, 2], [15], [7], device='cpu')

equation = Equation(grid, Lotka, bconds).set_strategy('autograd')

start = time.time()

model = Solver(grid, equation, model, 'autograd').solve(lambda_bound=10,
                                        verbose=True, learning_rate=1e-3, eps=1e-6, tmin=30000, tmax=5e6,
                                        use_cache=False, cache_dir='../cache/',cache_verbose=True,
                                        save_always=False, print_every=5000,
                                        patience=30,loss_oscillation_window=100, no_improvement_patience=10000,
                                        optimizer_mode='Adam', cache_model=None,
                                        step_plot_print=True, step_plot_save=False, tol=20, n_t=1001)

end = time.time()