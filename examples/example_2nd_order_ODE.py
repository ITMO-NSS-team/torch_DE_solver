import torch
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import adaptive_lambda, cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device


solver_device('cuda')

domain = Domain()

domain.variable('t', [0, 0.25], 100, dtype='float32')

boundaries = Conditions()

boundaries.dirichlet({'t': 0}, value=-1)

bop2 = {
        'dy/dt':
            { 'coeff': 1,
              'dy/dt': [0],
              'pow': 1
            }
        }

boundaries.operator({'t': 0}, bop2, value=2.17628)

grid = domain.variable_dict['t'].reshape(-1,1)

def t_func(grid):
    return grid

# y_tt - 10*y_t + 9*y - 5*t = 0

equation = Equation()

ode = {
        'd2y/dt2':
            {
              'coeff': 1,
              'd2y/dt2': [0,0],
              'pow': 1
            },
        '10*dy/dt':
            {
              'coeff': -10,
              'dy/dt': [0],
              'pow': 1
            },
        '9*y':
            {
              'coeff': 9,
              'dy/dt': [None],
              'pow': 1
            },
        '-5*t':
            {
              'coeff': -5*t_func(grid),
              'dy/dt': [None],
              'pow': 0
            }
      }

equation.add(ode)

net = torch.nn.Sequential(
        torch.nn.Linear(1, 50),
        torch.nn.Tanh(),
        torch.nn.Linear(50, 50),
        torch.nn.Tanh(),
        torch.nn.Linear(50, 50),
        torch.nn.Tanh(),
        torch.nn.Linear(50, 50),
        torch.nn.Tanh(),
        torch.nn.Linear(50, 1)
        )

model = Model(net, domain, equation, boundaries)

model.compile('autograd', lambda_operator=1, lambda_bound=100)

cb_es = early_stopping.EarlyStopping(eps=1e-8,
                                     loss_window=100,
                                     no_improvement_patience=1000,
                                     patience=5,
                                     info_string_every=2000,
                                     randomize_parameter=1e-5)

img_dir=os.path.join(os.path.dirname( __file__ ), 'ODE_img')

cb_cache = cache.Cache(cache_verbose=True, model_randomize_parameter=1e-5)

cb_plots = plot.Plots(save_every=2000, print_every=None, img_dir=img_dir)

cb_lambda = adaptive_lambda.AdaptiveLambda(sampling_N=2)

optimizer = Optimizer('Adam', {'lr': 1e-4}) # gamma=None, lr_decay=1000,

model.train(optimizer, 1e5, save_model=True, callbacks=[cb_cache, cb_es, cb_plots, cb_lambda])

def sln(t):
    return 50/81 + (5/9) * t + (31/81) * torch.exp(9*t) - 2 * torch.exp(t)

plt.plot(grid.detach().cpu().numpy(), sln(grid).detach().cpu().numpy(), label='Exact')
plt.plot(grid.detach().cpu().numpy(), net(grid).detach().cpu().numpy(), '--', label='Predicted')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper right')
plt.show()