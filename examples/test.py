import torch
import numpy as np
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

#sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import adaptive_lambda, cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer

domain = Domain()

domain.variable('x', [0,1], 20)
domain.variable('t', [0,1], 20)

boundaries = Conditions()

bnd1 = domain.variable_dict['x']

boundaries.dirichlet({'x': [0, 1], 't': 0}, value=torch.sin(np.pi*bnd1))

bnd2 = domain.variable_dict['x']

boundaries.dirichlet({'x': [0, 1], 't': 1}, value=torch.sin(np.pi*bnd2))

boundaries.dirichlet({'x': 0, 't': [0, 1]}, value=0)

boundaries.dirichlet({'x': 1, 't': [0, 1]}, value=0)

equation = Equation()

wave_eq = {
    '4*d2u/dx2**1':
        {
            'coeff': 4,
            'd2u/dx2': [0, 0],
            'pow': 1
        },
    '-d2u/dt2**1':
        {
            'coeff': -1,
            'd2u/dt2': [1,1],
            'pow':1
        }
}

equation.add(wave_eq)

net = torch.nn.Sequential(
    torch.nn.Linear(2, 256),
    torch.nn.Tanh(),
    torch.nn.Linear(256,256),
    torch.nn.Tanh(),
    torch.nn.Linear(256, 256),
    torch.nn.Tanh(),
    torch.nn.Linear(256, 256),
    torch.nn.Tanh(),
    torch.nn.Linear(256, 1))

model = Model(net, domain, equation, boundaries)

model.compile('autograd', 1, 10)

cache = cache.Cache(cache_verbose=True)

es = early_stopping.EarlyStopping(eps=1e-5, loss_window=100)

plots = plot.Plots(print_every=None, save_every=100, title='wave_eq')

optimizer = Optimizer('Adam', {'lr': 1e-3})

model.train(optimizer, 10001, save_model=True, callbacks=[cache, plots])