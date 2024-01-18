import torch
import numpy as np
import os
import sys


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device

solver_device('cpu')

m = 0.2
L = 1
Q = -0.1
Sq = 1
mu_water = 0.89e-3
mu_o = 4.62e-3
Swi0 = 0.
Sk = 1.
t_end = 1.

domain = Domain()

domain.variable('x', [0, 1], 21, dtype='float32')
domain.variable('t', [0, 1], 21, dtype='float32')

boundaries = Conditions()

##initial cond
boundaries.dirichlet({'x': [0, 1], 't': 0}, value=Swi0)

##boundary cond
boundaries.dirichlet({'x': 0, 't': [0, 1]}, value=Sk)

net = torch.nn.Sequential(
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
    return (1-net(x))**2

def k_water(x):
    return (net(x))**2

def dk_water(x):
    return 2*net(x)

def dk_oil(x):
    return -2*(1-net(x))

def df(x):
    return (dk_water(x)*(k_water(x)+mu_water/mu_o*k_oil(x))-
            k_water(x)*(dk_water(x)+mu_water/mu_o*dk_oil(x)))/(k_water(x)+mu_water/mu_o*k_oil(x))**2

def coef_model(x):
    return -Q/Sq*df(x)

equation = Equation()

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

equation.add(buckley_eq)

model = Model(net, domain, equation, boundaries)

model.compile('autograd', lambda_operator=1, lambda_bound=1)

img_dir=os.path.join(os.path.dirname( __file__ ), 'Buckley_NN_img')


cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                     loss_window=100,
                                     no_improvement_patience=500,
                                     patience=5,
                                     abs_loss=1e-5,
                                     randomize_parameter=1e-5,
                                     info_string_every=500)

cb_plots = plot.Plots(save_every=500, print_every=None, img_dir=img_dir)

optimizer = Optimizer('Adam', {'lr': 1e-3})

model.train(optimizer, 1e6, save_model=False, callbacks=[cb_es, cb_plots])
                                    
                                     
                                    