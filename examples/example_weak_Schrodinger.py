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

domain = Domain()
domain.variable('x', [-5, 5], 41)
domain.variable('t', [0, np.pi/2], 41)

boundaries = Conditions()

## BOUNDARY AND INITIAL CONDITIONS
fun = lambda x: 2/np.cosh(x)

# u(x,0) = 2sech(x), v(x,0) = 0
x = domain.variable_dict['x']
boundaries.dirichlet({'x': [-5, 5], 't': 0}, value=fun(x), var=0)
boundaries.dirichlet({'x': [-5, 5], 't': 0}, value=0, var=1)


# u(-5,t) = u(5,t)
boundaries.periodic([{'x': -5, 't': [0, np.pi/2]}, {'x': 5, 't': [0, np.pi/2]}], var=0)

# v(-5,t) = v(5,t)
boundaries.periodic([{'x': -5, 't': [0, np.pi/2]}, {'x': 5, 't': [0, np.pi/2]}], var=1)


# du/dx (-5,t) = du/dx (5,t)
bop3_real = {
            'du/dx':
                {
                    'coeff': 1,
                    'du/dx': [0],
                    'pow': 1,
                    'var': 0
                }
}
boundaries.periodic([{'x': -5, 't': [0, np.pi/2]}, {'x': 5, 't': [0, np.pi/2]}], operator=bop3_real, var=0)

# dv/dx (-5,t) = dv/dx (5,t)
bop3_imag = {
            'dv/dx':
                {
                    'coeff': 1,
                    'dv/dx': [0],
                    'pow': 1,
                    'var': 1
                }
}

boundaries.periodic([{'x': -5, 't': [0, np.pi/2]}, {'x': 5, 't': [0, np.pi/2]}], operator=bop3_imag, var=1)

'''
schrodinger equation:
i * dh/dt + 1/2 * d2h/dx2 + abs(h)**2 * h = 0 
real part: 
du/dt + 1/2 * d2v/dx2 + (u**2 + v**2) * v
imag part:
dv/dt - 1/2 * d2u/dx2 - (u**2 + v**2) * u
u = var:0
v = var:1
'''

equation = Equation()

schrodinger_eq_real = {
    'du/dt':
        {
            'coeff': 1,
            'term': [1],
            'pow': 1,
            'var': 0
        },
    '1/2*d2v/dx2':
        {
            'coeff': 1 / 2,
            'term': [0, 0],
            'pow': 1,
            'var': 1
        },
    'v * u**2':
        {
            'coeff': 1,
            'term': [[None], [None]],
            'pow': [1, 2],
            'var': [1, 0]
        },
    'v**3':
        {
            'coeff': 1,
            'term': [None],
            'pow': 3,
            'var': 1
        }

}
schrodinger_eq_imag = {
    'dv/dt':
        {
            'coeff': 1,
            'term': [1],
            'pow': 1,
            'var': 1
        },
    '-1/2*d2u/dx2':
        {
            'coeff': - 1 / 2,
            'term': [0, 0],
            'pow': 1,
            'var': 0
        },
    '-u * v ** 2':
        {
            'coeff': -1,
            'term': [[None], [None]],
            'pow': [1, 2],
            'var': [0, 1]
        },
    '-u ** 3':
        {
            'coeff': -1,
            'term': [None],
            'pow': 3,
            'var': 0
        }

}

equation.add(schrodinger_eq_real)
equation.add(schrodinger_eq_imag)

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
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 2)
    )

def v(grid):
    return torch.cos(grid[:,0] + grid[:,1]) # torch.ones_like(grid[:,0]) + torch.cos(grid[:,0] + grid[:,1]) # for more accurate in more time

weak_form=[v]

model =  Model(net, domain, equation, boundaries)

model.compile("autograd", lambda_operator=1, lambda_bound=1, weak_form=weak_form)

img_dir=os.path.join(os.path.dirname( __file__ ), 'schroedinger_weak_img')

cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                    loss_window=10,
                                    no_improvement_patience=500,
                                    patience=10,
                                    randomize_parameter=1e-6,
                                    info_string_every=1)

cb_plots = plot.Plots(save_every=10, print_every=None, img_dir=img_dir)

optimizer = Optimizer('LBFGS', {'lr': 0.9})

model.train(optimizer, 1e5, save_model=False, callbacks=[cb_es, cb_plots])
