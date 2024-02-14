import torch
import math
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import adaptive_lambda, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device

solver_device('cuda')
grid_res = 30

domain = Domain()
domain.variable('t', [0, 2], grid_res)
domain.variable('x', [0, 5], grid_res)

boundaries = Conditions()

# Boundary conditions at x=0
bop1_u = {
                'du/dt':
                    {
                    'coeff': 1,
                    'term': [0],
                    'pow': 1,
                    'var': 0
                    }
            }
t = domain.variable_dict['t']
boundaries.operator({'t': [0, 2], 'x': 0}, operator=bop1_u, value=t * torch.sin(t))

# Boundary conditions at x=5
# u_t = t*sin(t)
bop2_u = {
        'du/dt':
            {
                'coeff': 1,
                'term': [0],
                'pow': 1,
                'var': 0
            }
    }

boundaries.operator({'t': [0, 2], 'x': 5}, operator=bop2_u, value=t * torch.sin(t))

# Boundary conditions at x=0
# p(0,t) = 0
boundaries.dirichlet({'t': [0, 2], 'x': 0}, value=0, var=1)

# Boundary conditions at x=5
# p(5,t) = 0
boundaries.dirichlet({'t': [0, 2], 'x': 5}, value=0, var=1)

 # Initial condition at t=0
x = domain.variable_dict['x']
boundaries.dirichlet({'t': 0, 'x': [0, 5]}, value=torch.sin((math.pi * x) / 5) + 1, var=0)

ro = 1
mu = 1

equation = Equation()

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

equation.add(NS_1)
equation.add(NS_2)


net = torch.nn.Sequential(
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

model = Model(net, domain, equation, boundaries)

model.compile('autograd', lambda_operator=1, lambda_bound=1000, tol=0.1)

cb_es = early_stopping.EarlyStopping(eps=1e-5,
                                     loss_window=100,
                                     no_improvement_patience=1000,
                                     patience=10,
                                     info_string_every=5000,
                                     randomize_parameter=1e-5)

img_dir = os.path.join(os.path.dirname(__file__), 'navier_stokes_img')

cb_plots = plot.Plots(save_every=5000, print_every=None, img_dir=img_dir)

optimizer = Optimizer('Adam', {'lr': 1e-5})

model.train(optimizer, 1e6, save_model=True, callbacks=[cb_es, cb_plots])
