import torch
import numpy as np
import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append('../')
sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.models import Fourier_embedding

solver_device('gpu')

# if the casual_loss is used the time parameter must be
# at the first place in the grid

domain = Domain()

domain.variable('t', [0, 1], 51, dtype='float32')
domain.variable('x', [-1, 1], 51, dtype='float32')

boundaries = Conditions()

# Initial conditions at t=0
x = domain.variable_dict['x']

value = x**2*torch.cos(np.pi*x)

boundaries.dirichlet({'x': [-1, 1], 't': 0}, value=value)

    
# Initial conditions at t=1
boundaries.periodic([{'x': -1, 't': [0, 1]}, {'x': 1, 't': [0, 1]}])

bop3= {
        'du/dx':
            {
                'coeff': 1,
                'du/dx': [1],
                'pow': 1,
                'var': 0
            }
}

boundaries.periodic([{'x': -1, 't': [0, 1]}, {'x': 1, 't': [0, 1]}], operator=bop3)

equation = Equation()

AC = {
    '1*du/dt**1':
        {
            'coeff': 1,
            'du/dt': [0],
            'pow': 1,
            'var': 0
        },
    '-0.0001*d2u/dx2**1':
        {
            'coeff': -0.0001,
            'd2u/dx2': [1,1],
            'pow': 1,
            'var': 0
        },
    '+5u**3':
        {
            'coeff': 5,
            'u': [None],
            'pow': 3,
            'var': 0
        },
    '-5u**1':
        {
            'coeff': -5,
            'u': [None],
            'pow': 1,
            'var': 0
        }
}

equation.add(AC)

FFL = Fourier_embedding(L=[None, 2], M=[None, 10])

out = FFL.out_features

net = torch.nn.Sequential(
    FFL,
    torch.nn.Linear(out, 128),
    torch.nn.Tanh(),
    torch.nn.Linear(128,128),
    torch.nn.Tanh(),
    torch.nn.Linear(128,128),
    torch.nn.Tanh(),
    torch.nn.Linear(128,1)
)
    
model = Model(net, domain, equation, boundaries)

model.compile('autograd', lambda_operator=1, lambda_bound=100, tol=10)

img_dir = os.path.join(os.path.dirname( __file__ ), 'AC_eq_img')

cb_cache = cache.Cache(cache_verbose=False, model_randomize_parameter=1e-5)

cb_es = early_stopping.EarlyStopping(eps=1e-7,
                                     loss_window=100,
                                     no_improvement_patience=1000,
                                     patience=5,
                                     abs_loss=1e-5,
                                     info_string_every=1000,
                                     randomize_parameter=1e-5)

cb_plots = plot.Plots(save_every=1000, print_every=None, img_dir=img_dir)

optimizer = Optimizer('Adam', {'lr': 1e-3}, gamma=0.9, decay_every=1000)

model.train(optimizer, 1e5, save_model=True, callbacks=[cb_cache, cb_es, cb_plots])
