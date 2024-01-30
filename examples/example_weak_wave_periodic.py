import torch
import numpy as np
import sys
import os
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))


from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device

solver_device('cpu')
# Grid
domain = Domain()

domain.variable('x', [0, 1], 50)
domain.variable('t', [0, 1], 50)

boundaries = Conditions()

# u(x,0)=1e4*sin^2(x(x-1)/10)
x = domain.variable_dict['x']
func_bnd1 = lambda x: 10 ** 4 * torch.sin((1/10) * x * (x-1)) ** 2
boundaries.dirichlet({'x': [0, 1], 't': 0}, value=func_bnd1(x))

func_bnd2 = lambda x: 10 ** 3 * torch.sin((1/10) * x * (x-1)) ** 2
# du/dx (x,0) = 1e3*sin^2(x(x-1)/10)
bop2 = {
        'du/dt':
            {
                'coeff': 1,
                'du/dt': [1],
                'pow': 1,
                'var': 0
            }
}
boundaries.operator({'x': [0, 1], 't': 0}, operator=bop2, value=func_bnd2(x))

# u(0,t) = u(1,t)
boundaries.periodic([{'x': 0, 't': [0, 1]}, {'x': 1, 't': [0, 1]}])

# du/dt(0,t) = du/dt(1,t)
bop4= {
        'du/dx':
            {
                'coeff': 1,
                'du/dx': [0],
                'pow': 1,
                'var': 0
            }
}
boundaries.periodic([{'x': 0, 't': [0, 1]}, {'x': 1, 't': [0, 1]}], operator=bop4)

equation = Equation()

# wave equation is d2u/dt2-(1/4)*d2u/dx2=0
C = 4
wave_eq = {
    'd2u/dt2':
        {
            'coeff': 1,
            'd2u/dt2': [1, 1],
            'pow': 1
        },
        '-1/C*d2u/dx2':
        {
            'coeff': -1/C,
            'd2u/dx2': [0, 0],
            'pow': 1
        }
}

equation.add(wave_eq)

net = torch.nn.Sequential(
         torch.nn.Linear(2, 100),
         torch.nn.Tanh(),
         torch.nn.Linear(100, 100),
         torch.nn.Tanh(),
         torch.nn.Linear(100, 100),
         torch.nn.Tanh(),
         torch.nn.Linear(100, 1))

def v(grid):
    return torch.cos(grid[:,0])+grid[:,1]
weak_form = [v]

start = time.time()

model =  Model(net, domain, equation, boundaries)

model.compile("NN", lambda_operator=1, lambda_bound=1000, h=0.01)

cb_es = early_stopping.EarlyStopping(eps=1e-6, no_improvement_patience=500, info_string_every=1000)

img_dir = os.path.join(os.path.dirname( __file__ ), 'wave_periodic_weak_img')

cb_plots = plot.Plots(save_every=100, print_every=None, img_dir=img_dir)

optimizer = Optimizer('Adam', {'lr': 1e-2})

model.train(optimizer, 1e5, save_model=True, callbacks=[cb_es, cb_plots])

end = time.time()
print('Time taken 10= ', end - start)
