import torch
import numpy as np
import os
import time
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import early_stopping, plot, cache
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device

solver_device('cpu')

p_l = 1
v_l = 0
Ro_l = 1
gam_l = 1.4

p_r = 0.1
v_r = 0
Ro_r = 0.125
gam_r = 1.4

x0 = 0.5
h = 0.05

domain = Domain()
domain.variable('x', [0, 1], 15)
domain.variable('t', [0, 0.2], 15)
x = domain.variable_dict['x']
h = x[1]-x[0]

## BOUNDARY AND INITIAL CONDITIONS
# p:0, v:1, Ro:2

def u0(x,x0):
    if x>x0:
        return [p_r, v_r, Ro_r]
    else:
        return [p_l, v_l, Ro_l]

boundaries = Conditions()

# Initial conditions at t=0
u_init0 = np.zeros(x.shape[0])
u_init1 = np.zeros(x.shape[0])
u_init2 = np.zeros(x.shape[0])
j=0
for i in x:
    u_init0[j] = u0(i, x0)[0]
    u_init1[j] = u0(i, x0)[1]
    u_init2[j] = u0(i, x0)[2]
    j +=1

bndval1_0 = torch.from_numpy(u_init0)
bndval1_1 = torch.from_numpy(u_init1)
bndval1_2 = torch.from_numpy(u_init2)

boundaries.dirichlet({'x': [0, 1], 't': 0}, value=bndval1_0, var=0)
boundaries.dirichlet({'x': [0, 1], 't': 0}, value=bndval1_1, var=1)
boundaries.dirichlet({'x': [0, 1], 't': 0}, value=bndval1_2, var=2)

#  Boundary conditions at x=0
boundaries.dirichlet({'x': 0, 't': [0, 0.2]}, value=p_l, var=0)
boundaries.dirichlet({'x': 0, 't': [0, 0.2]}, value=v_l, var=1)
boundaries.dirichlet({'x': 0, 't': [0, 0.2]}, value=Ro_l, var=2)

# Boundary conditions at x=1
boundaries.dirichlet({'x': 1, 't': [0, 0.2]}, value=p_r, var=0)
boundaries.dirichlet({'x': 1, 't': [0, 0.2]}, value=v_r, var=1)
boundaries.dirichlet({'x': 1, 't': [0, 0.2]}, value=Ro_r, var=2)

'''
gas dynamic system equations:
Eiler's equations system for Sod test in shock tube

'''

equation = Equation()

gas_eq1={
        'dro/dt':
        {
            'coeff': 1,
            'term': [1],
            'pow': 1,
            'var': 2
        },
        'v*dro/dx':
        {
            'coeff': 1,
            'term': [[None], [0]],
            'pow': [1, 1],
            'var': [1, 2]
        },
        'ro*dv/dx':
        {
            'coeff': 1,
            'term': [[None], [0]],
            'pow': [1, 1],
            'var': [2, 1]
        }
    }
gas_eq2 = {
        'ro*dv/dt':
        {
            'coeff': 1,
            'term': [[None], [1]],
            'pow': [1, 1],
            'var': [2, 1]
        },
        'ro*v*dv/dx':
        {
            'coeff': 1,
            'term': [[None],[None], [0]],
            'pow': [1, 1, 1],
            'var': [2, 1, 1]
        },
        'dp/dx':
        {
            'coeff': 1,
            'term': [0],
            'pow': 1,
            'var': 0
        }
    }
gas_eq3 =  {
        'dp/dt':
        {
            'coeff': 1,
            'term': [1],
            'pow': 1,
            'var': 0
        },
        'gam*p*dv/dx':
        {
            'coeff': gam_l,
            'term': [[None], [0]],
            'pow': [1, 1],
            'var': [0, 1]
        },
        'v*dp/dx':
        {
            'coeff': 1,
            'term': [[None], [0]],
            'pow': [1, 1],
            'var': [1, 0]
        }

    }

equation.add(gas_eq1)
equation.add(gas_eq2)
equation.add(gas_eq3)

net = torch.nn.Sequential(
        torch.nn.Linear(2, 150),
        torch.nn.Tanh(),
        torch.nn.Linear(150, 150),
        torch.nn.Tanh(),
        torch.nn.Linear(150, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 3)
        )

def v(grid):
    return torch.sin(grid[:,0])+torch.sin(2*grid[:,0])+grid[:,1]
weak_form = [v]

model =  Model(net, domain, equation, boundaries)

model.compile("NN", lambda_operator=1, lambda_bound=100, h=h, weak_form=weak_form)

img_dir=os.path.join(os.path.dirname( __file__ ), 'SOD_NN_weak_img')

cb_cache = cache.Cache(cache_verbose=False, model_randomize_parameter=1e-5)

cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                    loss_window=100,
                                    no_improvement_patience=500,
                                    patience=2,
                                    randomize_parameter=1e-5,
                                    abs_loss=0.0035,
                                    info_string_every=100
                                    )

cb_plots = plot.Plots(save_every=100, print_every=None, img_dir=img_dir)

optimizer = Optimizer('Adam', {'lr': 1e-3})

model.train(optimizer, 1e5, save_model=False, callbacks=[cb_es, cb_plots])
