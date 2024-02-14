# Lotka-Volterra equations also known as predator-prey equations, describe the variation in populations
# of two species which interact via predation.
# For example, wolves (predators) and deer (prey). This is a classical model to represent the dynamic of two populations.

# Let αlpha > 0, beta > 0, delta > 0 and gamma > 0 . The system is given by

# dx/dt = x(alpha-beta*y)
# dy/dt = y(-delta+gamma*x)

# Where 'x' represents prey population and 'y' predators population. It’s a system of first-order ordinary differential equations.
import torch
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import early_stopping, plot, cache
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device

alpha = 20.
beta = 20.
delta = 20.
gamma = 20.
x0 = 4.
y0 = 2.
t0 = 0.
tmax = 1.

solver_device('cuda')

domain = Domain()
domain.variable('t', [t0, tmax], 110)
t = domain.variable_dict['t']
h = (t[1]-t[0]).item()

boundaries = Conditions()
#initial conditions
boundaries.dirichlet({'t': 0}, value=x0, var=0)
boundaries.dirichlet({'t': 0}, value=y0, var=1)

equation = Equation()

#equation system
# eq1: dx/dt = x(alpha-beta*y)
# eq2: dy/dt = y(-delta+gamma*x)

# x var: 0
# y var:1

eq1 = {
    'dx/dt':{
        'coeff': 1,
        'term': [0],
        'pow': 1,
        'var': [0]
    },
    '-x*alpha':{
        'coeff': -alpha,
        'term': [None],
        'pow': 1,
        'var': [0]
    },
    '+beta*x*y':{
        'coeff': beta,
        'term': [[None], [None]],
        'pow': [1, 1],
        'var': [0, 1]
    }
}

eq2 = {
    'dy/dt':{
        'coeff': 1,
        'term': [0],
        'pow': 1,
        'var': [1]
    },
    '+y*delta':{
        'coeff': delta,
        'term': [None],
        'pow': 1,
        'var': [1]
    },
    '-gamma*x*y':{
        'coeff': -gamma,
        'term': [[None], [None]],
        'pow': [1, 1],
        'var': [0, 1]
    }
}

equation.add(eq1)
equation.add(eq2)

net = torch.nn.Sequential(
        torch.nn.Linear(1, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 2)
    )

def v(grid):
    return (0.5+0.5*torch.sin(grid[:,0]))*(2/h)**(0.5)/10
weak_form = [v]

model =  Model(net, domain, equation, boundaries)

model.compile("NN", lambda_operator=1, lambda_bound=100, h=h, weak_form=weak_form)

cb_es = early_stopping.EarlyStopping(eps=1e-6, no_improvement_patience=500, info_string_every=500)

cb_cache = cache.Cache(cache_verbose=True, model_randomize_parameter=1e-5)

img_dir=os.path.join(os.path.dirname( __file__ ), 'img_weak_Lotka_Volterra')

cb_plots = plot.Plots(save_every=1000, print_every=None, img_dir=img_dir)

optimizer = Optimizer('Adam', {'lr': 1e-4})

callbacks = [cb_es, cb_cache, cb_plots]

model.train(optimizer, 5e6, save_model=False, callbacks=callbacks)
