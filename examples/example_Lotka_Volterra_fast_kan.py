# Lotka-Volterra equations also known as predator-prey equations, describe the variation in populations
# of two species which interact via predation.
# For example, wolves (predators) and deer (prey). This is a classical model to represent the dynamic of two populations.

# Let αlpha > 0, beta > 0, delta > 0 and gamma > 0 . The system is given by

# dx/dt = x(alpha-beta*y)
# dy/dt = y(-delta+gamma*x)

# Where 'x' represents prey population and 'y' predators population. It’s a system of first-order ordinary differential equations.
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import integrate
import time
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device, check_device, device_type

import fastkan

solver_device('сpu')

alpha = 20.
beta = 20.
delta = 20.
gamma = 20.
x0 = 4.
y0 = 2.
t0 = 0.
tmax = 1.
Nt = 300

domain = Domain()

domain.variable('t', [t0, tmax], Nt)

h = 0.0001

# initial conditions
boundaries = Conditions()
boundaries.dirichlet({'t': 0}, value=x0, var=0)
boundaries.dirichlet({'t': 0}, value=y0, var=1)

# equation system
# eq1: dx/dt = x(alpha-beta*y)
# eq2: dy/dt = y(-delta+gamma*x)

# x var: 0
# y var:1

equation = Equation()

eq1 = {
    'dx/dt': {
        'coeff': 1,
        'term': [0],
        'pow': 1,
        'var': [0]
    },
    '-x*alpha': {
        'coeff': -alpha,
        'term': [None],
        'pow': 1,
        'var': [0]
    },
    '+beta*x*y': {
        'coeff': beta,
        'term': [[None], [None]],
        'pow': [1, 1],
        'var': [0, 1]
    }
}

eq2 = {
    'dy/dt': {
        'coeff': 1,
        'term': [0],
        'pow': 1,
        'var': [1]
    },
    '+y*delta': {
        'coeff': delta,
        'term': [None],
        'pow': 1,
        'var': [1]
    },
    '-gamma*x*y': {
        'coeff': -gamma,
        'term': [[None], [None]],
        'pow': [1, 1],
        'var': [0, 1]
    }
}

equation.add(eq1)
equation.add(eq2)

net = fastkan.FastKAN(
    [1, 20, 20, 20, 20, 2],
    grid_min=-10.,
    grid_max=10.,
    num_grids=2,
    use_base_update=True,
    use_layernorm=False,
    base_activation=F.tanh,
    spline_weight_init_scale=0.1
)

model = Model(net, domain, equation, boundaries)

model.compile("NN", lambda_operator=1, lambda_bound=100, h=h)

img_dir = os.path.join(os.path.dirname(__file__), 'img_Lotka_Volterra_fast_kan')

start = time.time()

cb_cache = cache.Cache(cache_verbose=True, model_randomize_parameter=1e-5)

cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                     loss_window=100,
                                     no_improvement_patience=1000,
                                     patience=5,
                                     randomize_parameter=1e-5,
                                     info_string_every=10)

cb_plots = plot.Plots(save_every=1000, print_every=1000, img_dir=img_dir)

optimizer = Optimizer('Adam', {'lr': 5e-4})

model.train(optimizer, 5e6, save_model=True, callbacks=[cb_es, cb_cache, cb_plots])

end = time.time()

print('Time taken = {}'.format(end - start))


# scipy.integrate solution of Lotka_Volterra equations and comparison with NN results

def deriv(X, t, alpha, beta, delta, gamma):
    x, y = X
    dotx = x * (alpha - beta * y)
    doty = y * (-delta + gamma * x)
    return np.array([dotx, doty])


t = np.linspace(0., tmax, Nt)

X0 = [x0, y0]
res = integrate.odeint(deriv, X0, t, args=(alpha, beta, delta, gamma))
x, y = res.T

grid = domain.build('NN')

plt.figure()
plt.grid()
plt.title("odeint and NN methods comparing")
plt.plot(t, x, '+', label='preys_odeint')
plt.plot(t, y, '*', label="predators_odeint")
plt.plot(grid, net(grid)[:, 0].detach().numpy().reshape(-1), label='preys_NN')
plt.plot(grid, net(grid)[:, 1].detach().numpy().reshape(-1), label='predators_NN')
plt.xlabel('Time t, [days]')
plt.ylabel('Population')
plt.legend(loc='upper right')
plt.show()

plt.figure()
plt.grid()
plt.title('Phase plane: prey vs predators')
plt.plot(net(grid)[:, 0].detach().numpy().reshape(-1), net(grid)[:, 1].detach().numpy().reshape(-1), '-*', label='NN')
plt.plot(x, y, label='odeint')
plt.xlabel('preys')
plt.ylabel('predators')
plt.legend()
plt.show()