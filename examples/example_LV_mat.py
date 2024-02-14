# Lotka-Volterra equations also known as predator-prey equations, describe the variation in populations
# of two species which interact via predation.
# For example, wolves (predators) and deer (prey). This is a classical model to represent the dynamic of two populations.

# Let αlpha > 0, beta > 0, delta > 0 and gamma > 0 . The system is given by

# dx/dt = x(alpha-beta*y)
# dy/dt = y(-delta+gamma*x)

# Where 'x' represents prey population and 'y' predators population. It’s a system of first-order ordinary differential equations.
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import time
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.models import mat_model


alpha = 20.
beta = 20.
delta = 20.
gamma = 20.
x0 = 4.
y0 = 2.
t0 = 0.
tmax = 2.

Nt = 400

solver_device('cpu')

domain = Domain()
domain.variable('t', [t0, tmax], Nt)

#initial conditions
boundaries = Conditions()
boundaries.dirichlet({'t': 0}, value=x0, var=0)
boundaries.dirichlet({'t': 0}, value=y0, var=1)

#equation system
# eq1: dx/dt = x(alpha-beta*y)
# eq2: dy/dt = y(-delta+gamma*x)

# x var: 0
# y var:1

equation = Equation()

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

net = mat_model(domain, equation)

model =  Model(net, domain, equation, boundaries)

model.compile("mat", lambda_operator=1, lambda_bound=100, derivative_points=3)

img_dir=os.path.join(os.path.dirname( __file__ ), 'img_Lotka_Volterra_mat')

start = time.time()

cb_cache = cache.Cache(cache_verbose=True, model_randomize_parameter=1e-5)

cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                    loss_window=100,
                                    no_improvement_patience=1000,
                                    patience=5,
                                    randomize_parameter=1e-5,
                                    info_string_every=100)

cb_plots = plot.Plots(save_every=100, print_every=None, img_dir=img_dir)

optimizer = Optimizer('LBFGS', {'lr': 1}, gamma=0.9, decay_every=400)

model.train(optimizer, 1e5, save_model=True, callbacks=[cb_cache, cb_es, cb_plots])

end = time.time()

def exact():
    # scipy.integrate solution of Lotka_Volterra equations and comparison with NN results

    def deriv(X, t, alpha, beta, delta, gamma):
        x, y = X
        dotx = x * (alpha - beta * y)
        doty = y * (-delta + gamma * x)
        return np.array([dotx, doty])

    t = np.linspace(0.,tmax, Nt+1)

    X0 = [x0, y0]
    res = integrate.odeint(deriv, X0, t, args = (alpha, beta, delta, gamma))
    x, y = res.T
    return np.array([x.reshape(-1), y.reshape(-1)])

u_exact = exact()

u_exact=torch.from_numpy(u_exact)

net = net.to('cpu')

error_rmse=torch.sqrt(torch.mean((u_exact-net)**2))

t = domain.variable_dict['t']

plt.figure()
plt.grid()
plt.plot(t, u_exact[0].detach().numpy().reshape(-1), '+', label = 'x_odeint')
plt.plot(t, u_exact[1].detach().numpy().reshape(-1), '*', label = "y_odeint")
plt.plot(t, net[0].detach().numpy().reshape(-1), label='x_tedeous')
plt.plot(t, net[1].detach().numpy().reshape(-1), label='y_tedeous')
plt.xlabel('Time, t')
plt.ylabel('Population')
plt.legend(loc='upper right')
plt.show()

