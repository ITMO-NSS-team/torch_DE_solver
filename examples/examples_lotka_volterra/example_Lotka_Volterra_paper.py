# Lotka-Volterra equations also known as predator-prey equations, describe the variation in populations
# of two species which interact via predation.
# For example, wolves (predators) and deer (prey).
# This is a classical model to represent the dynamic of two populations.

# Let αlpha > 0, beta > 0, delta > 0 and gamma > 0 . The system is given by

# dx/dt = x(alpha-beta*y)
# dy/dt = y(-delta+gamma*x)

# Where 'x' represents prey population and 'y' predators population.
# It’s a system of first-order ordinary differential equations.
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import time
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device, check_device, device_type
from tedeous.models import Fourier_embedding

alpha = 20.
beta = 20.
delta = 20.
gamma = 20.
x0 = 4.
y0 = 2.
t0 = 0.
tmax = 1

from copy import deepcopy


def Lotka_experiment(grid_res, CACHE):
    exp_dict_list = []
    solver_device('gpu')

    net = torch.nn.Sequential(
        torch.nn.Linear(1, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 2)
    )

    domain = Domain()
    domain.variable('t', [0, 1], grid_res)

    boundaries = Conditions()
    # initial conditions
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

    model = Model(net, domain, equation, boundaries, batch_size=64)

    model.compile("autograd", lambda_operator=1, lambda_bound=100)

    img_dir = os.path.join(os.path.dirname(__file__), 'img_Lotka_Volterra_paper')

    start = time.time()

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=1000,
                                         no_improvement_patience=500,
                                         patience=3,
                                         info_string_every=100,
                                         randomize_parameter=1e-5)

    cb_plots = plot.Plots(save_every=1000, print_every=None, img_dir=img_dir)

    # cb_cache = cache.Cache(cache_verbose=True, model_randomize_parameter=1e-5)

    optimizer = Optimizer('Adam', {'lr': 1e-4})

    model.train(optimizer, 2e5, save_model=True, callbacks=[cb_es, cb_plots])

    end = time.time()

    rmse_t_grid = np.linspace(0, 1, grid_res + 1)

    rmse_t = torch.from_numpy(rmse_t_grid)

    rmse_grid = rmse_t.reshape(-1, 1).float()

    def exact():
        # scipy.integrate solution of Lotka_Volterra equations and comparison with NN results

        def deriv(X, t, alpha, beta, delta, gamma):
            x, y = X
            dotx = x * (alpha - beta * y)
            doty = y * (-delta + gamma * x)
            return np.array([dotx, doty])

        t = np.linspace(0, 1, grid_res + 1)

        X0 = [x0, y0]
        res = integrate.odeint(deriv, X0, t, args=(alpha, beta, delta, gamma))
        x, y = res.T
        return np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))

    u_exact = exact()

    u_exact = torch.from_numpy(u_exact)

    error_rmse = torch.sqrt(torch.mean((u_exact - net(rmse_grid)) ** 2, 0).sum())

    exp_dict_list.append(
        {'grid_res': grid_res, 'time': end - start, 'RMSE': error_rmse.detach().numpy(), 'type': 'Lotka_eqn',
         'cache': CACHE})

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))

    # t = domain.variable_dict['t']
    grid = domain.build('NN')

    t = np.linspace(0, 1, grid_res + 1)

    plt.figure()
    plt.grid()
    plt.title("odeint and NN methods comparing")
    plt.plot(t, u_exact[:, 0].detach().numpy().reshape(-1), '+', label='preys_odeint')
    plt.plot(t, u_exact[:, 1].detach().numpy().reshape(-1), '*', label="predators_odeint")
    plt.plot(grid.cpu(), net(grid.cpu())[:, 0].detach().numpy().reshape(-1), label='preys_NN')
    plt.plot(grid.cpu(), net(grid.cpu())[:, 1].detach().numpy().reshape(-1), label='predators_NN')
    plt.xlabel('Time t, [days]')
    plt.ylabel('Population')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(img_dir, 'compare_{}.png'.format(grid_res)))

    return exp_dict_list


nruns = 1

exp_dict_list = []

CACHE = False

for grid_res in range(60, 1001, 100):
    for _ in range(nruns):
        exp_dict_list.append(Lotka_experiment(grid_res, CACHE))

import pandas as pd

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)
df.boxplot(by='grid_res', column='time', fontsize=42, figsize=(20, 10))
df.boxplot(by='grid_res', column='RMSE', fontsize=42, figsize=(20, 10), showfliers=False)
df.to_csv('benchmarking_data/Lotka_experiment_50_90_cache={}.csv'.format(str(CACHE)))
