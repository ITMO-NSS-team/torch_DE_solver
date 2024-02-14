# this script perfom comparing between TEDEOuS and DeepXDE algorithms for burgers equation solution
# It's required to install DeepXDE and tensorflow.

import torch
import numpy as np
import time
import deepxde as dde
import pandas as pd
from scipy.integrate import quad
import sys
import os
import tensorflow as tf

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import cache, early_stopping
from tedeous.optimizers.optimizer import Optimizer


def solver_burgers(grid_res, cache_flag, optimizer, iterations):
    exp_dict_list = []
    start = time.time()
    mu = 0.01 / np.pi

    domain = Domain()
    domain.variable('x', [-1, 1], grid_res)
    domain.variable('t', [0, 1], grid_res)

    boundaries = Conditions()
    x = domain.variable_dict['x']
    boundaries.dirichlet({'x': [-1, 1], 't': 0}, value=-torch.sin(np.pi * x))

    boundaries.dirichlet({'x': -1, 't': [0, 1]}, value=0)

    boundaries.dirichlet({'x': 1, 't': [0, 1]}, value=0)

    equation = Equation()

    burgers_eq = {
        'du/dt**1':
            {
                'coeff': 1.,
                'du/dt': [1],
                'pow': 1,
                'var': 0
            },
        '+u*du/dx':
            {
                'coeff': 1,
                'u*du/dx': [[None], [0]],
                'pow': [1, 1],
                'var': [0, 0]
            },
        '-mu*d2u/dx2':
            {
                'coeff': -mu,
                'd2u/dx2': [0, 0],
                'pow': 1,
                'var': 0
            }
    }

    equation.add(burgers_eq)

    net = torch.nn.Sequential(
        torch.nn.Linear(2, 20),
        torch.nn.Tanh(),
        torch.nn.Linear(20, 20),
        torch.nn.Tanh(),
        torch.nn.Linear(20, 20),
        torch.nn.Tanh(),
        torch.nn.Linear(20, 20),
        torch.nn.Tanh(),
        torch.nn.Linear(20, 1)
    )

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=1, lambda_bound=1)

    cb_cache = cache.Cache(cache_verbose=False, model_randomize_parameter=1e-5)

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                        loss_window=100,
                                        no_improvement_patience=100,
                                        patience=2,
                                        randomize_parameter=1e-5,
                                        verbose=False)
    if cache:
        callbacks = [cb_cache, cb_es]
    else:
        callbacks = [cb_es]
    
    if type(optimizer) is list:
        for mode in optimizer:
            optim = Optimizer(mode, {'lr': 1e-3})
            model.train(optim, iterations, save_model=cache_flag, callbacks=callbacks)
    else:
        optim = Optimizer(optimizer, {'lr': 1e-3})
        model.train(optim, iterations, save_model=cache_flag, callbacks=callbacks)

    end = time.time()
    time_part = end - start

    x1 = torch.from_numpy(np.linspace(-1, 1, grid_res + 1))
    t1 = torch.from_numpy(np.linspace(0, 1, grid_res + 1))
    grid1 = torch.cartesian_prod(x1, t1).float()

    u_exact = exact(grid1)
    error_rmse = torch.sqrt(torch.mean((u_exact - net(grid1)) ** 2))
    exp_dict_list.append({'grid_res': grid_res, 'time': time_part, 'RMSE': error_rmse.detach().numpy(),
                           'type': 'solver_burgers', 'cache': cache_flag})

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))

    return exp_dict_list


def deepxde_burgers(grid_res, optimizer, iterations):
    exp_dict_list = []
    start = time.time()
    domain = (grid_res + 1) ** 2 - (grid_res + 1) * 4
    bound = (grid_res + 1) * 2
    init = (grid_res + 1) * 1

    def pde(x, y):
        dy_x = dde.grad.jacobian(y, x, i=0, j=0)
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t + y * dy_x - 0.01 / np.pi * dy_xx

    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    ic = dde.icbc.IC(
        geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial
    )

    data = dde.data.TimePDE(
        geomtime, pde, [bc, ic], num_domain=domain, num_boundary=bound, num_initial=init
    )
    net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")

    model = dde.Model(data, net)
    if type(optimizer) is list:
        model.compile('adam', lr=1e-3)
        model.train(iterations=iterations)
        model.compile("L-BFGS")
        losshistory, train_state = model.train()
    else:
        model.compile('adam', lr=1e-3)
        model.train(iterations=iterations)

    end = time.time()
    time_part = end - start

    x = torch.from_numpy(np.linspace(-1, 1, grid_res + 1))
    t = torch.from_numpy(np.linspace(0, 1, grid_res + 1))
    h = (x[1] - x[0]).item()
    grid = torch.cartesian_prod(x, t).float()

    u_exact = exact(grid)
    error_rmse = torch.sqrt(torch.mean((u_exact - model.predict(grid)) ** 2))

    # end_loss = Solution(grid, equation, model, 'autograd').loss_evaluation(lambda_bound=1)
    exp_dict_list.append(
        {'grid_res': grid_res, 'time': time_part, 'RMSE': error_rmse.detach().numpy(), 'type': 'deepxde_burgers',
         'optimizer': len(optimizer)})

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))
    return exp_dict_list


def exact(grid):
    mu = 0.01 / np.pi

    def f(y):
        return np.exp(-np.cos(np.pi * y) / (2 * np.pi * mu))

    def integrand1(m, x, t):
        return np.sin(np.pi * (x - m)) * f(x - m) * np.exp(-m ** 2 / (4 * mu * t))

    def integrand2(m, x, t):
        return f(x - m) * np.exp(-m ** 2 / (4 * mu * t))

    def u(x, t):
        if t == 0:
            return -np.sin(np.pi * x)
        else:
            return -quad(integrand1, -np.inf, np.inf, args=(x, t))[0] / quad(integrand2, -np.inf, np.inf, args=(x, t))[
                0]

    solution = []
    for point in grid:
        solution.append(u(point[0].item(), point[1].item()))

    return torch.tensor(solution)


nruns = 10
###########################
exp_dict_list = []

cache_flag = False

for grid_res in range(10, 41, 10):
    for _ in range(nruns):
        exp_dict_list.append(solver_burgers(grid_res, cache_flag, 'Adam', 2000))

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)
df.to_csv('examples/benchmarking_data/solver_burgers_10_40_cache={}.csv'.format(str(cache_flag)))
###########################

###########################
exp_dict_list = []

for grid_res in range(10, 41, 10):
    for _ in range(nruns):
        exp_dict_list.append(deepxde_burgers(grid_res, 'Adam', 15000))

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)
df.to_csv('examples/benchmarking_data/deepxde_burgers_10_40_Adam.csv')
###########################

###########################
exp_dict_list = []

cache_flag = True

for grid_res in range(10, 41, 10):
    for _ in range(nruns):
        exp_dict_list.append(solver_burgers(grid_res, cache_flag, 'Adam', 2000))

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)
df.to_csv('examples/benchmarking_data/solver_burgers_10_40_cache={}.csv'.format(str(cache_flag)))
###########################

###########################
exp_dict_list = []

for grid_res in range(30, 41, 10):
    for _ in range(nruns):
        exp_dict_list.append(deepxde_burgers(grid_res, ['Adam', 'LBFGS'], 15000))

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)
df.to_csv('examples/benchmarking_data/deepxde_burgers_10_40_Adam_LBFGS.csv')