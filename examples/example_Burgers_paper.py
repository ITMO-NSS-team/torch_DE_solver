# this script perfom comparing between TEDEOuS and DeepXDE algorithms for burgers equation solution
# It's required to install DeepXDE and tensorflow.

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
import deepxde as dde
import pandas as pd
from scipy.integrate import quad
import sys
import os
import tensorflow as tf

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append('../')
sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.input_preprocessing import Equation
from tedeous.solver import Solver
from tedeous.solution import Solution


def solver_burgers(grid_res, cache, optimizer, iterations):
    exp_dict_list = []
    start = time.time()
    mu = 0.01 / np.pi
    x = torch.from_numpy(np.linspace(-1, 1, grid_res + 1))
    t = torch.from_numpy(np.linspace(0, 1, grid_res + 1))
    h = (x[1] - x[0]).item()
    grid = torch.cartesian_prod(x, t).float()

    ##initial cond
    bnd1 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()
    bndval1 = -torch.sin(np.pi * bnd1[:, 0])

    ##boundary cond
    bnd2 = torch.cartesian_prod(torch.from_numpy(np.array([-1.], dtype=np.float64)), t).float()
    bndval2 = torch.zeros_like(bnd2[:, 0])

    ##boundary cond
    bnd3 = torch.cartesian_prod(torch.from_numpy(np.array([1.], dtype=np.float64)), t).float()
    bndval3 = torch.zeros_like(bnd3[:, 0])

    bconds = [[bnd1, bndval1, 'dirichlet'],
              [bnd2, bndval2, 'dirichlet'],
              [bnd3, bndval3, 'dirichlet']]

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

    model = torch.nn.Sequential(
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

    equation = Equation(grid, burgers_eq, bconds).set_strategy('autograd')
    if type(optimizer) is list:
        for mode in optimizer:
            model = Solver(grid, equation, model, 'autograd').solve(lambda_bound=1, verbose=0, learning_rate=1e-3,
                                                                      eps=1e-6, tmin=10, tmax=iterations,
                                                                      use_cache=cache, cache_dir='../cache/',
                                                                      patience=2,
                                                                      save_always=cache, no_improvement_patience=100,
                                                                      optimizer_mode=mode)
    else:
        model = Solver(grid, equation, model, 'autograd').solve(lambda_bound=1, verbose=0, learning_rate=1e-3,
                                                                  eps=1e-6, tmin=10, tmax=iterations, use_cache=cache,
                                                                  cache_dir='../cache/', patience=2,
                                                                  save_always=cache, no_improvement_patience=100,
                                                                  optimizer_mode='Adam')
    end = time.time()
    time_part = end - start

    x1 = torch.from_numpy(np.linspace(-1, 1, grid_res + 1))
    t1 = torch.from_numpy(np.linspace(0, 1, grid_res + 1))
    grid1 = torch.cartesian_prod(x1, t1).float()

    u_exact = exact(grid1)
    error_rmse = torch.sqrt(torch.mean((u_exact - model(grid1)) ** 2))
    end_loss, _ = Solution(grid = grid, equal_cls = equation, model = model,
                        mode = 'autograd', weak_form=None, lambda_operator=1, lambda_bound=1).evaluate()
    exp_dict_list.append({'grid_res': grid_res, 'time': time_part, 'RMSE': error_rmse.detach().numpy(),
                          'loss': end_loss.detach().numpy(), 'type': 'solver_burgers', 'cache': cache})

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))
    print('loss {}= {}'.format(grid_res, end_loss))

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

cache = False

for grid_res in range(10, 41, 10):
    for _ in range(nruns):
        exp_dict_list.append(solver_burgers(grid_res, cache, 'Adam', 2000))

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)
df.to_csv('examples/benchmarking_data/solver_burgers_10_40_cache={}.csv'.format(str(cache)))
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

cache = True

for grid_res in range(10, 41, 10):
    for _ in range(nruns):
        exp_dict_list.append(solver_burgers(grid_res, cache, 'Adam', 2000))

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)
df.to_csv('examples/benchmarking_data/solver_burgers_10_40_cache={}.csv'.format(str(cache)))
###########################

###########################
exp_dict_list = []

for grid_res in range(30, 41, 10):
    for _ in range(nruns):
        exp_dict_list.append(deepxde_burgers(grid_res, ['Adam', 'LBFGS'], 15000))

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)
df.to_csv('examples/benchmarking_data/deepxde_burgers_10_40_Adam_LBFGS.csv')