import os
import sys
import time

import numpy as np
import torch
from scipy.integrate import quad

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
current_file = os.path.abspath(os.path.dirname(__file__))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import early_stopping, save_model
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.eval import integration

import pandas as pd

solver_device('cuda')

mu = 0.01 / np.pi


def u(grid):
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


def u_net(net, x):
    net = net.to('cpu')
    x = x.to('cpu')
    return net(x).detach()


def l2_norm(net, x):
    x = x.to('cpu')
    net = net.to('cpu')
    predict = net(x).detach().cpu().reshape(-1)
    exact = u(x).detach().cpu().reshape(-1)
    l2_norm = torch.sqrt(sum((predict - exact) ** 2))
    return l2_norm.detach().cpu().numpy()


def l2_norm_mat(net, x):
    x = x.to('cpu')
    net = net.to('cpu')
    predict = net.detach().cpu().reshape(-1)
    exact = u(x).detach().cpu().reshape(-1)
    l2_norm = torch.sqrt(sum((predict - exact) ** 2))
    return l2_norm.detach().cpu().numpy()


def l2_norm_fourier(net, x):
    x = x.to(torch.device('cuda:0'))
    predict = net(x).detach().cpu().reshape(-1)
    exact = u(x).detach().cpu().reshape(-1)
    l2_norm = torch.sqrt(sum((predict - exact) ** 2))
    return l2_norm.detach().cpu().numpy()


def burgers1d_problem_formulation(grid_res):
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

    grid = domain.build('autograd')

    return grid, domain, equation, boundaries


def experiment_data_amount_burgers_1d_adam(grid_res, iter, exp_name='burgers1d_adam_5_starts',
                                                      save_plot=True):
    solver_device('cuda')
    exp_dict_list = []

    grid, domain, equation, boundaries = burgers1d_problem_formulation(grid_res)

    net = torch.nn.Sequential(
        torch.nn.Linear(2, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 1)
    )

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=1 / 2, lambda_bound=1 / 2)

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=100,
                                         no_improvement_patience=100,
                                         patience=2,
                                         randomize_parameter=1e-5,
                                         verbose=False,
                                         info_string_every=10)

    optim = Optimizer('Adam', {'lr': 1e-3})

    start = time.time()
    path_to_folder = os.path.join(current_file, "trajectories", "burgers", "adam_5_starts", f"adam_{iter}".format(iter))
    os.makedirs(path_to_folder, exist_ok=True)
    cb_sm = save_model.SaveModel(path_to_folder, every_step=500)
    model.train(optim, 5e3, save_model=False, callbacks=[cb_es, cb_sm], info_string_every=500)
    end = time.time()

    time_adam = end - start

    grid = domain.build('autograd')

    grid_test = torch.cartesian_prod(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100))

    u_exact_train = u(grid).reshape(-1)

    u_exact_test = u(grid_test).reshape(-1)

    error_adam_train = torch.sqrt(torch.mean((u_exact_train - net(grid).reshape(-1)) ** 2))

    error_adam_test = torch.sqrt(torch.mean((u_exact_test - net(grid_test).reshape(-1)) ** 2))

    loss_adam = model.solution_cls.evaluate()[0].detach().cpu().numpy()

    lu_f = model.solution_cls.operator.operator_compute()

    lu_f, gr = integration(lu_f, grid)

    lu_f_adam, _ = integration(lu_f, gr)

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE_adam {}= {}'.format(grid_res, error_adam_test))


    exp_dict = {'grid_res': grid_res,
                'error_adam_train': error_adam_train.item(),
                'error_adam_test': error_adam_test.item(),
                'loss_adam': loss_adam.item(),
                "lu_f_adam": lu_f_adam.item(),
                'time_adam': time_adam,
                'type': exp_name}
    
    exp_dict_list.append(exp_dict)

    return exp_dict_list


exp_dict_list = []
nruns = 5

for grid_res in range(80, 81, 10):
    for i in range(nruns):
        exp_dict_list.append(experiment_data_amount_burgers_1d_adam(grid_res, i))
        exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
