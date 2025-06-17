import torch
import numpy as np
import os
import sys
import time
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model

from tedeous.callbacks import early_stopping, plot, cache
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.models import mat_model

solver_device('gpu')

x_min, x_max = 0, 1
t_max = 1


def exact_func_1(grid):
    x, t = grid[:, 0], grid[:, 1]
    sln = torch.cos(2 * np.pi * t) * torch.sin(np.pi * x)
    return sln


def exact_func_2(grid, a=4):
    x, t = grid[:, 0], grid[:, 1]
    sln = torch.sin(np.pi * x) * torch.cos(2 * np.pi * t) + 0.5 * \
          torch.sin(a * np.pi * x) * torch.cos(2 * a * np.pi * t)
    return sln


def wave_1d_basic_problem_formulation(grid_res):
    domain = Domain()
    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('t', [0, t_max], grid_res)

    # x = domain.variable_dict['x']
    # t = domain.variable_dict['t']

    grid = domain.build('autograd')

    boundaries = Conditions()

    # Initial conditions ###############################################################################################

    # init_func = torch.sin(torch.pi * x) * torch.sin(4 * torch.pi * x) / 2

    # u(x, 0) = f_init(x, 0)
    boundaries.dirichlet({'x': [x_min, x_max], 't': 0}, value=exact_func_1)

    # u_t(x, 0) = 0
    bop = {
        'du/dt':
            {
                'coeff': 1,
                'term': [1],
                'pow': 1,
                'var': 0
            }
    }
    boundaries.operator({'x': [x_min, x_max], 't': 0}, operator=bop, value=0)

    # Boundary conditions ##############################################################################################

    # bnd_func = torch.sin(torch.pi * x) * torch.cos(2 * torch.pi * t) + \
    #            torch.sin(4 * torch.pi * x) / 2 * torch.cos(8 * torch.pi * t)

    # u(0, t) = f_bnd(x, t)
    boundaries.dirichlet({'x': x_min, 't': [0, t_max]}, value=exact_func_1)

    # u(1, t) = f_bnd(x, t)
    boundaries.dirichlet({'x': x_max, 't': [0, t_max]}, value=exact_func_1)

    equation = Equation()

    # Operator: d2u/dt2 - 4 * d2u/dx2 = 0

    wave_eq = {
        'd2u/dt2**1':
            {
                'coeff': 1,
                'd2u/dt2': [1, 1],
                'pow': 1
            },
        '-C*d2u/dx2**1':
            {
                'coeff': -4,
                'd2u/dx2': [0, 0],
                'pow': 1
            }
    }

    equation.add(wave_eq)

    return grid, domain, equation, boundaries


def experiment_data_amount_wave_1d_pso_adam_lbfgs_nncg(grid_res, exp_name='wave_1d_pso_adam_lbfgs_nncg'):
    exp_dict_list = []

    neurons = 100

    grid, domain, equation, boundaries = wave_1d_basic_problem_formulation(grid_res)
    grid_test = torch.cartesian_prod(torch.linspace(x_min, x_max, 100),
                                     torch.linspace(0, t_max, 100))

    u_exact_train = exact_func_1(grid).reshape(-1, 1)
    u_exact_test = exact_func_1(grid_test).reshape(-1, 1)

    net = torch.nn.Sequential(
        torch.nn.Linear(2, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, 1)
    )

    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=1, lambda_bound=100)

    # PSO/CSO optimizer

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=100,
                                         no_improvement_patience=500,
                                         patience=3,
                                         randomize_parameter=1e-5,
                                         info_string_every=10)

    optim = Optimizer('PSO', {'lr': 1e-3})

    start = time.time()
    model.train(optim, 100, callbacks=[cb_es])
    end = time.time()

    run_time_pso = end - start
    error_train_pso = torch.sqrt(torch.mean((u_exact_train - net(grid)) ** 2, dim=0))
    error_test_pso = torch.sqrt(torch.mean((u_exact_test - net(grid_test)) ** 2, dim=0))
    loss_pso = model.solution_cls.evaluate()[0].detach().cpu().numpy()

    print('Time taken PSO {}= {}'.format(grid_res, run_time_pso))
    print('RMSE {}= {}'.format(grid_res, error_test_pso))

    # Adam optimizer

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=100,
                                         no_improvement_patience=500,
                                         patience=3,
                                         randomize_parameter=1e-5,
                                         info_string_every=100)

    optim = Optimizer('Adam', {'lr': 1e-4})

    start = time.time()
    model.train(optim, 1000, callbacks=[cb_es])
    end = time.time()

    run_time_adam = end - start
    error_train_adam = torch.sqrt(torch.mean((u_exact_train - net(grid)) ** 2, dim=0))
    error_test_adam = torch.sqrt(torch.mean((u_exact_test - net(grid_test)) ** 2, dim=0))
    loss_adam = model.solution_cls.evaluate()[0].detach().cpu().numpy()

    print('Time taken adam {}= {}'.format(grid_res, run_time_adam))
    print('RMSE {}= {}'.format(grid_res, error_test_adam))

    # LBFGS optimizer

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=100,
                                         no_improvement_patience=500,
                                         patience=3,
                                         randomize_parameter=1e-5,
                                         info_string_every=10)

    optim = Optimizer('LBFGS', {'lr': 1e-1})

    start = time.time()
    model.train(optim, 100, callbacks=[cb_es])
    end = time.time()

    run_time_lbfgs = end - start
    error_train_lbfgs = torch.sqrt(torch.mean((u_exact_train - net(grid)) ** 2, dim=0))
    error_test_lbfgs = torch.sqrt(torch.mean((u_exact_test - net(grid_test)) ** 2, dim=0))
    loss_lbfgs = model.solution_cls.evaluate()[0].detach().cpu().numpy()

    print('Time taken LBFGS {}= {}'.format(grid_res, run_time_lbfgs))
    print('RMSE {}= {}'.format(grid_res, error_test_lbfgs))

    # NNCG optimizer

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=100,
                                         no_improvement_patience=500,
                                         patience=3,
                                         randomize_parameter=1e-5,
                                         info_string_every=1)

    optim = Optimizer('NNCG', {
        "mu": 1e-1,
        "lr": 0.5,
        "rank": 10,
        "line_search_fn": "armijo",
        "precond_update_frequency": 5,
        "eigencdecomp_shift_attepmt_count": 10,
        'cg_max_iters': 1000,
        "verbose": False
        }
    )

    start = time.time()
    model.train(optim, 50, callbacks=[cb_es])
    end = time.time()

    run_time_nncg = end - start
    error_train_nncg = torch.sqrt(torch.mean((u_exact_train - net(grid)) ** 2, dim=0))
    error_test_nncg = torch.sqrt(torch.mean((u_exact_test - net(grid_test)) ** 2, dim=0))
    loss_nncg = model.solution_cls.evaluate()[0].detach().cpu().numpy()

    print('Time taken NNCG {}= {}'.format(grid_res, run_time_nncg))
    print('RMSE {}= {}'.format(grid_res, error_test_nncg))

    # full experiment dict

    exp_dict = {'grid_res': grid_res,
                'error_train_pso': error_train_pso.item(),
                'error_test_pso': error_test_pso.item(),
                'error_train_adam': error_train_adam.item(),
                'error_test_adam': error_test_adam.item(),
                'error_train_LBFGS': error_train_lbfgs.item(),
                'error_test_LBFGS': error_test_lbfgs.item(),
                'error_train_NNCG': error_train_nncg.item(),
                'error_test_NNCG': error_test_nncg.item(),
                'loss_pso': loss_pso.item(),
                'loss_adam': loss_adam.item(),
                'loss_LBFGS': loss_lbfgs.item(),
                'loss_NNCG': loss_nncg.item(),
                'time_pso': run_time_pso,
                'time_adam': run_time_adam,
                'time_LBFGS': run_time_lbfgs,
                'time_NNCG': run_time_nncg,
                'type': exp_name}

    exp_dict_list.append(exp_dict)

    return exp_dict_list


if __name__ == '__main__':

    results_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__))), 'results_handmade_chain')

    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    nruns = 1

    exp_dict_list = []

    for grid_res in range(10, 101, 10):
        for _ in range(nruns):
            exp_dict_list.append(experiment_data_amount_wave_1d_pso_adam_lbfgs_nncg(grid_res))
            exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
            df = pd.DataFrame(exp_dict_list_flatten)
            df_path = os.path.join(results_dir, 'wave_1d_pso_adam_lbfgs_nncg_{}.csv'.format(grid_res))
            df.to_csv(df_path)
