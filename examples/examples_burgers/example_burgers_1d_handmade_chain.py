import torch
import os
import sys
import time
import numpy as np
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model

from tedeous.callbacks import early_stopping, plot, cache
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.utils import exact_solution_data

solver_device('gpu')

data_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../PINNacle_data/burgers1d.npy"))

mu = 0.01 / np.pi

x_min, x_max = -1, 1
t_max = 1


def burgers_1d_problem_formulation(grid_res):
    """
    Formulates the 1D Burgers' equation problem for neural network-based solving.
    
        This method sets up the domain, initial/boundary conditions, and the Burgers' equation
        itself, preparing the problem for approximation using a neural network. The configurations
        returned are essential for training a neural network to learn the solution of the differential
        equation. This setup is a crucial initial step in leveraging neural networks to solve
        the Burgers' equation.
    
        Args:
            grid_res: The resolution of the grid used for discretizing the domain.
    
        Returns:
            tuple: A tuple containing the grid, domain, equation, and boundaries.
                - grid: The discretized grid.
                - domain: The domain object.
                - equation: The equation object representing the Burgers' equation.
                - boundaries: The boundary conditions object.
    """
    domain = Domain()
    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('t', [0, t_max], grid_res)

    grid = domain.build('autograd')

    boundaries = Conditions()

    # Initial conditions ###############################################################################################

    # u(x, 0) = -sin(pi * x)
    boundaries.dirichlet({'x': [x_min, x_max], 't': 0}, value=lambda grid: -torch.sin(np.pi * grid[:, 0]))

    # Boundary conditions ##############################################################################################

    # u(x_min, t) = 0
    boundaries.dirichlet({'x': x_min, 't': [0, t_max]}, value=0)

    # u(x_max, t) = 0
    boundaries.dirichlet({'x': x_max, 't': [0, t_max]}, value=0)

    equation = Equation()

    # Operator: u_t + u * u_x - mu * u_xx = 0

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

    return grid, domain, equation, boundaries


def experiment_data_amount_burgers_1d_pso_adam_lbfgs_nncg(grid_res, exp_name='burgers_1d_pso_adam_lbfgs_nncg'):
    """
    Conducts an experiment to evaluate the performance of different optimization algorithms in solving the 1D Burgers equation using a neural network.
    
        This method trains a neural network to approximate the solution of the 1D Burgers equation,
        comparing the effectiveness of Particle Swarm Optimization (PSO), Adam, LBFGS, and NNCG optimizers.
        By varying the grid resolution, the method assesses how each optimizer performs in terms of training
        and testing error, loss, and runtime, providing insights into their suitability for solving
        differential equations with neural networks. The goal is to identify the most efficient and accurate
        optimizer for this specific problem.
    
        Args:
            grid_res (int): The resolution of the spatial grid used for training.
            exp_name (str, optional): The name of the experiment. Defaults to 'burgers_1d_pso_adam_lbfgs_nncg'.
    
        Returns:
            list: A list containing a dictionary with the results of the experiment. The dictionary includes:
                - grid_res (int): The grid resolution used.
                - error_train_pso (float): Training error for PSO.
                - error_test_pso (float): Testing error for PSO.
                - error_train_adam (float): Training error for Adam.
                - error_test_adam (float): Testing error for Adam.
                - error_train_LBFGS (float): Training error for LBFGS.
                - error_test_LBFGS (float): Testing error for LBFGS.
                - error_train_NNCG (float): Training error for NNCG.
                - error_test_NNCG (float): Testing error for NNCG.
                - loss_pso (float): Loss value for PSO.
                - loss_adam (float): Loss value for Adam.
                - loss_LBFGS (float): Loss value for LBFGS.
                - loss_NNCG (float): Loss value for NNCG.
                - time_pso (float): Runtime for PSO.
                - time_adam (float): Runtime for Adam.
                - time_LBFGS (float): Runtime for LBFGS.
                - time_NNCG (float): Runtime for NNCG.
                - type (str): The experiment name.
    """
    exp_dict_list = []

    pde_dim_in = 2
    pde_dim_out = 1

    neurons = 100

    grid, domain, equation, boundaries = burgers_1d_problem_formulation(grid_res)
    grid_test = torch.cartesian_prod(torch.linspace(x_min, x_max, 100),
                                     torch.linspace(0, t_max, 100))

    u_exact_train = exact_solution_data(grid, data_file, pde_dim_in, pde_dim_out, t_dim_flag=True).reshape(-1, 1)
    u_exact_test = exact_solution_data(grid_test, data_file, pde_dim_in, pde_dim_out, t_dim_flag=True).reshape(-1, 1)

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
    model.train(optim, 30, callbacks=[cb_es])
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
                                         info_string_every=10)

    optim = Optimizer('Adam', {'lr': 1e-4})

    start = time.time()
    model.train(optim, 100, callbacks=[cb_es])
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
        "chunk_size":8,
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
            exp_dict_list.append(experiment_data_amount_burgers_1d_pso_adam_lbfgs_nncg(grid_res))
            exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
            df = pd.DataFrame(exp_dict_list_flatten)
            df_path = os.path.join(results_dir, 'burgers_1d_pso_adam_lbfgs_nncg_{}.csv'.format(grid_res))
            df.to_csv(df_path)
