import torch
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import time
from scipy.integrate import quad

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.models import mat_model, Fourier_embedding
from tedeous.callbacks import plot, early_stopping, adaptive_lambda
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.eval import integration

import pandas as pd

solver_device('cuda')

mu = 0.01 / np.pi


def soliton(x,t):
    """
    Calculates the solution of a soliton equation at a given spatial position and time.
    This function leverages a predefined mathematical formulation to estimate the state
    of the system, providing a numerical approximation of the soliton's behavior.
    
    Args:
        x (torch.Tensor): The spatial position at which to evaluate the soliton.
        t (torch.Tensor): The time at which to evaluate the soliton.
    
    Returns:
        torch.Tensor: The calculated value of the soliton at the given position and time.
    """
    E=np.exp(1)
    s=((18*torch.exp((1/125)*(t + 25*x))*(16*torch.exp(2*t) +
       1000*torch.exp((126*t)/125 + (4*x)/5) + 9*torch.exp(2*x) + 576*torch.exp(t + x) +
       90*torch.exp((124*t)/125 + (6*x)/5)))/(5*(40*torch.exp((126*t)/125) +
        18*torch.exp(t + x/5) + 9*torch.exp((6*x)/5) + 45*torch.exp(t/125 + x))**2))
    return s


def u(grid):
    """
    Calculates the soliton solution at given points to evaluate the neural network's approximation.
    
        This method iterates through a grid of (x, t) points, computes the soliton
        value at each point using the analytical `soliton` function, and returns a
        tensor of these solutions. This provides a ground truth for comparison
        against the neural network's predicted solution.
    
        Args:
            grid: A list of points, where each point is a tuple (x, t) representing
                the spatial and temporal coordinates.
    
        Returns:
            torch.Tensor: A tensor containing the soliton solutions for each point in the grid.
                These values serve as the analytical solution for evaluating the neural network.
    """
    solution = []
    for point in grid:
        x = point[0]
        t = point[1]
        s = soliton(x, t)
        solution.append(s)
    return torch.tensor(solution)


def u_net(net, x):
    """
    Applies a neural network to an input and detaches the result for CPU-based differential equation solving.
    
        This method ensures that both the network and the input are processed on the CPU,
        applies the network to the input, and then detaches the result
        from the computation graph. This is done to ensure compatibility and
        efficiency when solving differential equations, as it avoids potential
        GPU memory issues and ensures consistent performance across different systems.
    
        Args:
            net: The neural network to apply.
            x: The input to the neural network.
    
        Returns:
            The output of the neural network, detached from the computation graph.
    """
    net = net.to('cpu')
    x = x.to('cpu')
    return net(x).detach()


def l2_norm(net, x):
    """
    Calculates the L2 norm between the neural network's approximation and the analytical solution. This metric quantifies the accuracy of the neural network in solving the differential equation by measuring the difference between the predicted solution and the true solution.
    
        Args:
            net: The neural network model used to approximate the solution.
            x: The input tensor representing the domain over which the solution is approximated.
    
        Returns:
            numpy.ndarray: The L2 norm, a scalar value representing the overall error in the approximation, returned as a NumPy array.
    """
    x = x.to('cpu')
    net = net.to('cpu')
    predict = net(x).detach().cpu().reshape(-1)
    exact = u(x).detach().cpu().reshape(-1)
    l2_norm = torch.sqrt(sum((predict - exact) ** 2))
    return l2_norm.detach().cpu().numpy()


def l2_norm_mat(net, x):
    """
    Calculates the L2 norm between the neural network's prediction and the exact solution of the differential equation.
    
        This metric quantifies the accuracy of the neural network's approximation
        by measuring the difference between the predicted solution and the true solution.
        It helps assess how well the network has learned to solve the differential equation.
        
        Args:
            net (torch.nn.Module): The neural network model used to approximate the solution.
            x (torch.Tensor): The input tensor representing the spatial or temporal domain of the differential equation.
        
        Returns:
            numpy.ndarray: The L2 norm between the network's prediction and the exact solution, as a NumPy array.
    """
    x = x.to('cpu')
    net = net.to('cpu')
    predict = net.detach().cpu().reshape(-1)
    exact = u(x).detach().cpu().reshape(-1)
    l2_norm = torch.sqrt(sum((predict - exact) ** 2))
    return l2_norm.detach().cpu().numpy()


def l2_norm_fourier(net, x):
    """
    Computes the L2 norm in Fourier space to evaluate the accuracy of the neural network's solution compared to the exact solution.
    
        This metric quantifies the difference between the predicted and actual solutions in the frequency domain,
        providing insights into how well the network captures the underlying dynamics of the differential equation.
    
        Args:
            net: The neural network model used to approximate the solution.
            x: The input tensor representing the spatial or temporal domain of the differential equation.
    
        Returns:
            np.ndarray: The L2 norm between the prediction and the exact solution in Fourier space as a NumPy array.
    """
    x = x.to(torch.device('cuda:0'))
    predict = net(x).detach().cpu().reshape(-1)
    exact = u(x).detach().cpu().reshape(-1)
    l2_norm = torch.sqrt(sum((predict - exact) ** 2))
    return l2_norm.detach().cpu().numpy()


def kdv_problem_formulation(grid_res):
    """
    Applies initial conditions to the KdV equation at t=0. This step is crucial for defining the starting state of the system,
    allowing the neural network to learn the solution's evolution over time. By setting the initial state using a known soliton solution,
    we provide a well-defined starting point for the solver to approximate the solution of the KdV equation.
    
    Args:
        grid_res (int): Resolution of the grid.
    
    Returns:
        tuple: A tuple containing the grid, domain, equation, and boundaries.
    """
    domain = Domain()
    domain.variable('x', [-10, 10], grid_res)
    domain.variable('t', [0, 1], grid_res)

    boundaries = Conditions()

    # u(0,t) = u(1,t)
    boundaries.periodic([{'x': -10, 't': [0, 1]}, {'x': 10, 't': [0, 1]}])

    """
    Initial conditions at t=0
    """

    x = domain.variable_dict['x']

    boundaries.dirichlet({'x': [-10, 10], 't': 0}, value=soliton(x, torch.tensor([0])))

    equation = Equation()

    # operator is du/dt+6u*(du/dx)+d3u/dx3-sin(x)*cos(t)=0
    kdv = {
        '1*du/dt**1':
            {
                'coeff': 1,
                'du/dt': [1],
                'pow': 1,
                'var': 0
            },
        '6*u**1*du/dx**1':
            {
                'coeff': 6,
                'u*du/dx': [[None], [0]],
                'pow': [1, 1],
                'var': [0, 0]
            },
        'd3u/dx3**1':
            {
                'coeff': 1,
                'd3u/dx3': [0, 0, 0],
                'pow': 1,
                'var': 0
            }
    }

    equation.add(kdv)

    grid = domain.build('autograd')

    return grid, domain, equation, boundaries


def experiment_data_amount_kdv_PSO(grid_res, exp_name='kdv_PSO', save_plot=True):
    """
    Compares the performance of Adam and PSO optimizers in solving the KdV equation using a neural network.
    
    This method sets up the KdV problem, trains a neural network model using both Adam and PSO, and then evaluates
    their effectiveness based on training time, loss values, and error metrics. The goal is to assess how well each
    optimizer can minimize the error between the neural network's solution and the analytical solution of the KdV equation.
    
    Args:
        grid_res (int): The resolution of the grid used for solving the KdV equation.
        exp_name (str, optional): The name of the experiment. Defaults to 'kdv_PSO'.
        save_plot (bool, optional): A boolean indicating whether to save plots. Defaults to True.
    
    Returns:
        list: A list containing a dictionary with the experimental results. The dictionary includes training time,
            loss, and error metrics for both Adam and PSO optimizers, allowing for a direct comparison of their performance
            in solving the KdV equation.
    """
    solver_device('cuda')
    exp_dict_list = []

    grid, domain, equation, boundaries = kdv_problem_formulation(grid_res)

    net = torch.nn.Sequential(
        torch.nn.Linear(2, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 1)
    )

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=1, lambda_bound=100)

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=100,
                                         no_improvement_patience=1000,
                                         patience=5,
                                         randomize_parameter=1e-6,
                                         info_string_every=50)

    optim = Optimizer('Adam', {'lr': 1e-3})

    start = time.time()
    model.train(optim, 2e5, callbacks=[cb_es])
    end = time.time()

    time_adam = end - start

    grid = domain.build('autograd')

    grid_test = torch.cartesian_prod(torch.linspace(-10, 10, 100), torch.linspace(0, 1, 100))

    u_exact_train = u(grid).reshape(-1)

    u_exact_test = u(grid_test).reshape(-1)

    error_adam_train = torch.sqrt(torch.mean((u_exact_train - net(grid).reshape(-1)) ** 2))

    error_adam_test = torch.sqrt(torch.mean((u_exact_test - net(grid_test).reshape(-1)) ** 2))

    loss_adam = model.solution_cls.evaluate()[0].detach().cpu().numpy()

    lu_f = model.solution_cls.operator.operator_compute()

    lu_f, gr = integration(lu_f, grid)

    lu_f_adam, _ = integration(lu_f, gr)

    ########

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=100,
                                         no_improvement_patience=100,
                                         patience=2,
                                         randomize_parameter=1e-5,
                                         verbose=False,
                                         info_string_every=500)

    optim = Optimizer('PSO', {'pop_size': 20,  # 30
                              'b': 0.4,  # 0.5
                              'c2': 0.5,  # 0.05
                              'c1': 0.5,
                              'variance': 5e-4,
                              'lr': 1e-4})
    start = time.time()
    model.train(optim, 2e4, save_model=False, callbacks=[cb_es], info_string_every=20)
    end = time.time()
    time_pso = end - start

    error_pso_train = torch.sqrt(torch.mean((u_exact_train - net(grid).reshape(-1)) ** 2))

    error_pso_test = torch.sqrt(torch.mean((u_exact_test - net(grid_test).reshape(-1)) ** 2))

    loss_pso = model.solution_cls.evaluate()[0].detach().cpu().numpy()

    lu_f = model.solution_cls.operator.operator_compute()

    grid = domain.build('autograd')

    lu_f, gr = integration(lu_f, grid)

    lu_f_pso, _ = integration(lu_f, gr)

    #########

    exp_dict = {'grid_res': grid_res,
                'error_adam_train': error_adam_train.item(),
                'error_adam_test': error_adam_test.item(),
                'error_PSO_train': error_pso_train.item(),
                'error_PSO_test': error_pso_test.item(),
                'loss_adam': loss_adam.item(),
                'loss_pso': loss_pso.item(),
                "lu_f_adam": lu_f_adam.item(),
                "lu_f_pso": lu_f_pso.item(),
                'time_adam': time_adam,
                'time_pso': time_pso,
                'type': exp_name}

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE_adam {}= {}'.format(grid_res, error_adam_test))
    print('RMSE_pso {}= {}'.format(grid_res, error_pso_test))

    exp_dict_list.append(exp_dict)

    return exp_dict_list


def experiment_data_amount_kdv_CSO(grid_res, exp_name='kdv_CSO', save_plot=True):
    """
    Performs an experiment comparing Adam and CSO optimizers for solving the KdV equation.
        
        This method trains a neural network model to approximate the solution of the KdV equation
        using both Adam and CSO optimizers. By comparing these optimizers, the experiment aims
        to evaluate their effectiveness in training neural networks to solve differential equations.
        The evaluation is based on training and testing errors, loss values, computational time,
        and an integrated operator value. This helps to understand the performance characteristics
        of different optimization strategies within the neural differential equation solving framework.
        
        Args:
            grid_res: The grid resolution for the spatial domain.
            exp_name: The name of the experiment (default: 'kdv_CSO').
            save_plot: A boolean indicating whether to save plots (default: True).
        
        Returns:
            list: A list containing a dictionary with the experiment results, including
            training and testing errors for both Adam and CSO, loss values,
            computational times, and integrated operator values.
    """
    solver_device('cuda')
    exp_dict_list = []

    grid, domain, equation, boundaries = kdv_problem_formulation(grid_res)

    net = torch.nn.Sequential(
        torch.nn.Linear(2, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 1)
    )

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=1, lambda_bound=100)

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=100,
                                         no_improvement_patience=1000,
                                         patience=5,
                                         randomize_parameter=1e-6,
                                         info_string_every=500)

    optim = Optimizer('Adam', {'lr': 1e-3})

    start = time.time()
    model.train(optim, 2e5, callbacks=[cb_es])
    end = time.time()

    time_adam = end - start

    grid = domain.build('autograd')

    grid_test = torch.cartesian_prod(torch.linspace(-10, 10, 100), torch.linspace(0, 1, 100))

    u_exact_train = u(grid).reshape(-1)

    u_exact_test = u(grid_test).reshape(-1)

    error_adam_train = torch.sqrt(torch.mean((u_exact_train - net(grid).reshape(-1)) ** 2))

    error_adam_test = torch.sqrt(torch.mean((u_exact_test - net(grid_test).reshape(-1)) ** 2))

    loss_adam = model.solution_cls.evaluate()[0].detach().cpu().numpy()

    lu_f = model.solution_cls.operator.operator_compute()

    lu_f, gr = integration(lu_f, grid)

    lu_f_adam, _ = integration(lu_f, gr)

    ########

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=100,
                                         no_improvement_patience=100,
                                         patience=2,
                                         randomize_parameter=1e-5,
                                         verbose=False,
                                         info_string_every=20)

    optim = Optimizer('CSO', {'pop_size': 20,  # 30
                              'variance': 5e-4,
                              'lr': 1e-4})
    start = time.time()
    model.train(optim, 2e4, save_model=False, callbacks=[cb_es])
    end = time.time()
    time_pso = end - start

    error_pso_train = torch.sqrt(torch.mean((u_exact_train - net(grid).reshape(-1)) ** 2))

    error_pso_test = torch.sqrt(torch.mean((u_exact_test - net(grid_test).reshape(-1)) ** 2))

    loss_pso = model.solution_cls.evaluate()[0].detach().cpu().numpy()

    lu_f = model.solution_cls.operator.operator_compute()

    grid = domain.build('autograd')

    lu_f, gr = integration(lu_f, grid)

    lu_f_pso, _ = integration(lu_f, gr)

    #########

    exp_dict = {'grid_res': grid_res,
                'error_adam_train': error_adam_train.item(),
                'error_adam_test': error_adam_test.item(),
                'error_PSO_train': error_pso_train.item(),
                'error_PSO_test': error_pso_test.item(),
                'loss_adam': loss_adam.item(),
                'loss_pso': loss_pso.item(),
                "lu_f_adam": lu_f_adam.item(),
                "lu_f_pso": lu_f_pso.item(),
                'time_adam': time_adam,
                'time_pso': time_pso,
                'type': exp_name}

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE_adam {}= {}'.format(grid_res, error_adam_test))
    print('RMSE_pso {}= {}'.format(grid_res, error_pso_test))

    exp_dict_list.append(exp_dict)

    return exp_dict_list


if __name__ == '__main__':

    if not os.path.isdir('..\\results'):
        os.mkdir('..\\results')

    exp_dict_list = []

    nruns = 1

    for grid_res in range(100, 101, 10):
        for _ in range(nruns):
            exp_dict_list.append(experiment_data_amount_kdv_CSO(grid_res))
            exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
            df = pd.DataFrame(exp_dict_list_flatten)
            df.to_csv('..\\results\\kdv_CSO_adaptive_Z_V_UP_20_{}.csv'.format(grid_res))
