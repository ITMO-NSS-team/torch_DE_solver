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
    """
    Computes the solution to a partial differential equation on a given grid using a neural network-based approach.
    
        This method leverages numerical integration to approximate the solution u(x, t) at each point in the grid.
        The solution is obtained by evaluating integrals with integrands dependent on x, t, and a parameter mu,
        effectively using a neural network to learn the underlying solution manifold. This approach allows us to approximate solutions to differential equations by training a neural network to represent the solution space.
    
        Args:
            grid (list): A list of tuples, where each tuple represents a point (x, t) in the grid.
    
        Returns:
            torch.Tensor: A tensor containing the solution u(x, t) for each point in the grid, as approximated by the neural network.
    """
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
            x: The input tensor representing the spatial or temporal domain of the differential equation.
    
        Returns:
            numpy.ndarray: The L2 norm, a scalar value representing the overall error, as a NumPy array.
    """
    x = x.to('cpu')
    net = net.to('cpu')
    predict = net(x).detach().cpu().reshape(-1)
    exact = u(x).detach().cpu().reshape(-1)
    l2_norm = torch.sqrt(sum((predict - exact) ** 2))
    return l2_norm.detach().cpu().numpy()


def l2_norm_mat(net, x):
    """
    Calculates the L2 norm between the neural network's prediction and the exact solution of the differential equation. This metric quantifies the accuracy of the neural network's approximation.
    
        Args:
            net (torch.nn.Module): The neural network model used to approximate the solution.
            x (torch.Tensor): The input data (typically the independent variable of the differential equation).
    
        Returns:
            numpy.ndarray: The L2 norm between the network's prediction and the exact solution, as a NumPy array. This value represents the overall error in the approximation.
    """
    x = x.to('cpu')
    net = net.to('cpu')
    predict = net.detach().cpu().reshape(-1)
    exact = u(x).detach().cpu().reshape(-1)
    l2_norm = torch.sqrt(sum((predict - exact) ** 2))
    return l2_norm.detach().cpu().numpy()


def l2_norm_fourier(net, x):
    """
    Computes the L2 norm in Fourier space to evaluate the accuracy of the neural network's solution compared to the exact solution. This metric quantifies the error in the frequency domain, providing insights into how well the network captures the different frequency components of the solution.
    
        Args:
            net: The neural network model.
            x: The input tensor.
    
        Returns:
            np.ndarray: The L2 norm between the prediction and the exact solution as a NumPy array.
    """
    x = x.to(torch.device('cuda:0'))
    predict = net(x).detach().cpu().reshape(-1)
    exact = u(x).detach().cpu().reshape(-1)
    l2_norm = torch.sqrt(sum((predict - exact) ** 2))
    return l2_norm.detach().cpu().numpy()


def burgers1d_problem_formulation(grid_res):
    """
    Sets up the 1D Burgers' equation problem for neural network-based solving.
    
    This method configures the domain, boundary conditions, and the Burgers' equation itself,
    preparing it for approximation using a neural network. The setup includes defining the spatial and
    temporal domains, specifying Dirichlet boundary conditions, and formulating the equation in a
    format suitable for neural network training. This is a crucial step in leveraging neural networks
    to find solutions to differential equations by translating the problem into a trainable format.
    
    Args:
        grid_res (int): The resolution of the grid, determining the number of points in the spatial and temporal domains.
    
    Returns:
        tuple: A tuple containing the grid, domain, equation, and boundaries.
            - grid (torch.Tensor): The computational grid, a tensor representing the discretized domain.
            - domain (Domain): The problem domain, defining the spatial and temporal extents.
            - equation (Equation): The equation to be solved, represented in a symbolic format.
            - boundaries (Conditions): The boundary conditions, specifying the solution's behavior at the domain's edges.
    """
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
    """
    Performs a Burgers' equation experiment using the Adam optimizer to approximate its solution with a neural network.
        
        This method sets up and runs a Burgers' equation simulation using the Adam optimizer.
        It trains a neural network to approximate the solution and records training and testing errors, loss, and execution time to evaluate the approximation quality.
        
        Args:
          grid_res: The grid resolution for the simulation, influencing the training data density.
          iter: The iteration number for the experiment, used for file naming.
          exp_name: The name of the experiment (default: 'burgers1d_adam_5_starts').
          save_plot: A boolean indicating whether to save plots (default: True).
        
        Returns:
          list: A list containing a dictionary with the experiment results, including grid resolution,
            training and testing errors, loss, LU-factor, execution time, and experiment type.
            The LU-factor is related to the differential operator approximation.
    """
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
                                         info_string_every=500)

    optim = Optimizer('Adam', {'lr': 1e-3})

    start = time.time()
    path_to_folder = os.path.join(current_file, "trajectories", "burgers", "adam_5_starts", f"adam_{iter}".format(iter))
    os.makedirs(path_to_folder, exist_ok=True)
    cb_sm = save_model.SaveModel(path_to_folder, every_step=500)
    model.train(optim, 5e3, save_model=False, callbacks=[cb_es, cb_sm])
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
