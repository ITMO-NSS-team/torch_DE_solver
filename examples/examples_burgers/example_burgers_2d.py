import torch
import os
import sys
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model

from tedeous.callbacks import early_stopping, plot, cache
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.utils import exact_solution_data
from tedeous.utils import init_data

solver_device('gpu')

data_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../PINNacle_data/burgers2d_0.npy"))

data_init_u_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../PINNacle_data/burgers2d_init_u_0.npy"))

data_init_v_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../PINNacle_data/burgers2d_init_v_0.npy"))

mu = 0.001


def init_w(x, y, size, L):
    """
    Initializes a tensor based on spatial coordinates to represent a component of the differential equation's solution space.
    
        This method calculates a tensor by summing sinusoidal and cosinusoidal
        components, each weighted by random values. The components are determined
        by the input coordinates x and y, and the summation is performed over a
        range defined by L. This process effectively creates a feature map that captures spatial dependencies,
        allowing the neural network to learn the underlying solution surface of the differential equation.
    
        Args:
          x (float): The x-coordinate.
          y (float): The y-coordinate.
          size (int): The size of the random tensors used for weighting.
          L (int): The range for the summation.
    
        Returns:
          torch.Tensor: A tensor resulting from the summation of the sinusoidal
            and cosinusoidal components, representing a localized feature at the given coordinates.
    """
    a = torch.randn(size, size)
    b = torch.randn(size, size)

    result = torch.zeros(size)
    for i in range(-L, L + 1):
        for j in range(-L, L + 1):
            new_component = a[i, j] * torch.sin(2 * torch.pi * (i * x + j * y)) \
                      + b[i, j] * torch.cos(2 * torch.pi * (i * x + j * y))
            result += new_component

    return result


def init_u(grid):
    """
    Initializes the 'u' variable, representing one of the components of the differential equation's solution.
    
        This method computes an initial state for 'u' based on spatial coordinates,
        serving as a starting point for the neural network's approximation of the
        differential equation's solution. The initial state is constructed using a
        deterministic function `init_w` combined with a random component to introduce
        diversity in the initial conditions, aiding in exploration of the solution space.
    
        Args:
            grid (torch.Tensor): A tensor containing the grid coordinates (x, y) at which
                the solution is to be approximated.
    
        Returns:
            torch.Tensor: The initialized 'u' variable, calculated as
                2 * init_w(x, y, size, L) + c_u, where init_w provides a base initial
                condition derived from the grid, and c_u is a tensor of random numbers
                added to diversify the initial state.
    """
    x, y = grid[:, 0], grid[:, 1]
    size = int(len(x) ** 1)
    L = int(torch.max(x))
    c_u = torch.randn(size)
    return 2 * init_w(x, y, size, L) + c_u


def init_v(grid):
    """
    Initializes the velocity field for solving differential equations.
    
        This method leverages a grid to initialize the velocity field, a crucial step
        in setting up the neural network-based differential equation solver. The
        initialization combines a random component with a scaled result from another
        initialization function to provide a suitable starting point for the solver's
        optimization process. This approach helps the solver to explore the solution
        space more effectively.
    
        Args:
            grid (torch.Tensor): The input grid containing x and y coordinates,
                                 representing the spatial domain of the differential
                                 equation.
    
        Returns:
            torch.Tensor: The initialized velocity field, ready for use in the
                          differential equation solver.
    """
    x, y = grid[:, 0], grid[:, 1]
    size = int(len(x) ** 1)
    L = int(torch.max(x))
    c_v = torch.randn(size)
    return 2 * init_w(x, y, size, L) + c_v


def burgers_2d_coupled_experiment(grid_res):
    """
    Performs a 2D coupled Burgers' equation experiment.
        
        This method sets up and runs a simulation of the 2D coupled Burgers'
        equation using a neural network-based solver. It defines the domain,
        boundary conditions, equation, and network architecture, then trains
        the model and evaluates its performance by comparing the neural network
        solution to the exact solution. This allows for assessing the accuracy
        and efficiency of the neural network approach for solving this PDE.
        
        Args:
            grid_res: The grid resolution for the spatial and temporal domains.
        
        Returns:
            tuple[list[dict], list[dict]]: A tuple containing two lists of dictionaries.
                The first list contains a dictionary with the results for u,
                and the second list contains a dictionary with the results for v.
                Each dictionary includes the grid resolution, training time,
                root mean squared error (RMSE), experiment type, and a flag
                indicating whether caching was used.
    """
    exp_dict_list_u, exp_dict_list_v = [], []

    x_min, L = 0, 4
    y_min, L = 0, 4
    T = 1
    # grid_res = 20

    pde_dim_in = 3
    pde_dim_out = 2

    domain = Domain()
    domain.variable('x', [x_min, L], grid_res)
    domain.variable('y', [y_min, L], grid_res)
    domain.variable('t', [0, T], grid_res)

    boundaries = Conditions()

    # Initial conditions ###############################################################################################

    # # With use custom functions for IC
    #
    # # u(x, y, 0)
    # boundaries.dirichlet({'x': [0, L], 'y': [0, L], 't': 0}, value=init_u, var=0)
    #
    # # v(x, y, 0)
    # boundaries.dirichlet({'x': [0, L], 'y': [0, L], 't': 0}, value=init_v, var=1)

    # With use IC data

    init_u_data = lambda grid: init_data(grid[:, :2], data_init_u_file)
    init_v_data = lambda grid: init_data(grid[:, :2], data_init_v_file)

    # u(x, y, 0)
    boundaries.dirichlet({'x': [0, L], 'y': [0, L], 't': 0}, value=init_u_data, var=0)

    # v(x, y, 0)
    boundaries.dirichlet({'x': [0, L], 'y': [0, L], 't': 0}, value=init_v_data, var=1)

    # Boundary conditions ##############################################################################################

    # u(0, y, t) = u(L, y, t)
    boundaries.periodic([{'x': 0, 'y': [0, L], 't': [0, T]}, {'x': L, 'y': [0, L], 't': [0, T]}], var=0)

    # u(x, 0, t) = u(x, L, t)
    boundaries.periodic([{'x': [0, L], 'y': 0, 't': [0, T]}, {'x': [0, L], 'y': L, 't': [0, T]}], var=0)

    # v(0, y, t) = v(L, y, t)
    boundaries.periodic([{'x': 0, 'y': [0, L], 't': [0, T]}, {'x': L, 'y': [0, L], 't': [0, T]}], var=1)

    # v(x, 0, t) = v(x, L, t)
    boundaries.periodic([{'x': [0, L], 'y': 0, 't': [0, T]}, {'x': [0, L], 'y': L, 't': [0, T]}], var=1)

    equation = Equation()

    # Operator 1: u_t + u * u_x + v * u_y - mu * (u_xx + u_yy) = 0

    burgers_u = {
        'du/dt**1':
            {
                'coeff': 1.,
                'du/dt': [2],
                'pow': 1,
                'var': 0
            },
        '+u*du/dx':
            {
                'coeff': 1.,
                'u*du/dx': [[None], [0]],
                'pow': [1, 1],
                'var': [0, 0]
            },
        '+v*du/dy':
            {
                'coeff': 1.,
                'u*du/dy': [[None], [1]],
                'pow': [1, 1],
                'var': [1, 0]
            },
        '-mu*d2u/dx2':
            {
                'coeff': -mu,
                'd2u/dx2': [0, 0],
                'pow': 1,
                'var': 0
            },
        '-mu*d2u/dy2':
            {
                'coeff': -mu,
                'd2u/dy2': [1, 1],
                'pow': 1,
                'var': 0
            }
    }

    # Operator 2: v_t + u * v_x + v * v_y - mu * (v_xx + v_yy) = 0

    burgers_v = {
        'dv/dt**1':
            {
                'coeff': 1.,
                'dv/dt': [2],
                'pow': 1,
                'var': 1
            },
        '+u*dv/dx':
            {
                'coeff': 1.,
                'u*dv/dx': [[None], [0]],
                'pow': [1, 1],
                'var': [0, 1]
            },
        '+v*dv/dy':
            {
                'coeff': 1.,
                'v*dv/dy': [[None], [1]],
                'pow': [1, 1],
                'var': [1, 1]
            },
        '-mu*d2v/dx2':
            {
                'coeff': -mu,
                'd2v/dx2': [0, 0],
                'pow': 1,
                'var': 1
            },
        '-mu*d2v/dy2':
            {
                'coeff': -mu,
                'd2v/dy2': [1, 1],
                'pow': 1,
                'var': 1
            }
    }

    equation.add(burgers_u)
    equation.add(burgers_v)

    neurons = 100
    net = torch.nn.Sequential(
        torch.nn.Linear(pde_dim_in, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, pde_dim_out)
    )

    start = time.time()

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=1, lambda_bound=100)

    cb_cache = cache.Cache(cache_verbose=False, model_randomize_parameter=1e-5)

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         patience=5,
                                         randomize_parameter=1e-5,
                                         info_string_every=10)

    img_dir = os.path.join(os.path.dirname(__file__), 'burgers_2d_coupled_img')

    cb_plots = plot.Plots(save_every=100,
                          print_every=None,
                          img_dir=img_dir,
                          img_dim='3d',
                          scatter_flag=False,
                          n_samples=4,
                          plot_axes=[0, 1],
                          fixed_axes=[2],
                          var_transpose=False)

    optimizer = Optimizer('Adam', {'lr': 5e-3})

    model.train(optimizer, 5e5, save_model=True, callbacks=[cb_es, cb_plots, cb_cache])

    end = time.time()

    grid = domain.build('NN').to('cuda')
    net = net.to('cuda')

    predicted_u, predicted_v = net(grid)[:, 0], net(grid)[:, 1]
    exact_u, exact_v = exact_solution_data(grid, data_file, pde_dim_in, pde_dim_out, t_dim_flag=True)

    error_rmse_u = torch.sqrt(torch.mean((exact_u - predicted_u) ** 2))
    error_rmse_v = torch.sqrt(torch.mean((exact_v - predicted_v) ** 2))

    exp_dict_list_u.append({
        'grid_res': grid_res,
        'time': end - start,
        'RMSE_u_func': error_rmse_u.detach().cpu().numpy(),
        'type': 'burgers_2d_coupled',
        'cache': True
    })
    exp_dict_list_v.append({
        'grid_res': grid_res,
        'time': end - start,
        'RMSE_v_func': error_rmse_v.detach().cpu().numpy(),
        'type': 'burgers_2d_coupled',
        'cache': True
    })

    print('Time taken {}= {}'.format(grid_res, end - start))

    print('RMSE_u_func {}= {}'.format(grid_res, error_rmse_u))
    print('RMSE_v_func {}= {}'.format(grid_res, error_rmse_v))

    return exp_dict_list_u, exp_dict_list_v


nruns = 10

exp_dict_list_u, exp_dict_list_v = [], []

for grid_res in range(20, 201, 20):
    for _ in range(nruns):
        list_u, list_v = burgers_2d_coupled_experiment(grid_res)
        exp_dict_list_u.append(list_u)
        exp_dict_list_v.append(list_v)

import pandas as pd

exp_dict_list_u_flatten = [item for sublist in exp_dict_list_u for item in sublist]
exp_dict_list_v_flatten = [item for sublist in exp_dict_list_v for item in sublist]

df_u = pd.DataFrame(exp_dict_list_u_flatten)
df_v = pd.DataFrame(exp_dict_list_v_flatten)

df_u.to_csv('examples/benchmarking_data/burgers_2d_coupled_experiment_20_200_cache_u_func={}.csv'.format(str(True)))
df_v.to_csv('examples/benchmarking_data/burgers_2d_coupled_experiment_20_200_cache_v_func={}.csv'.format(str(True)))
