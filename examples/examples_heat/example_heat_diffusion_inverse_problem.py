import torch
import numpy as np
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
import time

solver_device('gpu')

data_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../PINNacle_data/heatinv_points.npy'))


# Function u(x, y, t)
def u_func(grid):
    """
    Calculates the analytical solution of the differential equation.
    
        This function provides a ground truth for evaluating the neural network's ability to approximate the solution.
        
        Args:
            grid (torch.Tensor): A tensor of shape (N, 3) containing the spatial (x, y) and temporal (t) coordinates where the solution is to be evaluated.
        
        Returns:
            torch.Tensor: A tensor of shape (N,) containing the analytical solution values at the corresponding coordinates.
    """
    x, y, t = grid[:, 0], grid[:, 1], grid[:, 2]
    sln = torch.sin(np.pi * x) * torch.sin(np.pi * y) * torch.exp(-t)
    return sln


# Function a(x, y)
def a_diffusion_coeff(grid):
    """
    Calculates a spatially varying diffusion coefficient.
    
        This method computes the diffusion coefficient 'a' based on spatial coordinates
        (x, y) from the input grid. The coefficient is calculated using a formula
        involving sine functions, allowing the diffusion to vary across the domain.
        This is useful for modeling systems where diffusion properties are not uniform.
    
        Args:
            grid (torch.Tensor): A 2D tensor where each row represents a spatial
                coordinate (x, y).
    
        Returns:
            torch.Tensor: The calculated diffusion coefficient 'a' as a tensor,
                with values corresponding to each spatial location in the grid.
                The negative of the diffusion coefficient is returned.
    
        WHY: The diffusion coefficient is calculated based on spatial coordinates to
        model systems where diffusion properties vary across the domain, enabling
        the neural network to learn and approximate solutions to differential
        equations with spatially dependent parameters.
    """
    x, y = grid[:, 0], grid[:, 1]
    a = 2 + torch.sin(np.pi * x) * torch.sin(np.pi * y)
    return -a


# Source function f(x, y, t)
def f_right_hand(grid):
    """
    Calculates the right-hand side of the differential equation.
    
        This function computes the value of the right-hand side of the differential
        equation at a given point in space and time. This value is used to train the neural network
        to approximate the solution of the differential equation. It leverages trigonometric functions
        and exponential decay to define the equation's behavior.
    
        Args:
            grid (torch.Tensor): A tensor of shape (N, 3) representing the spatial and temporal coordinates (x, y, t).
    
        Returns:
            torch.Tensor: A tensor of shape (N,) representing the value of the right-hand side of the differential equation at each point.
    """
    x, y, t = grid[:, 0], grid[:, 1], grid[:, 2]
    sin, cos, pi = torch.sin, torch.cos, np.pi

    term1 = (4 * pi ** 2 - 1) * sin(pi * x) * sin(pi * y)
    term2 = pi ** 2 * (2 * sin(pi * x) ** 2 * sin(pi * y) ** 2 -
                       cos(pi * x) ** 2 * sin(pi * y) ** 2 -
                       sin(pi * x) ** 2 * cos(pi * y) ** 2)

    return -torch.exp(-t) * (term1 + term2)


x_min, x_max = -1, 1
y_min, y_max = -1, 1
t_max = 1
grid_res = 10
N_samples = 2500

domain = Domain()

domain.variable('x', [x_min, x_max], grid_res)
domain.variable('y', [y_min, y_max], grid_res)
domain.variable('t', [0, t_max], grid_res)

x = domain.variable_dict['x']
y = domain.variable_dict['y']
t = domain.variable_dict['t']

data = np.load(data_file)

x_data = torch.tensor(data[:, 0]).reshape(-1)
y_data = torch.tensor(data[:, 1]).reshape(-1)
t_data = torch.tensor(data[:, 2]).reshape(-1)

boundaries = Conditions()

data_grid = torch.stack([x_data, y_data, t_data], dim=1)
u_bnd_val = u_func(data_grid).reshape(-1, 1) + torch.normal(0, 0.1, size=(2500, 1)).to(data_grid.device)

ind_bnd = np.random.choice(len(data_grid), N_samples, replace=False)

bnd_data = data_grid[ind_bnd]
u_bnd_val = u_bnd_val[ind_bnd]

boundaries.data(bnd=bnd_data, operator=None, value=u_bnd_val, var=0)
boundaries.dirichlet({'x': x_min, 'y': [y_min, y_max], 't': [0, t_max]}, value=a_diffusion_coeff, var=1)
boundaries.dirichlet({'x': x_max, 'y': [y_min, y_max], 't': [0, t_max]}, value=a_diffusion_coeff, var=1)
boundaries.dirichlet({'x': [x_min, x_max], 'y': y_min, 't': [0, t_max]}, value=a_diffusion_coeff, var=1)
boundaries.dirichlet({'x': [x_min, x_max], 'y': y_max, 't': [0, t_max]}, value=a_diffusion_coeff, var=1)

equation = Equation()

# Operator: −∇(a∇u) = f

heat_inverse = {
    'du/dt**1':
        {
            'coeff': 1,
            'du/dt': [2],
            'pow': 1,
            'var': 0
        },
    '-(da/dx * du/dx)**1':
        {
            'coeff': -1,
            'd2u/dx2': [[0], [0]],
            'pow': [1, 1],
            'var': [1, 0]
        },
    '-a * d2u/dx2**1':
        {
            'coeff': -1,
            'd2u/dx2': [[None], [0, 0]],
            'pow': [1, 1],
            'var': [1, 0]
        },
    '-(da/dy * du/dy)**1':
        {
            'coeff': -1,
            'd2u/dy2': [[1], [1]],
            'pow': [1, 1],
            'var': [1, 0]
        },
    '-a * d2u/dy2**1':
        {
            'coeff': -1,
            'd2u/dx2': [[None], [1, 1]],
            'pow': [1, 1],
            'var': [1, 0]
        },
    'f(x, y, t)':
        {
            'coeff': f_right_hand,
            'term': [None],
            'pow': 0
        }
}

equation.add(heat_inverse)

neurons = 100

net = torch.nn.Sequential(
    torch.nn.Linear(3, neurons),
    torch.nn.Tanh(),
    torch.nn.Linear(neurons, neurons),
    torch.nn.Tanh(),
    torch.nn.Linear(neurons, neurons),
    torch.nn.Tanh(),
    torch.nn.Linear(neurons, neurons),
    torch.nn.Tanh(),
    torch.nn.Linear(neurons, neurons),
    torch.nn.Tanh(),
    torch.nn.Linear(neurons, neurons),
    torch.nn.Tanh(),
    torch.nn.Linear(neurons, 2)
)

for m in net.modules():
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)

model = Model(net, domain, equation, boundaries)

model.compile('autograd', lambda_operator=1, lambda_bound=100)

img_dir = os.path.join(os.path.dirname(__file__), 'heat_2d_inverse_problem_img')

cb_cache = cache.Cache(cache_verbose=True, model_randomize_parameter=1e-6)

cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                     loss_window=100,
                                     no_improvement_patience=1000,
                                     patience=5,
                                     randomize_parameter=1e-6,
                                     info_string_every=10)

cb_plots = plot.Plots(save_every=100,
                      print_every=None,
                      img_dir=img_dir,
                      img_dim='2d',
                      scatter_flag=False,
                      plot_axes=[0, 1],
                      fixed_axes=[2],
                      n_samples=4,
                      img_rows=2,
                      img_cols=2)

optimizer = Optimizer('Adam', {'lr': 1e-3})

callbacks = [cb_cache, cb_es, cb_plots]

start = time.time()

model.train(optimizer, 1e6, save_model=True, callbacks=callbacks)
