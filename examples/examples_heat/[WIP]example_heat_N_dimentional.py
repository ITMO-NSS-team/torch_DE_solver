import torch
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import adaptive_lambda, cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device

solver_device('gpu')


def x_norm(grid):
    """
    Computes the squared Euclidean norms of the input vectors, excluding the last component.
    
        This function calculates the squared Euclidean norm for each vector in the input grid,
        excluding the last element of each vector. This is useful for computing distance-related
        features or scaling factors in the context of solving differential equations, where
        the last component might represent a time variable or another independent parameter.
    
        Args:
            grid (torch.Tensor): A tensor of shape (N, M) representing a set of N vectors, each with M components.
    
        Returns:
            torch.Tensor: A tensor of shape (N, 1) containing the squared Euclidean norms of the input vectors,
                          excluding the last component.
    """
    return (grid[:, :-1] ** 2).sum(axis=1).reshape(-1, 1)


def g_x(grid):
    """
    Calculates a Gaussian function based on the normalized input grid, which is then used as a component in the neural network's solution of a differential equation.
    
        The method normalizes the input grid, applies an exponential function,
        and incorporates the last column of the grid. This Gaussian function contributes to the overall approximation
        of the differential equation's solution by the neural network.
    
        Args:
            grid (torch.Tensor): The input grid representing the domain of the differential equation.
    
        Returns:
            torch.Tensor: The result of applying the Gaussian function to the grid, contributing to the neural network's solution.
    """
    return torch.exp(x_norm(grid) / 2 + grid[:, -1:])


def bop_generation(coeff_x, i_dim):
    """
    Generates a dictionary representing a Bilinear Operator (BOP) for the neural network-based differential equation solver.
    
    This method constructs a dictionary that defines a bilinear operator,
    specifically for terms involving 'alpha * u' and 'beta * nx_i * du/dx_i'.
    It sets the coefficients, terms, and powers associated with each part
    of the operator. This operator is used to define the structure of the
    differential equation within the neural network solver.
    
    Args:
        coeff_x (float): Coefficient for the 'beta * nx_i * du/dx_i' term.
        i_dim (int): The dimension index 'i' for the 'beta * nx_i * du/dx_i' term,
                 representing the spatial dimension in the differential equation.
    
    Returns:
        dict: A dictionary representing the Bilinear Operator (BOP) with
        the 'alpha * u' and 'beta * nx_i * du/dx_i' terms defined. This
        dictionary is used to specify the differential equation's structure
        to the neural network solver.
    """
    bop = {
        'alpha * u':
            {
                'coeff': 0,
                'term': [None],
                'pow': 1
            },
        'beta * nx_i * du/dx_i':
            {
                'coeff': coeff_x,
                'term': [i_dim],
                'pow': 1
            }
    }
    return bop


n_dim = 5
k = 1 / n_dim

x_min, x_max = -1, 1
t_max = 1
domains_lst = [[x_min, x_max]] * n_dim
grid_res = 10

domain = Domain()

for i in range(n_dim):
    domain.variable(f'x_{i + 1}', domains_lst[i], grid_res)

domain.variable('t', [0, t_max], grid_res)

boundaries = Conditions()

variable_names_lst = list(domain.variable_dict.keys())

# Initial conditions ###################################################################################################

bnd = {x_i: [x_min, x_max] for x_i in variable_names_lst}
bnd['t'] = 0
boundaries.dirichlet(bnd, value=g_x)

# Boundary conditions ##################################################################################################

for i in range(n_dim):
    d_min = {variable_names_lst[i]: x_min}
    d_max = {variable_names_lst[i]: x_max}
    for j in range(n_dim):
        if i != j:
            d_min[variable_names_lst[j]] = [x_min, x_max]
            d_max[variable_names_lst[j]] = [x_min, x_max]

    d_min['t'] = [0, t_max]
    d_max['t'] = [0, t_max]

    operator_min = bop_generation(x_min, i)
    operator_max = bop_generation(x_max, i)

    boundaries.robin(d_min, operator=operator_min, value=g_x)
    boundaries.robin(d_max, operator=operator_max, value=g_x)

equation = Equation()


def forcing_term(grid):
    """
    Calculates the forcing term for a given grid.
    
        The forcing term is computed as the negative product of a constant `k`,
        the L2 norm of the grid coordinates, and the function `g_x` evaluated at the grid.
        This term is crucial for shaping the solution learned by the neural network, guiding it towards satisfying the differential equation.
    
        Args:
            grid (torch.Tensor): The grid on which to calculate the forcing term.
    
        Returns:
            torch.Tensor: The calculated forcing term.
    """
    return -k * x_norm(grid) * g_x(grid)


# Operator: du/dt = k * ∆u + f(x, t)

heat_N_dim = {
    'du/dt':
        {
            'coeff': 1,
            'term': [2],
            'pow': 1
        }
}
for i in range(n_dim):
    heat_N_dim[f'd2u/dx{i}2'] = {
        'coeff': -k,
        'term': [i, i],
        'pow': 1
    }

heat_N_dim['f(x_1, ..., x_n)'] = {
    'coeff': forcing_term,
    'term': [None],
    'pow': 0
}

equation.add(heat_N_dim)

neurons = 100

net = torch.nn.Sequential(
    torch.nn.Linear(n_dim + 1, neurons),
    torch.nn.Tanh(),
    torch.nn.Linear(neurons, neurons),
    torch.nn.Tanh(),
    torch.nn.Linear(neurons, neurons),
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

img_dir = os.path.join(os.path.dirname(__file__), 'example_heat_N_dimensional_img')

cb_cache = cache.Cache(cache_verbose=True, model_randomize_parameter=1e-5)

cb_es = early_stopping.EarlyStopping(eps=1e-9,
                                     loss_window=100,
                                     no_improvement_patience=1000,
                                     patience=5,
                                     info_string_every=10,
                                     randomize_parameter=1e-5)

# cb_plots = plot.Plots(save_every=100,
#                       print_every=None,
#                       img_dir=img_dir,
#                       img_dim='2d',
#                       scatter_flag=True,
#                       plot_axes=[0, 1, 2],
#                       fixed_axes=[3],
#                       n_samples=4,
#                       img_rows=2,
#                       img_cols=2)

optimizer = Optimizer('Adam', {'lr': 5e-3})

model.train(optimizer, 1e5, save_model=True, callbacks=[cb_cache, cb_es])
