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


def exact_solution(grid):
    """
    Computes the analytical solution of the differential equation on the given grid.
    
    This function calculates the exact solution by summing the sine of each
    coordinate in the grid. This provides a ground truth for evaluating the neural network's approximation.
    
    Args:
        grid (torch.Tensor): The input grid of points where the solution is evaluated.
            Shape: (num_points, n_dim), where num_points is the number of points in the grid and n_dim is the number of dimensions.
    
    Returns:
        torch.Tensor: The exact solution at each point in the grid.
            Shape: (num_points,). This represents the analytical solution of the differential equation at each grid point.
    """
    u_exact = 0
    for i in range(n_dim):
        u_exact += torch.sin(grid[:, i])

    return u_exact


def forcing_term(grid):
    """
    Computes the forcing term for the Poisson equation.
    
    This term is crucial as it dictates the behavior of the solution
    obtained by the neural network, guiding it towards the true solution.
    
    Args:
        grid (torch.Tensor): The spatial grid on which to evaluate the forcing term.
    
    Returns:
        torch.Tensor: The forcing term evaluated on the grid.
    """
    return torch.pi ** 2 / 4 * exact_solution(grid)


n_dim = 4
coeff_laplacian = -1
pow_laplacian = 1

x_min, x_max = 0, 1
domains_lst = [[x_min, x_max]] * n_dim
grid_res = 10

domain = Domain()

for i in range(n_dim):
    domain.variable(f'x_{i + 1}', domains_lst[i], grid_res)

boundaries = Conditions()

variable_names_lst = list(domain.variable_dict.keys())
for i in range(n_dim):
    d_min = {variable_names_lst[i]: x_min}
    d_max = {variable_names_lst[i]: x_max}
    for j in range(n_dim):
        if i != j:
            d_min[variable_names_lst[j]] = [x_min, x_max]
            d_max[variable_names_lst[j]] = [x_min, x_max]

    boundaries.dirichlet(d_min, value=exact_solution)
    boundaries.dirichlet(d_max, value=exact_solution)

equation = Equation()

# Operator: −∆u = pi ** 2 / 4 * sum(sin(pi / 2 * x_i))

poisson = {}
for i in range(n_dim):
    poisson[f'd2u/dx{i}2'] = {
        'coeff': coeff_laplacian,
        'term': [i, i],
        'pow': pow_laplacian
    }

poisson['f(x_1, ..., x_n)'] = {
    'coeff': forcing_term,
    'term': [None],
    'pow': 0
}

equation.add(poisson)

neurons = 100

net = torch.nn.Sequential(
    torch.nn.Linear(n_dim, neurons),
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

img_dir = os.path.join(os.path.dirname(__file__), 'poisson_N_dimentional_img')

cb_cache = cache.Cache(cache_verbose=True, model_randomize_parameter=1e-5)

cb_es = early_stopping.EarlyStopping(eps=1e-9,
                                     loss_window=100,
                                     no_improvement_patience=1000,
                                     patience=5,
                                     info_string_every=6,
                                     randomize_parameter=1e-5)

cb_plots = plot.Plots(save_every=100,
                      print_every=None,
                      img_dir=img_dir,
                      img_dim='4d',
                      plot_axes=[0, 1, 2],
                      fixed_axes=[3],
                      n_samples=4,
                      img_rows=2,
                      img_cols=2)

optimizer = Optimizer('Adam', {'lr': 5e-3})

model.train(optimizer, 1e5, save_model=True, callbacks=[cb_cache, cb_es, cb_plots])
