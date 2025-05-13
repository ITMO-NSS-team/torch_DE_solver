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
    return (grid[:, :-1] ** 2).sum(axis=1).reshape(-1, 1)


def g_x(grid):
    return torch.exp(x_norm(grid) / 2 + grid[:, -1:])


def bop_generation(coeff_x, i_dim):
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
