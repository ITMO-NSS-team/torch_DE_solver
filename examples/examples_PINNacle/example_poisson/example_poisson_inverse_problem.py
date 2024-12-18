import torch
import os
import sys
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import adaptive_lambda, cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device

solver_device('cpu')


def a_ref(grid):
    x, y = grid[:, 0], grid[:, 1]
    return 1 / (1 + x ** 2 + y ** 2 + (x - 1) ** 2 + (y - 1) ** 2)


def u_func(grid):
    x, y = grid[:, 0], grid[:, 1]
    return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)


def forcing_term(grid):
    x, y = grid[:, 0], grid[:, 1]

    term_1 = 2 * torch.pi ** 2 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y) * a_ref(grid)

    term_2 = 2 * torch.pi ** 2 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y) * a_ref(grid) + \
             2 * torch.pi * ((2 * x + 1) * torch.cos(torch.pi * x) * torch.sin(torch.pi * y) +
                             (2 * y + 1) * torch.sin(torch.pi * x) * torch.cos(torch.pi * y)) * a_ref(grid) ** 2
    return term_1 + term_2


x_min, x_max = 0, 1
y_min, y_max = 0, 1
grid_res = 50
N_samples = 2500

domain = Domain()
domain.variable('x', [x_min, x_max], grid_res)
domain.variable('y', [y_min, y_max], grid_res)

x = domain.variable_dict['x']
y = domain.variable_dict['y']

boundaries = Conditions()

bc_x = np.linspace(0, 1, int(N_samples ** 0.5))
bc_y = np.linspace(0, 1, int(N_samples ** 0.5))
bc_x, bc_y = np.meshgrid(bc_x, bc_y)

data_grid = torch.tensor(np.stack((bc_x.reshape(-1), bc_y.reshape(-1)))).T
u_bnd_val = u_func(data_grid).reshape(-1, 1) + torch.normal(0, 0.1, size=(2500, 1))

ind_bnd = np.random.choice(len(data_grid), N_samples, replace=False)

bnd_data = data_grid[ind_bnd]
u_bnd_val = u_bnd_val[ind_bnd]

boundaries.data(bnd=bnd_data, operator=None, value=u_bnd_val, var=0)
boundaries.dirichlet({'x': x_min, 'y': [y_min, y_max]}, value=a_ref, var=1)
boundaries.dirichlet({'x': x_max, 'y': [y_min, y_max]}, value=a_ref, var=1)
boundaries.dirichlet({'x': [x_min, x_max], 'y': y_min}, value=a_ref, var=1)
boundaries.dirichlet({'x': [x_min, x_max], 'y': y_max}, value=a_ref, var=1)

equation = Equation()

# Operator: −∇(a∇u) = f(x, y)

poisson_inverse = {
    'da/dx * du/dx':
        {
            'coeff': 1,
            'term': [[0], [0]],
            'pow': [1, 1],
            'var': [1, 0]
        },
    'a * d2u/dx2':
        {
            'coeff': 1,
            'term': [[None], [0, 0]],
            'pow': [1, 1],
            'var': [1, 0]
        },
    'da/dy * du/dy':
        {
            'coeff': 1,
            'term': [[1], [1]],
            'pow': [1, 1],
            'var': [1, 0]
        },
    'a * d2u/dy2':
        {
            'coeff': 1,
            'term': [[None], [1, 1]],
            'pow': [1, 1],
            'var': [1, 0]
        },
    'f(x, y)':
        {
            'coeff': forcing_term,
            'term': [None],
            'pow': 0
        }
}

equation.add(poisson_inverse)

neurons = 100

net = torch.nn.Sequential(
    torch.nn.Linear(2, neurons),
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

img_dir = os.path.join(os.path.dirname(__file__), 'example_poisson_inverse_problem_img')

cb_cache = cache.Cache(cache_verbose=True, model_randomize_parameter=1e-5)

cb_es = early_stopping.EarlyStopping(eps=1e-9,
                                     loss_window=100,
                                     no_improvement_patience=1000,
                                     patience=5,
                                     info_string_every=10,
                                     randomize_parameter=1e-5)

cb_plots = plot.Plots(save_every=50,
                      print_every=None,
                      img_dir=img_dir,
                      img_dim='2d')  # 3 image dimension options: 3d, 2d, 2d_scatter

optimizer = Optimizer('Adam', {'lr': 1e-3})

model.train(optimizer, 1e5, save_model=True, callbacks=[cb_cache, cb_es, cb_plots])
