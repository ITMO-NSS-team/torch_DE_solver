import torch
import os
import sys
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import adaptive_lambda, cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device

solver_device('cpu')

mu_1 = 1
mu_2 = 4
k = 8
A = 10

x_min, x_max = -1, 1
y_min, y_max = -1, 1
grid_res = 100

domain = Domain()
domain.variable('x', [x_min, x_max], grid_res)
domain.variable('y', [y_min, y_max], grid_res)

x = domain.variable_dict['x']
y = domain.variable_dict['y']

boundaries = Conditions()

# Circle type of removed domains #######################################################################################

removed_domains_lst = [
    {'circle': {'center': (0.5, 0.5), 'radius': 0.2}},
    {'circle': {'center': (0.4, -0.4), 'radius': 0.4}},
    {'circle': {'center': (-0.2, -0.7), 'radius': 0.1}},
    {'circle': {'center': (-0.6, 0.5), 'radius': 0.3}}
]

# CSG boundaries #######################################################################################################

boundaries.dirichlet({'circle': {'center': (0.5, 0.5), 'radius': 0.2}}, value=1)
boundaries.dirichlet({'circle': {'center': (0.4, -0.4), 'radius': 0.4}}, value=1)
boundaries.dirichlet({'circle': {'center': (-0.2, -0.7), 'radius': 0.1}}, value=1)
boundaries.dirichlet({'circle': {'center': (-0.6, 0.5), 'radius': 0.3}}, value=1)

# Non CSG boundaries ###################################################################################################

boundaries.dirichlet({'x': x_min, 'y': [y_min, y_max]}, value=0.2)
boundaries.dirichlet({'x': x_max, 'y': [y_min, y_max]}, value=0.2)
boundaries.dirichlet({'x': [x_min, x_max], 'y': y_min}, value=0.2)
boundaries.dirichlet({'x': [x_min, x_max], 'y': y_max}, value=0.2)


def forcing_term(grid):
    x, y = grid[:, 0], grid[:, 1]
    return -A * (mu_1 ** 2 + mu_2 ** 2 + x ** 2 + y ** 2) * \
           torch.sin(mu_1 * torch.pi * x) * \
           torch.sin(mu_2 * torch.pi * y)


equation = Equation()

# Operator: -d2u/dx2 - d2u/dy2 + k ** 2 * u = f(x, y)

poisson = {
    '-d2u/dx2':
        {
            'coeff': -1.,
            'term': [0, 0],
            'pow': 1,
        },
    '-d2u/dy2':
        {
            'coeff': -1.,
            'term': [1, 1],
            'pow': 1,
        },
    'k ** 2 * u':
        {
            'coeff': k ** 2,
            'term': [None],
            'pow': 1
        },
    'f(x, y)':
        {
            'coeff': forcing_term,
            'term': [None],
            'pow': 1
        }
}

equation.add(poisson)

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
    torch.nn.Linear(neurons, 1)
)

for m in net.modules():
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)

model = Model(net, domain, equation, boundaries)

model.compile('autograd', lambda_operator=1, lambda_bound=100, removed_domains=removed_domains_lst)

img_dir = os.path.join(os.path.dirname(__file__), 'poisson_2d_irregular_geometry_img')

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
                      img_dim='2d_scatter')  # 3 image dimension options: 3d, 2d, 2d_scatter

optimizer = Optimizer('Adam', {'lr': 5e-5})

model.train(optimizer, 1e5, save_model=True, callbacks=[cb_cache, cb_es, cb_plots])









