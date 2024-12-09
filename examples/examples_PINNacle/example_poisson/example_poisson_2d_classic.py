import torch
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import adaptive_lambda, cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device

solver_device('cpu')

x_min, x_max = -0.5, 0.5
y_min, y_max = -0.5, 0.5
grid_res = 50

domain = Domain()
domain.variable('x', [x_min, x_max], grid_res)
domain.variable('y', [y_min, y_max], grid_res)

x = domain.variable_dict['x']
y = domain.variable_dict['y']

boundaries = Conditions()

# Circle type of removed domains #######################################################################################

removed_domains_lst = [
    {'circle': {'center': (0.3, 0.3), 'radius': 0.1}},
    {'circle': {'center': (-0.3, 0.3), 'radius': 0.1}},
    {'circle': {'center': (0.3, -0.3), 'radius': 0.1}},
    {'circle': {'center': (-0.3, -0.3), 'radius': 0.1}}
]

# CSG boundaries #######################################################################################################

boundaries.dirichlet({'circle': {'center': (0.3, 0.3), 'radius': 0.1}}, value=0)
boundaries.dirichlet({'circle': {'center': (-0.3, 0.3), 'radius': 0.1}}, value=0)
boundaries.dirichlet({'circle': {'center': (0.3, -0.3), 'radius': 0.1}}, value=0)
boundaries.dirichlet({'circle': {'center': (-0.3, -0.3), 'radius': 0.1}}, value=0)

# Non CSG boundaries ###################################################################################################

boundaries.dirichlet({'x': x_min, 'y': [y_min, y_max]}, value=1)
boundaries.dirichlet({'x': x_max, 'y': [y_min, y_max]}, value=1)
boundaries.dirichlet({'x': [x_min, x_max], 'y': y_min}, value=1)
boundaries.dirichlet({'x': [x_min, x_max], 'y': y_max}, value=1)

equation = Equation()

# Operator: -d2u/dx2 - d2u/dy2 = 0

poisson = {
    '-d2u/dx2':
        {
            'coeff': -1.,
            'term': [0, 0],
            'pow': 1,
            'var': 0
        },
    '-d2u/dy2':
        {
            'coeff': -1.,
            'term': [1, 1],
            'pow': 1,
            'var': 0
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

img_dir = os.path.join(os.path.dirname(__file__), 'poisson_2d_classic_img.py')

cb_cache = cache.Cache(cache_verbose=True, model_randomize_parameter=1e-5)

cb_es = early_stopping.EarlyStopping(eps=1e-9,
                                     loss_window=100,
                                     no_improvement_patience=1000,
                                     patience=5,
                                     info_string_every=10,
                                     randomize_parameter=1e-5)

cb_plots = plot.Plots(save_every=500,
                      print_every=None,
                      img_dir=img_dir,
                      img_dim='2d_scatter')  # 3 image dimension options: 3d, 2d, 2d_scatter

optimizer = Optimizer('Adam', {'lr': 1e-3})

model.train(optimizer, 1e5, save_model=True, callbacks=[cb_cache, cb_es, cb_plots])





