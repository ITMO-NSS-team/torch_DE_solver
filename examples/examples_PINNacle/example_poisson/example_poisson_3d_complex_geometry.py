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

m1, m2, m3 = 1, 10, 5
mu_1, mu_2 = 1, 1
k1, k2 = 8, 10
A1, A2 = 20, 100

x_min, x_max = 0, 1
y_min, y_max = 0, 1
z_min, z_max = 0, 1
z_border = 0.5

grid_res = 20

domain = Domain()
domain.variable('x', [x_min, x_max], grid_res)
domain.variable('y', [y_min, y_max], grid_res)
domain.variable('z', [z_min, z_max], grid_res)

boundaries = Conditions()

# Circle type of removed domains #######################################################################################

removed_domains_lst = [
    {'circle': {'center': (0.4, 0.3, 0.6), 'radius': 0.2}},
    {'circle': {'center': (0.6, 0.7, 0.6), 'radius': 0.2}},
    {'circle': {'center': (0.2, 0.8, 0.7), 'radius': 0.1}},
    {'circle': {'center': (0.6, 0.2, 0.3), 'radius': 0.1}}
]

# # Non CSG boundaries #################################################################################################

# du/dn = 0

bop_x_min = {
    'du/dn':
        {
            'coeff': -1,
            'term': [0],
            'pow': 1,
            'var': 0
        }
}
boundaries.operator({'x': x_min, 'y': [y_min, y_max], 'z': [z_min, z_max]}, operator=bop_x_min, value=0)
bop_x_max = {
    'du/dn':
        {
            'coeff': 1,
            'term': [0],
            'pow': 1,
            'var': 0
        }
}
boundaries.operator({'x': x_max, 'y': [y_min, y_max], 'z': [z_min, z_max]}, operator=bop_x_max, value=0)

bop_y_min = {
    'du/dn':
        {
            'coeff': -1,
            'term': [1],
            'pow': 1,
            'var': 0
        }
}
boundaries.operator({'x': [x_min, x_max], 'y': y_min, 'z': [z_min, z_max]}, operator=bop_y_min, value=0)
bop_y_max = {
    'du/dn':
        {
            'coeff': 1,
            'term': [1],
            'pow': 1,
            'var': 0
        }
}
boundaries.operator({'x': [x_min, x_max], 'y': y_max, 'z': [z_min, z_max]}, operator=bop_y_max, value=0)

bop_z_min = {
    'du/dn':
        {
            'coeff': -1,
            'term': [2],
            'pow': 1,
            'var': 0
        }
}
boundaries.operator({'x': [x_min, x_max], 'y': [y_min, y_max], 'z': z_min}, operator=bop_z_min, value=0)
bop_z_max = {
    'du/dn':
        {
            'coeff': 1,
            'term': [2],
            'pow': 1,
            'var': 0
        }
}
boundaries.operator({'x': [x_min, x_max], 'y': [y_min, y_max], 'z': z_max}, operator=bop_z_max, value=0)


# CSG boundaries #######################################################################################################

# du/dn = 0

bop_x_R1 = {
    'du/dn':
        {
            'coeff': -1,
            'term': [0],
            'pow': 1,
            'var': 0
        }
}
boundaries.operator({'circle': {'center': (0.4, 0.3, 0.6), 'radius': 0.2}}, operator=bop_x_R1, value=0)
bop_y_R1 = {
    'du/dn':
        {
            'coeff': -1,
            'term': [1],
            'pow': 1,
            'var': 0
        }
}
boundaries.operator({'circle': {'center': (0.4, 0.3, 0.6), 'radius': 0.2}}, operator=bop_y_R1, value=0)
bop_z_R1 = {
    'du/dn':
        {
            'coeff': -1,
            'term': [2],
            'pow': 1,
            'var': 0
        }
}
boundaries.operator({'circle': {'center': (0.4, 0.3, 0.6), 'radius': 0.2}}, operator=bop_z_R1, value=0)

bop_x_R2 = {
    'du/dn':
        {
            'coeff': -1,
            'term': [0],
            'pow': 1,
            'var': 0
        }
}
boundaries.operator({'circle': {'center': (0.6, 0.7, 0.6), 'radius': 0.2}}, operator=bop_x_R2, value=0)
bop_y_R2 = {
    'du/dn':
        {
            'coeff': -1,
            'term': [1],
            'pow': 1,
            'var': 0
        }
}
boundaries.operator({'circle': {'center': (0.6, 0.7, 0.6), 'radius': 0.2}}, operator=bop_y_R2, value=0)
bop_z_R2 = {
    'du/dn':
        {
            'coeff': -1,
            'term': [2],
            'pow': 1,
            'var': 0
        }
}
boundaries.operator({'circle': {'center': (0.6, 0.7, 0.6), 'radius': 0.2}}, operator=bop_z_R2, value=0)

bop_x_R3 = {
    'du/dn':
        {
            'coeff': -1,
            'term': [0],
            'pow': 1,
            'var': 0
        }
}
boundaries.operator({'circle': {'center': (0.2, 0.8, 0.7), 'radius': 0.1}}, operator=bop_x_R3, value=0)
bop_y_R3 = {
    'du/dn':
        {
            'coeff': -1,
            'term': [1],
            'pow': 1,
            'var': 0
        }
}
boundaries.operator({'circle': {'center': (0.2, 0.8, 0.7), 'radius': 0.1}}, operator=bop_y_R3, value=0)
bop_z_R3 = {
    'du/dn':
        {
            'coeff': -1,
            'term': [2],
            'pow': 1,
            'var': 0
        }
}
boundaries.operator({'circle': {'center': (0.2, 0.8, 0.7), 'radius': 0.1}}, operator=bop_z_R3, value=0)

bop_x_R4 = {
    'du/dn':
        {
            'coeff': -1,
            'term': [0],
            'pow': 1,
            'var': 0
        }
}
boundaries.operator({'circle': {'center': (0.6, 0.2, 0.3), 'radius': 0.1}}, operator=bop_x_R4, value=0)
bop_y_R4 = {
    'du/dn':
        {
            'coeff': -1,
            'term': [1],
            'pow': 1,
            'var': 0
        }
}
boundaries.operator({'circle': {'center': (0.6, 0.2, 0.3), 'radius': 0.1}}, operator=bop_y_R4, value=0)
bop_z_R4 = {
    'du/dn':
        {
            'coeff': -1,
            'term': [2],
            'pow': 1,
            'var': 0
        }
}
boundaries.operator({'circle': {'center': (0.6, 0.2, 0.3), 'radius': 0.1}}, operator=bop_z_R4, value=0)


def forcing_term(grid):
    x, y, z = grid[:, 0], grid[:, 1], grid[:, 2]

    l = x ** 2 + y ** 2 + z ** 2

    addend_1 = torch.exp(torch.sin(m1 * torch.pi * x) +
                         torch.sin(m2 * torch.pi * y) +
                         torch.sin(m3 * torch.pi * z)) * (l - 1) / (l + 1)

    addend_2 = torch.sin(m1 * torch.pi * x) * \
               torch.sin(m2 * torch.pi * y) * \
               torch.sin(m3 * torch.pi * z)

    f = A1 * addend_1 + A2 * addend_2
    return -f


def select_mu(grid):
    z = grid[:, 2]
    return -torch.where(z < z_border, mu_1, mu_2).unsqueeze(dim=-1)


def select_k(grid):
    z = grid[:, 2]
    return -torch.where(z < z_border, k1 ** 2, k2 ** 2).unsqueeze(dim=-1)


equation = Equation()

# Operator: -mu * (d2u/dx2 + d2u/dy2 + d2u/dz2) - k ** 2 * u = f(x, y, z)

poisson = {
    '-d2u/dx2':
        {
            'coeff': select_mu,
            'term': [0, 0],
            'pow': 1,
        },
    '-d2u/dy2':
        {
            'coeff': select_mu,
            'term': [1, 1],
            'pow': 1,
        },
    '-d2u/dz2':
        {
            'coeff': select_mu,
            'term': [2, 2],
            'pow': 1,
        },
    'k ** 2 * u':
        {
            'coeff': select_k,
            'term': [None],
            'pow': 1
        },
    '-f(x, y, z)':
        {
            'coeff': forcing_term,
            'term': [None],
            'pow': 1
        }
}

equation.add(poisson)

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
    torch.nn.Linear(neurons, 1)
)

for m in net.modules():
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)

model = Model(net, domain, equation, boundaries)

model.compile('autograd', lambda_operator=1, lambda_bound=100, removed_domains=removed_domains_lst)

img_dir = os.path.join(os.path.dirname(__file__), 'poisson_3d_complex_geometry_img')

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

optimizer = Optimizer('Adam', {'lr': 1e-3})

model.train(optimizer, 1e5, save_model=True, callbacks=[cb_cache, cb_es, cb_plots])