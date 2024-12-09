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


x_min, x_max = -10, 10
y_min, y_max = -10, 10
grid_res = 50

domain = Domain()
domain.variable('x', [x_min, x_max], grid_res)
domain.variable('y', [y_min, y_max], grid_res)

x = domain.variable_dict['x']
y = domain.variable_dict['y']

split = (5, 5)
freq = 2
block_size = np.array([(x_max - x_min + 2e-5) / split[0], (y_max - y_min + 2e-5) / split[1]])

a_cof = np.loadtxt("poisson_a_coef.dat")
f_cof = np.loadtxt("poisson_f_coef.dat").reshape(split[0], split[1], freq, freq)


boundaries = Conditions()

# Operator: u + du/dn = 0

bop_x_min = {
    'u':
        {
            'coeff': 1,
            'term': [None],
            'pow': 1
        },
    'du/dn':
        {
            'coeff': -1,
            'term': [0],
            'pow': 1,
            'var': 0
        }
}
boundaries.robin({'x': x_min, 'y': [y_min, y_max]}, operator=bop_x_min, value=0)
bop_x_max = {
    'u':
        {
            'coeff': 1,
            'term': [None],
            'pow': 1
        },
    'du/dn':
        {
            'coeff': 1,
            'term': [0],
            'pow': 1,
            'var': 0
        }
}
boundaries.robin({'x': x_max, 'y': [y_min, y_max]}, operator=bop_x_max, value=0)

bop_y_min = {
    'u':
        {
            'coeff': 1,
            'term': [None],
            'pow': 1
        },
    'du/dn':
        {
            'coeff': -1,
            'term': [1],
            'pow': 1,
            'var': 0
        }
}
boundaries.robin({'x': [x_min, x_max], 'y': y_min}, operator=bop_y_min, value=0)
bop_y_max = {
    'u':
        {
            'coeff': 1,
            'term': [None],
            'pow': 1
        },
    'du/dn':
        {
            'coeff': 1,
            'term': [1],
            'pow': 1,
            'var': 0
        }
}
boundaries.robin({'x': [x_min, x_max], 'y': y_max}, operator=bop_y_max, value=-y)


def compute_domain(grid):
    reduced_x = (grid - np.array([x_min, y_min]) + 1e-5)
    dom = np.floor(reduced_x / block_size).astype("int32")
    res = reduced_x - dom * block_size
    return dom, res


def compute_a_coeff(grid):
    dom, _ = compute_domain(grid)
    return a_cof[dom[0], dom[1]]


a_coeff = np.vectorize(compute_a_coeff, signature="(2)->()")


def compute_forcing_term(grid):
    dom, res = compute_domain(grid)

    def f_fn(coef):
        ans = coef[0, 0]
        for i in range(coef.shape[0]):
            for j in range(coef.shape[1]):
                tmp = np.sin(np.pi * np.array((i, j)) * (res / block_size))
                ans += coef[i, j] * tmp[0] * tmp[1]
        return ans

    return f_fn(f_cof[dom[0], dom[1]])


forcing_term = np.vectorize(compute_forcing_term, signature="(2)->()")


def get_a_coeff(x):
    x = x.detach().cpu()
    return torch.Tensor(a_coeff(x)).unsqueeze(dim=-1)


def get_forcing_term(x):
    x = x.detach().cpu()
    return torch.Tensor(forcing_term(x)).unsqueeze(dim=-1)


equation = Equation()

# Operator: −∇(a(x)∇u) = f(x, y)

poisson = {
    'a * d2u/dx2':
        {
            'coeff': get_a_coeff,
            'term': [0, 0],
            'pow': 1,
        },
    'a * d2u/dy2':
        {
            'coeff': get_a_coeff,
            'term': [1, 1],
            'pow': 1,
        },
    'f(x, y)':
        {
            'coeff': get_forcing_term,
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

img_dir = os.path.join(os.path.dirname(__file__), 'poisson_2d_many_subdomains_img')

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
