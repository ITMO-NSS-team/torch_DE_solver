import torch
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../examples_navier_stokes')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device

solver_device('gpu')

k = 1
ro = 1
mu = 2 * torch.pi / 100

x_min, x_max = 0, 2 * torch.pi
y_min, y_max = 0, 2 * torch.pi
t_max = 2
grid_res = 10

domain = Domain()
domain.variable('x', [x_min, x_max], grid_res)
domain.variable('y', [y_min, y_max], grid_res)
domain.variable('t', [0, t_max], grid_res)

x = domain.variable_dict['x']
y = domain.variable_dict['y']
t = domain.variable_dict['t']

boundaries = Conditions()

# Initial conditions (TG vortex) ###################################################################################

boundaries.dirichlet({'x': [x_min, x_max], 'y': [y_min, y_max], 't': 0},
                     value=torch.sin(k * x) * torch.cos(k * y),
                     var=0)
boundaries.dirichlet({'x': [x_min, x_max], 'y': [y_min, y_max], 't': 0},
                     value=-torch.cos(k * x) * torch.sin(k * y),
                     var=1)
boundaries.dirichlet({'x': [x_min, x_max], 'y': [y_min, y_max], 't': 0},
                     value=-ro / 4 * (torch.cos(2 * k * x) + torch.cos(2 * k * y)),
                     var=2)

# Boundary conditions for u-function ###############################################################################

# u(x_min, y, t) = sin(x)
boundaries.dirichlet({'x': x_min, 'y': [y_min, y_max], 't': [0, t_max]}, value=torch.sin(x), var=0)
# u(x_max, y, t) = -sin(x)
boundaries.dirichlet({'x': x_max, 'y': [y_min, y_max], 't': [0, t_max]}, value=-torch.sin(x), var=0)
# u(x, y_min, t) = 0
boundaries.dirichlet({'x': [x_min, x_max], 'y': y_min, 't': [0, t_max]}, value=0, var=0)
# u(x, y_max, t) = 0
boundaries.dirichlet({'x': [x_min, x_max], 'y': y_max, 't': [0, t_max]}, value=0, var=0)

# Boundary conditions for v-function ###############################################################################

# v(x_min, y, t) = 0
boundaries.dirichlet({'x': x_min, 'y': [y_min, y_max], 't': [0, t_max]}, value=0, var=1)
# v(x_max, y, t) = 0
boundaries.dirichlet({'x': x_max, 'y': [y_min, y_max], 't': [0, t_max]}, value=0, var=1)
# v(x, y_min, t) = -sin(y)
boundaries.dirichlet({'x': [x_min, x_max], 'y': y_min, 't': [0, t_max]}, value=-torch.sin(y), var=1)
# v(x, y_max, t) = sin(y)
boundaries.dirichlet({'x': [x_min, x_max], 'y': y_max, 't': [0, t_max]}, value=torch.sin(y), var=1)

# Boundary conditions for p-function ###############################################################################

# p(x, y_max, t) = 0
boundaries.dirichlet({'x': x_min, 'y': [y_min, y_max], 't': [0, t_max]}, value=0, var=2)
# v(x, y_max, t) = 0
boundaries.dirichlet({'x': x_max, 'y': [y_min, y_max], 't': [0, t_max]}, value=0, var=2)

equation = Equation()

# operator 1: # operator: u_x + v_y = 0
NS_1 = {
    'du/dx':
        {
            'coeff': 1,
            'term': [0],
            'pow': 1,
            'var': 0
        },
    'dv/dy':
        {
            'coeff': 1,
            'term': [1],
            'pow': 1,
            'var': 1
        }
}

# operator 2: u * u_x + v * u_y + 1 / ro * p_x - mu * (u_xx + u_yy) = 0
NS_2 = {
    'u * du/dx':
        {
            'coeff': 1,
            'term': [[None], [0]],
            'pow': [1, 1],
            'var': [0, 0]
        },
    'v * du/dy':
        {
            'coeff': 1,
            'term': [[None], [1]],
            'pow': [1, 1],
            'var': [1, 0]
        },
    '1/ro * dp/dx':
        {
            'coeff': 1 / ro,
            'term': [0],
            'pow': 1,
            'var': 2
        },
    '-mu * d2u/dx2':
        {
            'coeff': -mu,
            'term': [0, 0],
            'pow': 1,
            'var': 0
        },
    '-mu * d2u/dy2':
        {
            'coeff': -mu,
            'term': [1, 1],
            'pow': 1,
            'var': 0
        }
}

# operator 3: u * v_x + v * v_y + 1 / ro * p_y - mu * (v_xx + v_yy) = 0
NS_3 = {
    'u * dv/dx':
        {
            'coeff': 1,
            'term': [[None], [0]],
            'pow': [1, 1],
            'var': [0, 1]
        },
    'v * dv/dy':
        {
            'coeff': 1,
            'term': [[None], [1]],
            'pow': [1, 1],
            'var': [1, 1]
        },
    '1/ro * dp/dy':
        {
            'coeff': 1 / ro,
            'term': [1],
            'pow': 1,
            'var': 2
        },
    '-mu * d2v/dx2':
        {
            'coeff': -mu,
            'term': [0, 0],
            'pow': 1,
            'var': 1
        },
    '-mu * d2v/dy2':
        {
            'coeff': -mu,
            'term': [1, 1],
            'pow': 1,
            'var': 1
        }
}

equation.add(NS_1)
equation.add(NS_2)
equation.add(NS_3)

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
    torch.nn.Linear(neurons, 3)
)

for m in net.modules():
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)

model = Model(net, domain, equation, boundaries)

model.compile('autograd', lambda_operator=1, lambda_bound=1000, tol=0.1)

cb_es = early_stopping.EarlyStopping(eps=1e-5,
                                     loss_window=100,
                                     no_improvement_patience=1000,
                                     patience=10,
                                     info_string_every=5,
                                     randomize_parameter=1e-5)

img_dir = os.path.join(os.path.dirname(__file__), 'navier_stokes_2d_TG_vortex_img')

cb_plots = plot.Plots(save_every=100,
                      print_every=None,
                      img_dir=img_dir,
                      img_dim='2d',
                      scatter_flag=False,
                      n_samples=4,
                      plot_axes=[0, 1],
                      fixed_axes=[2],
                      var_transpose=False,
                      figsize=(18, 6))

optimizer = Optimizer('Adam', {'lr': 1e-5})

model.train(optimizer, 1e6, save_model=True, callbacks=[cb_es, cb_plots])
