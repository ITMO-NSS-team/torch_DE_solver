import torch
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device

solver_device('cuda')

ro = 1
mu = 0.01

x_min, x_max = 0, 4
y_min, y_max = 0, 2
grid_res = 20

# Rectangle type of removed domains
removed_domains_lst = [
    {'rectangle': {'coords_min': [0, 1], 'coords_max': [2, 2]}}
]

domain = Domain(complex_geometry_flag=True)
domain.variable('x', [x_min, x_max], grid_res)
domain.variable('y', [y_min, y_max], grid_res)

y = domain.variable_dict['y']
first_one_index = int(torch.where(y == 1.0)[0][0])
y = y[:first_one_index + 1]

boundaries = Conditions()

# Inlet boundary condition u_in = 4y(1 - y) ############################################################################

# u(x_min, y) = f(y)
boundaries.dirichlet({'x': x_min, 'y': [y_min, 1]}, value=4 * y * (1 - y), var=0)

# No-slip boundary condition u = 0 #####################################################################################

# u(x_max, y) = 0
boundaries.dirichlet({'x': x_max, 'y': [y_min, y_max]}, value=0, var=0)
# u(x, y_min) = 0
boundaries.dirichlet({'x': [x_min, x_max], 'y': y_min}, value=0, var=0)
# u(x, y_max) = 0
boundaries.dirichlet({'x': [2, x_max], 'y': y_max}, value=0, var=0)
# u(x_rec, y_rec_min) = 0
boundaries.dirichlet({'x': [x_min, 2], 'y': 1}, value=0, var=0)
# u(x_rec_max, y_rec) = 0
boundaries.dirichlet({'x': 2, 'y': [1, y_max]}, value=0, var=0)

# Outlet pressure condition (p = 0 at outlet) ##########################################################################

# p(x_max, y) = 0
boundaries.dirichlet({'x': x_max, 'y': [y_min, y_max]}, value=0, var=2)

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
    torch.nn.Linear(neurons, neurons),
    torch.nn.Tanh(),
    torch.nn.Linear(neurons, 3)
)

for m in net.modules():
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)

model = Model(net, domain, equation, boundaries)

model.compile('autograd', lambda_operator=1, lambda_bound=100, removed_domains=removed_domains_lst)

cb_es = early_stopping.EarlyStopping(eps=1e-5,
                                     loss_window=100,
                                     no_improvement_patience=1000,
                                     patience=10,
                                     info_string_every=5,
                                     randomize_parameter=1e-5)

img_dir = os.path.join(os.path.dirname(__file__), 'navier_stokes_2d_back_step_flow_img')

cb_plots = plot.Plots(save_every=100,
                      print_every=None,
                      img_dir=img_dir,
                      img_dim='2d_scatter')  # 3 image dimension options: 3d, 2d, 2d_scatter

optimizer = Optimizer('Adam', {'lr': 1e-4})

model.train(optimizer, 1e6, save_model=True, callbacks=[cb_es, cb_plots])
