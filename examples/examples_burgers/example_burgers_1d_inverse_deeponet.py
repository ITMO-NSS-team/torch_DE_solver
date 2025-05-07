import torch
import numpy as np
import scipy
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import early_stopping, plot, inverse_task
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.models import parameter_registr

from tedeous.models import PI_DeepONet, MLP
from tedeous.data import build_deeponet

solver_device('cuda')

neurons = 60
m = neurons
P = neurons
Q = neurons
N = neurons * 2

domain = Domain(method='PI_DeepONet', N=N, m=m, P=P, Q=Q)
domain.variable('x', [0, 1], m, dtype='float32')
domain.variable('t', [0, 1], m, dtype='float32')

torch.manual_seed(0)
key = torch.randint(0, 10 ** 10, (1,), dtype=torch.int64)

train_data = build_deeponet(key, N, m, P, Q)

u_branch = train_data[0][0]
s_ic = train_data[0][2]
s_bc = train_data[1][2]

boundaries = Conditions(method='PI_deepONet', N=N, m=m, P=P, Q=Q)

data = scipy.io.loadmat(os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'wolfram_sln/Burgers.mat')))

x = torch.tensor(data['x']).reshape(-1)
t = torch.tensor(data['t']).reshape(-1)

usol = data['usol']

bnd1 = torch.cartesian_prod(x, t).float()
bndval1 = torch.tensor(usol).reshape(-1, 1)

id_f = np.random.choice(len(bnd1), 2000, replace=False)

bnd1 = bnd1[id_f]
bndval1 = bndval1[id_f]

boundaries.data(bnd=bnd1, operator=None, value=bndval1)

# NN mode
neurons = 60

branch_layers = [m, neurons, neurons, neurons, neurons]
trunk_layers = [2, neurons, neurons, neurons, 1]

# KAN mode
splines = 20

# branch with custom MLP
branch_net = MLP(branch_layers, activation=torch.tanh)

# # branch with torch.nn.Sequential
# branch_net = torch.nn.Sequential(
#     torch.nn.Linear(m, neurons),
#     torch.nn.Tanh(),
#     torch.nn.Linear(neurons, neurons),
#     torch.nn.Tanh(),
#     torch.nn.Linear(neurons, neurons),
#     torch.nn.Tanh(),
#     torch.nn.Linear(neurons, neurons),
#     torch.nn.Tanh(),
#     torch.nn.Linear(neurons, neurons)
# )
# for m in branch_net.modules():
#     if isinstance(m, torch.nn.Linear):
#         torch.nn.init.xavier_normal_(m.weight)
#         torch.nn.init.zeros_(m.bias)

# # branch with EfficientKAN
# branch_net = efficient_kan.KAN(
#     [grid_res, splines, 1],
#     grid_size=10,
#     spline_order=3,
#     scale_noise=0.1,
#     scale_base=1.0,
#     scale_spline=1.0,
#     base_activation=torch.nn.Tanh,
#     grid_eps=0.02,
#     grid_range=[-1, 1]
# )

# # branch with FastKAN
# branch_net = fastkan.FastKAN(
#     [grid_res, splines, splines, splines, splines],
#     grid_min=-5.,
#     grid_max=5.,
#     num_grids=2,
#     use_base_update=True,
#     base_activation=F.tanh,
#     spline_weight_init_scale=0.05
# )

# trunk with custom MLP
trunk_net = MLP(trunk_layers, activation=torch.tanh)

# # trunk with torch.nn.Sequential
# trunk_net = torch.nn.Sequential(
#     torch.nn.Linear(2, neurons),
#     torch.nn.Tanh(),
#     torch.nn.Linear(neurons, neurons),
#     torch.nn.Tanh(),
#     torch.nn.Linear(neurons, neurons),
#     torch.nn.Tanh(),
#     torch.nn.Linear(neurons, neurons),
#     torch.nn.Tanh(),
#     torch.nn.Linear(neurons, 1)
# )
# for m in trunk_net.modules():
#     if isinstance(m, torch.nn.Linear):
#         torch.nn.init.xavier_normal_(m.weight)
#         torch.nn.init.zeros_(m.bias)

# # trunk with EfficientKAN
# trunk_net = efficient_kan.KAN(
#     [2, splines, splines, splines, 1],
#     grid_size=10,
#     spline_order=3,
#     scale_noise=0.1,
#     scale_base=1.0,
#     scale_spline=1.0,
#     base_activation=torch.nn.Tanh,
#     grid_eps=0.02,
#     grid_range=[-1, 1]
# )

# # trunk with FastKAN
# trunk_net = fastkan.FastKAN(
#     [2, splines, splines, splines, splines],
#     grid_min=-5.,
#     grid_max=5.,
#     num_grids=2,
#     use_base_update=True,
#     base_activation=F.tanh,
#     spline_weight_init_scale=0.05
# )

net = PI_DeepONet(branch_net=branch_net, trunk_net=trunk_net)
# net = MLP(trunk_layers, activation=F.tanh)

print(f"neurons = {neurons}")
print(f"u_branch = {u_branch}")

net = PI_DeepONet(branch_net=branch_net, trunk_net=trunk_net)
# net = MLP(trunk_layers, activation=F.tanh)

print(f"neurons_branch = {neurons_branch}\nneurons_trunk = {neurons_trunk}")
print(f"u_branch = {u_branch}")

parameters = {'lam1': 2., 'lam2': 0.2}  # true parameters: lam1 = 1, lam2 = -0.01*pi

parameter_registr(net, parameters)

equation = Equation()

burgers_eq = {
    'du/dt**1':
        {
            'coeff': 1.,
            'du/dt': [1],
            'pow': 1,
            'var': 0
        },
    '+u*du/dx':
        {
            'coeff': net.lam1,
            'u*du/dx': [[None], [0]],
            'pow': [1, 1],
            'var': [0, 0]
        },
    '-mu*d2u/dx2':
        {
            'coeff': net.lam2,
            'd2u/dx2': [0, 0],
            'pow': 1,
            'var': 0
        }
}

equation.add(burgers_eq)

model = Model(net, domain, equation, boundaries)

model.compile('autograd', lambda_operator=1, lambda_bound=100)

img_dir = os.path.join(os.path.dirname( __file__ ), 'burgers_eq_img_deeponet')

cb_es = early_stopping.EarlyStopping(eps=1e-7,
                                     loss_window=100,
                                     no_improvement_patience=1000,
                                     patience=3,
                                     abs_loss=1e-5,
                                     randomize_parameter=1e-5,
                                     info_string_every=1)

cb_plots = plot.Plots(save_every=500, print_every=None, img_dir=img_dir)

cb_params = inverse_task.InverseTask(parameters=parameters, info_string_every=500)

optimizer = Optimizer('Adam', {'lr': 1e-4})

model.train(optimizer, 25e3, save_model=False, callbacks=[cb_es, cb_plots, cb_params])