# Lotka-Volterra equations also known as predator-prey equations, describe the variation in populations
# of two species which interact via predation.
# For example, wolves (predators) and deer (prey).
# This is a classical model to represent the dynamic of two populations.

# Let αlpha > 0, beta > 0, delta > 0 and gamma > 0 . The system is given by

# dx/dt = x(alpha-beta*y)
# dy/dt = y(-delta+gamma*x)

# Where 'x' represents prey population and 'y' predators population.
# It’s a system of first-order ordinary differential equations.
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import time
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../examples_lotka_volterra')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device, check_device, device_type

from tedeous.models import PI_DeepONet, MLP
from tedeous.data import build_deeponet

solver_device('сpu')

alpha = 0.55  # 0.55 (20.)
beta = 0.028  # 0.028 (20.)
delta = 0.84  # 0.026 (20.)
gamma = 0.026  # 0.84 (20.)
x0 = 30.  # 30 (4.)
y0 = 4.  # 4 (2.)
t0 = 0.
tmax = 20.  # 20 (1.)
Nt = 300

N = 10 ** 2
m = 10
P = 10
Q = 10

torch.manual_seed(0)
key = torch.randint(0, 10 ** 10, (1,), dtype=torch.int64)
train_data = build_deeponet(key, N, m, P, Q)

u_branch = train_data[0][0]
s_ic, s_bc = train_data[0][2], train_data[1][2]

domain = Domain(method='PI_DeepONet')

domain.variable('t', [t0, tmax], Nt)

h = 0.0001

# initial conditions
boundaries = Conditions(method='PI_DeepONet')
boundaries.dirichlet({'t': 0}, value=x0, var=0)
boundaries.dirichlet({'t': 0}, value=y0, var=1)

# equation system
# eq1: dx/dt = x(alpha-beta*y)
# eq2: dy/dt = y(-delta+gamma*x)

# x var: 0
# y var:1

equation = Equation()

eq1 = {
    'dx/dt': {
        'coeff': 1,
        'term': [0],
        'pow': 1,
        'var': [0]
    },
    '-x*alpha': {
        'coeff': -alpha,
        'term': [None],
        'pow': 1,
        'var': [0]
    },
    '+beta*x*y': {
        'coeff': beta,
        'term': [[None], [None]],
        'pow': [1, 1],
        'var': [0, 1]
    }
}

eq2 = {
    'dy/dt': {
        'coeff': 1,
        'term': [0],
        'pow': 1,
        'var': [1]
    },
    '+y*delta': {
        'coeff': delta,
        'term': [None],
        'pow': 1,
        'var': [1]
    },
    '-gamma*x*y': {
        'coeff': -gamma,
        'term': [[None], [None]],
        'pow': [1, 1],
        'var': [0, 1]
    }
}

equation.add(eq1)
equation.add(eq2)

# net = torch.nn.Sequential(
#     torch.nn.Linear(1, 100),
#     torch.nn.Tanh(),
#     torch.nn.Linear(100, 100),
#     torch.nn.Tanh(),
#     torch.nn.Linear(100, 100),
#     torch.nn.Tanh(),
#     torch.nn.Linear(100, 2)
# )

# NN mode
neurons_branch = 100
neurons_trunk = 100
branch_layers = [10, neurons_branch, neurons_branch, neurons_branch, neurons_branch]
trunk_layers = [1, neurons_trunk, neurons_trunk, neurons_trunk, neurons_trunk]

# KAN mode
splines = 100

# # branch with custom MLP
# branch_net = MLP(branch_layers, activation=torch.tanh)

# # branch with torch.nn.Sequential
branch_net = torch.nn.Sequential(
    torch.nn.Linear(10, neurons_branch),
    torch.nn.Tanh(),
    torch.nn.Linear(neurons_branch, neurons_branch),
    torch.nn.Tanh(),
    torch.nn.Linear(neurons_branch, neurons_branch),
    torch.nn.Tanh(),
    torch.nn.Linear(neurons_branch, neurons_branch),
    torch.nn.Tanh(),
    torch.nn.Linear(neurons_branch, neurons_branch)
)
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
#     [grid_res, splines, splines, splines, 1],
#     grid_min=-1.,
#     grid_max=1.,
#     num_grids=2,
#     use_base_update=True,
#     base_activation=F.tanh,
#     spline_weight_init_scale=0.05
# )

# # trunk with custom MLP
# trunk_net = MLP(trunk_layers, activation=torch.tanh)

# # trunk with torch.nn.Sequential
trunk_net = torch.nn.Sequential(
    torch.nn.Linear(1, neurons_trunk),
    torch.nn.Tanh(),
    torch.nn.Linear(neurons_trunk, neurons_trunk),
    torch.nn.Tanh(),
    torch.nn.Linear(neurons_trunk, neurons_trunk),
    torch.nn.Tanh(),
    torch.nn.Linear(neurons_trunk, neurons_trunk),
    torch.nn.Tanh(),
    torch.nn.Linear(neurons_trunk, neurons_trunk)
)
# for m in trunk_net.modules():
#     if isinstance(m, torch.nn.Linear):
#         torch.nn.init.xavier_normal_(m.weight)
#         torch.nn.init.zeros_(m.bias)

# normal initialization
# nn.init.xavier_normal_(layer.weight)  # best for PI DeepOnet
# nn.init.kaiming_normal_(layer.weight)
# nn.init.trunc_normal_(layer.weight)

#  uniform initialization
# nn.init.xavier_uniform_(layer.weight)
# nn.init.kaiming_uniform_(layer.weight)

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
#     [2, splines, splines, splines, 1],
#     grid_min=-1.,
#     grid_max=1.,
#     num_grids=2,
#     use_base_update=True,
#     base_activation=F.tanh,
#     spline_weight_init_scale=0.05
# )

net = PI_DeepONet(branch_net=branch_net, trunk_net=trunk_net)
# net = MLP(trunk_layers, activation=F.tanh)

model = Model(net, domain, equation, boundaries, method='PI_DeepONet', u=u_branch)

model.compile("NN", lambda_operator=1, lambda_bound=100, h=h)

img_dir = os.path.join(os.path.dirname(__file__), 'img_Lotka_Volterra_deeponet')

start = time.time()

cb_cache = cache.Cache(cache_verbose=True, model_randomize_parameter=1e-5)

cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                     loss_window=100,
                                     no_improvement_patience=1000,
                                     patience=5,
                                     randomize_parameter=1e-5,
                                     info_string_every=100)

cb_plots = plot.Plots(save_every=1000, print_every=1000, img_dir=img_dir, method='PI_DeepONet', u=u_branch)

optimizer = Optimizer('Adam', {'lr': 1e-4})

# model.train(optimizer, 5e6, save_model=True, callbacks=[cb_es, cb_cache, cb_plots])

end = time.time()

print('Time taken = {}'.format(end - start))


# scipy.integrate solution of Lotka_Volterra equations and comparison with NN results

def deriv(X, t, alpha, beta, delta, gamma):
    x, y = X
    dotx = x * (alpha - beta * y)
    doty = y * (-delta + gamma * x)
    return np.array([dotx, doty])


t = np.linspace(0., tmax, Nt)

X0 = [x0, y0]
res = integrate.odeint(deriv, X0, t, args=(alpha, beta, delta, gamma))
x, y = res.T

grid = domain.build('NN')

plt.figure()
plt.grid()
plt.title("odeint and NN methods comparing")
plt.plot(t, x, '+', label='preys_odeint')
plt.plot(t, y, '*', label="predators_odeint")
plt.plot(grid, net(u_branch, grid)[:, 0].detach().numpy().reshape(-1), label='preys_NN')
plt.plot(grid, net(u_branch, grid)[:, 1].detach().numpy().reshape(-1), label='predators_NN')
plt.xlabel('Time t, [days]')
plt.ylabel('Population')
plt.legend(loc='upper right')
plt.show()

plt.figure()
plt.grid()
plt.title('Phase plane: prey vs predators')
plt.plot(net(u_branch, grid)[:, 0].detach().numpy().reshape(-1),
         net(u_branch, grid)[:, 1].detach().numpy().reshape(-1), '-*', label='NN')
plt.plot(x, y, label='odeint')
plt.xlabel('preys')
plt.ylabel('predators')
plt.legend()
plt.show()
