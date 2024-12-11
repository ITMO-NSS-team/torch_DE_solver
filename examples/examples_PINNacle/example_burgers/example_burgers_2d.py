import torch
import numpy as np
import time
import scipy
import pandas as pd
from scipy.integrate import quad
import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import cache, early_stopping
from tedeous.optimizers.optimizer import Optimizer
from tedeous.callbacks.plot import Plots
from tedeous.utils import ic_data


def init_w(x, y, size, L):
    a = torch.randn(size, size)
    b = torch.randn(size, size)

    result = torch.zeros(size)
    for i in range(-L, L + 1):
        for j in range(-L, L + 1):
            new_component = a[i, j] * torch.sin(2 * torch.pi * (i * x + j * y)) \
                      + b[i, j] * torch.cos(2 * torch.pi * (i * x + j * y))
            result += new_component

    return result


def init_u(grid):
    x, y = grid[:, 0], grid[:, 1]
    size = int(len(x) ** 1)
    L = int(torch.max(x))
    c_u = torch.randn(size)
    return 2 * init_w(x, y, size, L) + c_u


def init_v(grid):
    x, y = grid[:, 0], grid[:, 1]
    size = int(len(x) ** 1)
    L = int(torch.max(x))
    c_v = torch.randn(size)
    return 2 * init_w(x, y, size, L) + c_v


def init_ics():
    datapath = "burgers2d_0.dat"
    ic_path = ("burgers2d_init_u_0.dat", "burgers2d_init_v_0.dat")

    t_transpose = True

    input_dim = 3
    output_dim = 2

    def trans_time_data_to_dataset(datapath, ref_data):
        data = ref_data
        slice = (data.shape[1] - input_dim + 1) // output_dim
        assert slice * output_dim == data.shape[
            1] - input_dim + 1, "Data shape is not multiple of pde.output_dim"

        with open(datapath, "r", encoding='utf-8') as f:
            def extract_time(string):
                index = string.find("t=")
                if index == -1:
                    return None
                return float(string[index + 2:].split(' ')[0])

            t = None
            for line in f.readlines():
                if line.startswith('%') and line.count('@') == slice * output_dim:
                    t = line.split('@')[1:]
                    t = list(map(extract_time, t))
            if t is None or None in t:
                raise ValueError("Reference Data not in Comsol format or does not contain time info")
            t = np.array(t[::output_dim])

        t, x0 = np.meshgrid(t, data[:, 0])
        list_x = [x0.reshape(-1)]
        for i in range(1, input_dim - 1):
            list_x.append(np.stack([data[:, i] for _ in range(slice)]).T.reshape(-1))
        list_x.append(t.reshape(-1))
        for i in range(output_dim):
            list_x.append(data[:, input_dim - 1 + i::output_dim].reshape(-1))
        return np.stack(list_x).T

    scale = 1

    def transform_fn(data):
        data[:, :input_dim] *= scale
        return data

    def load_ref_data(datapath, transform_fn=None, t_transpose=False):
        ref_data = np.loadtxt(datapath, comments="%", encoding='utf-8').astype(np.float32)
        if t_transpose:
            ref_data = trans_time_data_to_dataset(datapath, ref_data)
        if transform_fn is not None:
            ref_data = transform_fn(ref_data)
            return ref_data

    load_ref_data(datapath)
    load_ref_data(datapath, transform_fn, t_transpose)

    ics = (np.loadtxt(ic_path[0]), np.loadtxt(ic_path[1]))

    return ics


def ic_func(x, y, t, component=0):
    grid = torch.cartesian_prod(x, y)
    ics = init_ics()
    result = scipy.interpolate.LinearNDInterpolator(ics[component][:, :2], ics[component][:, 2:])(grid[:, :2])
    return result


mu = 0.001
u_component = 0
v_component = 1
L = 4
T = 1
grid_res = 10

domain = Domain()
domain.variable('x', [0, L], grid_res)
domain.variable('y', [0, L], grid_res)
domain.variable('t', [0, T], grid_res)

x = domain.variable_dict['x']
y = domain.variable_dict['y']
t = domain.variable_dict['t']

# init_u = (x + y - 2 * x * t) / (1 - 2 * t ** 2)
# init_v = (x - y - 2 * y * t) / (1 - 2 * t ** 2)

# init_u = torch.sin(2 * torch.pi * x) * torch.sin(2 * torch.pi * y)
# init_v = torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

boundaries = Conditions()

# Initial conditions ###############################################################################################

# # u(x, y, 0)
# boundaries.dirichlet({'x': [0, L], 'y': [0, L], 't': 0}, value=init_u, var=0)  # with burgers.dat file
#
# # v(x, y, 0)
# boundaries.dirichlet({'x': [0, L], 'y': [0, L], 't': 0}, value=init_v, var=1)

# u(x, y, 0)
boundaries.dirichlet({'x': [0, L], 'y': [0, L], 't': 0}, value=ic_data(x, y, t, component=u_component), var=0)

# v(x, y, 0)
boundaries.dirichlet({'x': [0, L], 'y': [0, L], 't': 0}, value=ic_data(x, y, t, component=v_component), var=1)

# Boundary conditions ##############################################################################################

# u(0, y, t) = u(L, y, t)
boundaries.periodic([{'x': 0, 'y': [0, L], 't': [0, T]}, {'x': L, 'y': [0, L], 't': [0, T]}], var=0)

# u(x, 0, t) = u(x, L, t)
boundaries.periodic([{'x': [0, L], 'y': 0, 't': [0, T]}, {'x': [0, L], 'y': L, 't': [0, T]}], var=0)

# v(0, y, t) = v(L, y, t)
boundaries.periodic([{'x': 0, 'y': [0, L], 't': [0, T]}, {'x': L, 'y': [0, L], 't': [0, T]}], var=1)

# v(x, 0, t) = v(x, L, t)
boundaries.periodic([{'x': [0, L], 'y': 0, 't': [0, T]}, {'x': [0, L], 'y': L, 't': [0, T]}], var=1)

equation = Equation()

# operator: u_t + u * u_x + v * u_y - mu * (u_xx + u_yy) = 0
burgers_u = {
    'du/dt**1':
        {
            'coeff': 1.,
            'du/dt': [2],
            'pow': 1,
            'var': 0
        },
    '+u*du/dx':
        {
            'coeff': 1.,
            'u*du/dx': [[None], [0]],
            'pow': [1, 1],
            'var': [0, 0]
        },
    '+v*du/dy':
        {
            'coeff': 1.,
            'u*du/dy': [[None], [1]],
            'pow': [1, 1],
            'var': [1, 0]
        },
    '-mu*d2u/dx2':
        {
            'coeff': -mu,
            'd2u/dx2': [0, 0],
            'pow': 1,
            'var': 0
        },
    '-mu*d2u/dy2':
        {
            'coeff': -mu,
            'd2u/dy2': [1, 1],
            'pow': 1,
            'var': 0
        }
}

# operator: v_t + u * v_x + v * v_y - mu * (v_xx + v_yy) = 0

burgers_v = {
    'dv/dt**1':
        {
            'coeff': 1.,
            'dv/dt': [2],
            'pow': 1,
            'var': 1
        },
    '+u*dv/dx':
        {
            'coeff': 1.,
            'u*dv/dx': [[None], [0]],
            'pow': [1, 1],
            'var': [0, 1]
        },
    '+v*dv/dy':
        {
            'coeff': 1.,
            'v*dv/dy': [[None], [1]],
            'pow': [1, 1],
            'var': [1, 1]
        },
    '-mu*d2v/dx2':
        {
            'coeff': -mu,
            'd2v/dx2': [0, 0],
            'pow': 1,
            'var': 1
        },
    '-mu*d2v/dy2':
        {
            'coeff': -mu,
            'd2v/dy2': [1, 1],
            'pow': 1,
            'var': 1
        }
}

equation.add(burgers_u)
equation.add(burgers_v)

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
    torch.nn.Linear(neurons, 2)
)

model = Model(net, domain, equation, boundaries)

model.compile('autograd', lambda_operator=1, lambda_bound=100)

cb_cache = cache.Cache(cache_verbose=False, model_randomize_parameter=1e-5)

cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                     randomize_parameter=1e-5,
                                     info_string_every=10)

img_dir = os.path.join(os.path.dirname(__file__), 'burgers_2d_img')

cb_plots = Plots(save_every=100,
                 print_every=None,
                 img_dir=img_dir,
                 img_dim='2d')

optimizer = Optimizer('Adam', {'lr': 5e-3})

model.train(optimizer, 5e6, save_model=True, callbacks=[cb_es, cb_plots, cb_cache])







