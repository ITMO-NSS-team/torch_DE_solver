import torch
import os
import sys
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device

solver_device('gpu')

eps = 1.0
N = 10

C_ring = torch.tensor([0.123456, 0.654321, 0.345612, 0.216543, 0.561234, 0.432165], dtype=torch.float32)
factor_ring = 10 ** 4
U_SCALE = float(10 ** 4)


def enforce_hermitian(kx, ky, h):
    k_to_idx = {(int(kx[i].item()), int(ky[i].item())): i for i in range(len(kx))}

    visited = set()

    for i in range(len(kx)):
        k = (kx[i].item(), ky[i].item())
        if k in visited:
            continue

        k_neg = (-k[0], -k[1])
        j = k_to_idx.get(k_neg, None)

        if j is None:
            continue

        visited.add(k)
        visited.add(k_neg)

        if i == j:
            h[i] = torch.real(h[i].clone())
            continue

        avg = 0.5 * (h[i] + torch.conj(h[j]))
        h[i] = avg
        h[j] = torch.conj(avg)

    return h, k_to_idx


def make_gaussian_init(Kmax=10, device="cpu"):
    kx_range = torch.arange(-N // 2, N // 2, device=device, dtype=torch.float32)
    ky_range = torch.arange(-N // 2, N // 2, device=device, dtype=torch.float32)

    kx_grid, ky_grid = torch.meshgrid(kx_range, ky_range, indexing="ij")
    kx = kx_grid.flatten()
    ky = ky_grid.flatten()

    k_sq = kx ** 2 + ky ** 2
    abs_k = torch.sqrt(k_sq)

    # h(k) ~ complex Gaussian
    real = torch.randn(len(kx), device=device)
    imag = torch.randn(len(kx), device=device)
    h = torch.complex(real, imag)

    # enforce h(-k)=conj(h(k))
    h, k_to_idx = enforce_hermitian(kx, ky, h)

    g_hat = torch.zeros_like(h)
    eps_small = 1e-12

    # кольца n=1..6
    for n in range(1, Kmax):
        mask = (abs_k >= n - 0.5) & (abs_k < n + 0.5)
        if mask.sum() == 0:
            continue

        H_n = torch.sum(torch.abs(h[mask]) ** 2).clamp_min(eps_small)
        scale = factor_ring * torch.sqrt(C_ring[n - 1] / H_n)
        g_hat[mask] = scale * h[mask]

    # cutoff: |k| >= 13/2
    g_hat[abs_k >= 6.5] = 0.0

    g_hat = g_hat.to(dtype=torch.complex64, device=device)

    def init_func(grid):
        x = grid[:, 0:1]
        y = grid[:, 1:2]

        device_grid = grid.device
        kx_local = kx.to(device_grid).unsqueeze(0)
        ky_local = ky.to(device_grid).unsqueeze(0)
        g_local = g_hat.to(device_grid)

        phase = x @ kx_local + y @ ky_local

        g_r = torch.real(g_local).unsqueeze(1)  # (M,1)
        g_i = torch.imag(g_local).unsqueeze(1)  # (M,1)

        u = phase.cos().matmul(g_r) - phase.sin().matmul(g_i)     # (N,1)
        return u / U_SCALE

    def exact_func(grid):
        x = grid[:, 0:1]
        y = grid[:, 1:2]
        t = grid[:, 2:3]

        device_grid = grid.device
        kx_local = kx.to(device_grid).unsqueeze(0)
        ky_local = ky.to(device_grid).unsqueeze(0)
        k_sq_local = k_sq.to(device_grid).unsqueeze(0)
        g_local = g_hat.to(device_grid)

        phase = x @ kx_local + y @ ky_local
        decay = torch.exp(-eps * t @ k_sq_local)

        g_r = torch.real(g_local).unsqueeze(1)
        g_i = torch.imag(g_local).unsqueeze(1)

        cos_part = phase.cos() * decay
        sin_part = phase.sin() * decay

        u = cos_part.matmul(g_r) - sin_part.matmul(g_i)  # (N,1)
        return u / U_SCALE

    return init_func, exact_func


def heat_2d_gaussian_init_experiment(grid_res):
    exp_dict_list = []

    x_min, x_max = 0, 2 * torch.pi
    y_min, y_max = 0, 2 * torch.pi
    t_max = 0.01

    pde_dim_in = 3
    pde_dim_out = 1

    domain = Domain()
    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('y', [y_min, y_max], grid_res)
    domain.variable('t', [0, t_max], 6)

    boundaries = Conditions()

    init_func, exact_func = make_gaussian_init(Kmax=6, device='cuda')

    # Initial condition ################################################################################################

    # u(x, y, 0)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': [y_min, y_max], 't': 0},
                         value=lambda grid: init_func(grid).to(grid.device))

    # Boundary conditions ###################################################################################

    # u(0, y, t) = u(2*pi, y, t)
    boundaries.periodic([{'x': x_min, 'y': [y_min, y_max], 't': [0, t_max]},
                         {'x': x_max, 'y': [y_min, y_max], 't': [0, t_max]}])

    # u(x, 0, t) = u(x, 2*pi, t)
    boundaries.periodic([{'x': [x_min, x_max], 'y': y_min, 't': [0, t_max]},
                         {'x': [x_min, x_max], 'y': y_max, 't': [0, t_max]}])

    # Operator: du/dt -  epsilon * (u_xx + u_yy) = 0

    equation = Equation()

    heat_LT = {
        'du/dt**1': {
            'coeff': 1,
            'term': [2],
            'pow': 1,
            'var': 0
        },
        '-epsilon * d2u/dx2**1': {
            'coeff': -eps,
            'term': [0, 0],
            'pow': 1,
            'var': 0
        },
        '-epsilon * d2u/dy2**1': {
            'coeff': -eps,
            'term': [1, 1],
            'pow': 1,
            'var': 0
        }
    }
    equation.add(heat_LT)

    neurons = 100

    net = torch.nn.Sequential(
        torch.nn.Linear(pde_dim_in, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, pde_dim_out)
    )

    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    start = time.time()

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=1, lambda_bound=100)

    img_dir = os.path.join(os.path.dirname(__file__), 'heat_2d_gaussian_init_img')

    cb_cache = cache.Cache(cache_verbose=False, model_randomize_parameter=1e-6)

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=100,
                                         no_improvement_patience=1000,
                                         patience=10,
                                         randomize_parameter=1e-6,
                                         info_string_every=10)

    cb_plots = plot.Plots(save_every=500,
                          print_every=None,
                          img_dir=img_dir,
                          img_dim='2d',
                          scatter_flag=False,
                          plot_axes=[0, 1],
                          fixed_axes=[2],
                          n_samples=4,
                          img_rows=2,
                          img_cols=2)

    optimizer = Optimizer('Adam', {'lr': 5e-4})

    callbacks = [cb_cache, cb_es, cb_plots]

    model.train(optimizer, 1e5, save_model=False, callbacks=callbacks)

    end = time.time()

    grid = domain.build('NN').to('cuda')
    net = net.to('cuda')

    exact_on_grid = exact_func(grid)
    pred = net(grid)
    error_rmse = torch.sqrt(torch.mean((exact_on_grid - pred) ** 2))

    exp_dict_list.append({
        'grid_res': grid_res,
        'time': end - start,
        'RMSE': error_rmse.detach().cpu().numpy(),
        'type': 'heat_2d_gaussian_init',
        'cache': True
    })

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))

    return exp_dict_list


nruns = 1

exp_dict_list = []
for grid_res in range(100, 1001, 100):
    for r in range(nruns):
        exp_dict_list.append(heat_2d_gaussian_init_experiment(grid_res))

import pandas as pd

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)
df.to_csv('examples/benchmarking_data/heat_2d_gaussian_init_experiment.csv')
