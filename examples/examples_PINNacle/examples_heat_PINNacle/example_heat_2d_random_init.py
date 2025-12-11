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

C_ring = torch.tensor([0.123456, 0.654321, 0.345612, 0.216543, 0.561234, 0.432165], dtype=torch.float32)
factor_ring = 10 ** 4


def make_gaussian_init(Kmax=6, seed=None, device="cpu"):
    if seed is not None:
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed(seed)

    kx_range = torch.arange(-Kmax, Kmax + 1, device=device)
    ky_range = torch.arange(-Kmax, Kmax + 1, device=device)

    kx_grid, ky_grid = torch.meshgrid(kx_range, ky_range, indexing='ij')

    kx = kx_grid.flatten()
    ky = ky_grid.flatten()

    k_sq = kx ** 2 + ky ** 2
    abs_k = torch.sqrt(k_sq)

    real = torch.randn(len(kx), device=device)
    imag = torch.randn(len(kx), device=device)
    h = torch.complex(real, imag)

    k_to_idx = {(int(kx[idx]), int(ky[idx])): idx for idx in range(len(kx))}

    for idx in range(len(kx)):
        kx_val = int(kx[idx].item())
        ky_val = int(ky[idx].item())
        neg_idx = k_to_idx.get((-kx_val, -ky_val), None)
        if neg_idx is None:
            continue
        if neg_idx == idx:
            h[idx] = torch.complex(torch.real(h[idx]), torch.tensor(0.0, device=h.device))
            continue
        if idx < neg_idx:
            avg = 0.5 * (h[idx] + torch.conj(h[neg_idx]))
            h[idx] = avg
            h[neg_idx] = torch.conj(avg)

    g_hat = torch.zeros_like(h)

    eps_small = 1e-12

    for n in range(1, 7):
        mask = (abs_k >= n - 0.5) & (abs_k < n + 0.5)
        if mask.sum() == 0:
            continue

        if mask.sum() > 0:
            H_n = torch.sum(torch.abs(h[mask]) ** 2)

            if H_n < eps_small:
                H_n = eps_small

            scale = factor_ring * torch.sqrt(C_ring[n - 1] / H_n)
            g_hat[mask] = scale * h[mask]

    g_hat[abs_k >= 6.5] = torch.complex(torch.tensor(0.0, device=device),
                                        torch.tensor(0.0, device=device))

    E_spec = torch.sum(torch.abs(g_hat) ** 2)
    E0 = 1.0
    if E_spec > 0:
        g_hat = g_hat * torch.sqrt(E0 / E_spec)

    def value_fn(grid):
        x = grid[:, 0]
        y = grid[:, 1]

        device_grid = grid.device

        kx_local = kx.to(device_grid)
        ky_local = ky.to(device_grid)
        g_hat_local = g_hat.to(device_grid)

        phase = (x.unsqueeze(1) * kx_local.unsqueeze(0) +
                 y.unsqueeze(1) * ky_local.unsqueeze(0))

        u_complex = torch.matmul(torch.exp(1j * phase), g_hat_local)
        return torch.real(u_complex)

    def exact_fn(grid):
        x = grid[:, 0]
        y = grid[:, 1]
        t = grid[:, 2]

        device_grid = grid.device

        kx_local = kx.to(device_grid)
        ky_local = ky.to(device_grid)
        k_sq_local = k_sq.to(device_grid)
        g_hat_local = g_hat.to(device_grid)

        phase = (x.unsqueeze(1) * kx_local.unsqueeze(0) +
                 y.unsqueeze(1) * ky_local.unsqueeze(0))

        decay = torch.exp(-eps * t.unsqueeze(1) * k_sq_local.unsqueeze(0))
        u_complex = torch.matmul(torch.exp(1j * phase) * decay, g_hat_local)
        return torch.real(u_complex)

    return value_fn, exact_fn


def heat_2d_gaussian_init_experiment(grid_res, seed=None):
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

    value_fn, exact_fn = make_gaussian_init(Kmax=6, seed=seed, device='cuda')

    # Initial condition ################################################################################################

    # u(x, y, 0)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': [y_min, y_max], 't': 0},
                         value=lambda grid: value_fn(grid).to(grid.device))

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

    # exact_on_grid = exact_fn(grid.cpu()).to('cuda')
    exact_on_grid = exact_fn(grid)
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
for grid_res in range(10, 101, 10):
    for r in range(nruns):
        exp_dict_list.append(heat_2d_gaussian_init_experiment(grid_res, seed=r))

import pandas as pd

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)
df.to_csv('examples/benchmarking_data/heat_2d_gaussian_init_experiment.csv')
