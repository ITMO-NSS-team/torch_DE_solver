import torch
import os
import sys
import numpy as np
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import adaptive_lambda, cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.utils import exact_solution_data

solver_device('gpu')
datapath = "poisson_manyarea.dat"


def poisson_2d_many_subdomains_experiment(grid_res):
    exp_dict_list = []

    x_min, x_max = -10, 10
    y_min, y_max = -10, 10
    # grid_res = 50

    pde_dim_in = 2
    pde_dim_out = 1

    domain = Domain()
    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('y', [y_min, y_max], grid_res)

    split = (5, 5)
    freq = 2
    block_size = np.array([(x_max - x_min + 2e-5) / split[0], (y_max - y_min + 2e-5) / split[1]])

    a_cof = np.loadtxt("poisson_a_coef.dat")
    f_cof = np.loadtxt("poisson_f_coef.dat").reshape(split[0], split[1], freq, freq)

    boundaries = Conditions()

    # Operator: u + du/dn = 0

    def bop_generation(func_coeff, deriv_coeff, deriv_dim):
        bop = {
            'u':
                {
                    'coeff': func_coeff,
                    'term': [None],
                    'pow': 1
                },
            'du/dn':
                {
                    'coeff': deriv_coeff,
                    'term': [deriv_dim],
                    'pow': 1
                }
        }
        return bop

    bop_x_min = bop_generation(1, -1, 0)
    boundaries.robin({'x': x_min, 'y': [y_min, y_max]}, operator=bop_x_min, value=lambda grid: -grid[:, 1])

    bop_x_max = bop_generation(1, 1, 0)
    boundaries.robin({'x': x_max, 'y': [y_min, y_max]}, operator=bop_x_max, value=lambda grid: -grid[:, 1])

    bop_y_min = bop_generation(1, -1, 1)
    boundaries.robin({'x': [x_min, x_max], 'y': y_min}, operator=bop_y_min, value=lambda grid: -grid[:, 1])

    bop_y_max = bop_generation(1, 1, 1)
    boundaries.robin({'x': [x_min, x_max], 'y': y_max}, operator=bop_y_max, value=lambda grid: -grid[:, 1])

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
        device_origin = x.device
        x = x.detach().cpu()
        return torch.Tensor(a_coeff(x)).unsqueeze(dim=-1).to(device_origin)

    def get_forcing_term(x):
        device_origin = x.device
        x = x.detach().cpu()
        return torch.Tensor(forcing_term(x)).unsqueeze(dim=-1).to(device_origin)

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
                'pow': 0
            }
    }

    equation.add(poisson)

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

    img_dir = os.path.join(os.path.dirname(__file__), 'poisson_2d_many_subdomains_img')

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
                          img_dim='2d')  # 3 image dimension options: 3d, 2d, 2d_scatter

    optimizer = Optimizer('Adam', {'lr': 1e-3})

    model.train(optimizer, 5e5, save_model=True, callbacks=[cb_cache, cb_es, cb_plots])

    end = time.time()

    grid = domain.build('NN').to('cuda')
    net = net.to('cuda')

    exact = exact_solution_data(grid, datapath, pde_dim_in, pde_dim_out).reshape(-1, 1)
    net_predicted = net(grid)

    error_rmse = torch.sqrt(torch.mean((exact - net_predicted) ** 2))

    exp_dict_list.append({
        'grid_res': grid_res,
        'time': end - start,
        'RMSE': error_rmse.detach().cpu().numpy(),
        'type': 'poisson_2d_many_subdomains',
        'cache': True
    })

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))

    return exp_dict_list


nruns = 10

exp_dict_list = []

for grid_res in range(50, 501, 50):
    for _ in range(nruns):
        exp_dict_list.append(poisson_2d_many_subdomains_experiment(grid_res))

import pandas as pd

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)
df.to_csv('examples/benchmarking_data/poisson_2d_many_subdomains_experiment_physical_50_500_cache={}.csv'
          .format(str(True)))
