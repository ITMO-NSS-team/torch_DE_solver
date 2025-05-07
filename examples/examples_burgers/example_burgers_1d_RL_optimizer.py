# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:33:44 2021

@author: user
"""
import torch
import os
import sys
import time
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../examples_PINNacle')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model

from tedeous.callbacks import early_stopping, plot, cache
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.utils import exact_solution_data

solver_device('gpu')

datapath = "../../PINNacle_data/burgers1d.npy"

mu = 0.01 / np.pi


def burgers_1d_experiment(grid_res):
    exp_dict_list = []

    x_min, x_max = -1, 1
    t_max = 1

    pde_dim_in = 2
    pde_dim_out = 1

    domain = Domain()
    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('t', [0, t_max], grid_res)

    boundaries = Conditions()

    # Initial conditions ###############################################################################################

    # u(x, 0) = -sin(pi * x)
    boundaries.dirichlet({'x': [x_min, x_max], 't': 0}, value=lambda grid: -torch.sin(np.pi * grid[:, 0]))

    # Boundary conditions ##############################################################################################

    # u(x_min, t) = 0
    boundaries.dirichlet({'x': x_min, 't': [0, t_max]}, value=0)

    # u(x_max, t) = 0
    boundaries.dirichlet({'x': x_max, 't': [0, t_max]}, value=0)

    equation = Equation()

    # Operator: u_t + u * u_x - mu * u_xx = 0

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
                'coeff': 1,
                'u*du/dx': [[None], [0]],
                'pow': [1, 1],
                'var': [0, 0]
            },
        '-mu*d2u/dx2':
            {
                'coeff': -mu,
                'd2u/dx2': [0, 0],
                'pow': 1,
                'var': 0
            }
    }

    equation.add(burgers_eq)

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

    model.compile('autograd', lambda_operator=1, lambda_bound=10)

    cb_cache = cache.Cache(cache_verbose=False, model_randomize_parameter=1e-5)

    img_dir = os.path.join(os.path.dirname(__file__), 'burgers_1d_img')

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=100,
                                         no_improvement_patience=1000,
                                         patience=100,
                                         randomize_parameter=1e-4,
                                         info_string_every=10)

    cb_plots = plot.Plots(save_every=None,
                          print_every=None,
                          img_dir=img_dir,
                          scatter_flag=False)

    # RL integration ###########################################################################################
    # The wrapper should be inside the model.train method
    # We need to change the optimizers inside the RL learning loop and substitute them into the model learning loop
    # We can change optimizers only inside model.train and inside the RL loop
    # Other optimizer replacement configurations (e.g., within the example equation) will cause the training loop to not
    # work correctly due to the update of the model.train method

    # Following parameters must be entered into the model
    # these parameters took from /landscape_visualization_origin/_aux/plot_loss_surface.py file:

    # optimizer = {
    #     'CSO': ({"lr": 1e-3}, 30),
    #     'Adam': ({"lr": 1e-4}, 70),
    #     'LBFGS': ({"lr": 1,
    #                "max_iter": 20,
    #                "max_eval": None,
    #                "tolerance_grad": 1e-05,
    #                "tolerance_change": 1e-07,
    #                "history_size": 50,
    #                "line_search_fn": "strong_wolfe"}, 50),
    #     # 'NGD': ({'grid_steps_number': 20}, 30),
    #     'NNCG': ({"mu": 1e-1,
    #               "lr": 1,
    #               "rank": 10,
    #               "line_search_fn": "armijo",
    #               "precond_update_frequency": 20,
    #               "eigencdecomp_shift_attepmt_count": 10,
    #               # 'cg_max_iters': 1000,
    #               "verbose": False}, 50)
    # }

    # optimizer = [
    #     {
    #         "name": "CSO",
    #         "params": {"lr": 5e-4},
    #         "epochs": 100
    #     },
    #     {
    #         "name": "Adam",
    #         "params": {"lr": 1e-4},
    #         "epochs": 1000
    #     },
    #     {
    #         "name": "LBFGS",
    #         "params": {
    #             "lr": 1,
    #             "max_iter": 20,
    #             "max_eval": None,
    #             "tolerance_grad": 1e-05,
    #             "tolerance_change": 1e-07,
    #             "history_size": 50,
    #             "line_search_fn": "strong_wolfe"
    #         }, "epochs": 100
    #     },
    #     {
    #         "name": "NNCG",
    #         "params": {
    #             "mu": 1e-1,
    #             "lr": 1,
    #             "rank": 10,
    #             "line_search_fn": "armijo",
    #             "precond_update_frequency": 20,
    #             "eigencdecomp_shift_attepmt_count": 10,
    #             # 'cg_max_iters': 1000,
    #             "verbose": False},
    #         "epochs": 50
    #     }
    # ]

    optimizer = {
        "type": ['Adam', 'RAdam', 'Adam', 'LBFGS', 'PSO', 'CSO', 'RMSprop'],
        "params": [0.1, 0.01, 0.001, 0.0001],
        "epochs": [500, 1000, 2000]
    }

    # optimizer = Optimizer('Adam', {'lr': 1e-4})

    AE_model_params = {
        "mode": "NN",
        "num_of_layers": 3,
        "layers_AE": [
            991,
            125,
            15
        ],
        "num_models": None,
        "from_last": False,
        "prefix": "model-",
        "every_nth": 1,
        "grid_step": 0.1,
        "d_max_latent": 2,
        "anchor_mode": "circle",
        "rec_weight": 10000.0,
        "anchor_weight": 0.0,
        "lastzero_weight": 0.0,
        "polars_weight": 0.0,
        "wellspacedtrajectory_weight": 0.0,
        "gridscaling_weight": 0.0,
        "device": "cuda"
    }

    AE_train_params = {
        "first_RL_epoch_AE_params": {
            "epochs": 10000,
            "patience_scheduler": 5000,
            "cosine_scheduler_patience": 5000,
        },
        "other_RL_epoch_AE_params": {
            "epochs": 20000,
            "patience_scheduler": 4000,
            "cosine_scheduler_patience": 1200,
        },
        "batch_size": 32,
        "every_epoch": 100,
        "learning_rate": 5e-4,
        "resume": True,
        "finetune_AE_model": False
    }

    loss_surface_params = {
        "loss_type": "loss_total",
        "every_nth": 1,
        "num_of_layers": 3,
        "layers_AE": [
            991,
            125,
            15
        ],
        "batch_size": 32,
        "num_models": None,
        "from_last": False,
        "prefix": "model-",
        "loss_name": "loss_total",
        "x_range": [-1.25, 1.25, 25],
        "vmax": -1.0,
        "vmin": -1.0,
        "vlevel": 30.0,
        "key_models": None,
        "key_modelnames": None,
        "density_type": "CKA",
        "density_p": 2,
        "density_vmax": -1,
        "density_vmin": -1,
        "colorFromGridOnly": True,
        "img_dir": img_dir
    }

    n_save_models = 10
    n_trajectories = 5

    model.train(optimizer,
                5e5,
                save_model=True,
                callbacks=[cb_es, cb_plots, cb_cache],
                rl_opt_flag=True,
                models_concat_flag=False,
                equation_params=equation_params,
                AE_model_params=AE_model_params,
                AE_train_params=AE_train_params,
                loss_surface_params=loss_surface_params,
                n_save_models=n_save_models,
                n_trajectories=n_trajectories)

    end = time.time()

    grid = domain.build('NN').to('cuda')
    net = net.to('cuda')

    exact = exact_solution_data(grid, datapath, pde_dim_in, pde_dim_out, t_dim_flag=True).reshape(-1, 1)
    net_predicted = net(grid)

    error_rmse = torch.sqrt(torch.mean((exact - net_predicted) ** 2))

    exp_dict_list.append({
        'grid_res': grid_res,
        'time': end - start,
        'RMSE': error_rmse.detach().cpu().numpy(),
        'type': 'burgers_1d',
        'cache': True
    })

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))

    return exp_dict_list


nruns = 10

exp_dict_list = []

for grid_res in range(50, 501, 50):
    for _ in range(nruns):
        exp_dict_list.append(burgers_1d_experiment(grid_res))

import pandas as pd

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)
df.to_csv('examples/benchmarking_data/burgers_1d_experiment_50_500_cache={}.csv'.format(str(True)))
