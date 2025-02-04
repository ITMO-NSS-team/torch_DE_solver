
import torch
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from scipy.integrate import quad

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname('AAAI_expetiments'))))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import plot, early_stopping, adaptive_lambda, save_model
from tedeous.optimizers.optimizer import Optimizer

from tedeous.optimizers.optimizer import Optimizer
from landscape_visualization._aux.visualization_model import VisualizationModel
from landscape_visualization._aux.early_stopping_plot import EarlyStopping as plot_early_stopping
from landscape_visualization._aux.plot_loss_surface import PlotLossSurface

import pandas as pd


mu = 0.01 / np.pi



def u(grid):
    
    def f(y):
        return np.exp(-np.cos(np.pi * y) / (2 * np.pi * mu))

    def integrand1(m, x, t):
        return np.sin(np.pi * (x - m)) * f(x - m) * np.exp(-m ** 2 / (4 * mu * t))

    def integrand2(m, x, t):
        return f(x - m) * np.exp(-m ** 2 / (4 * mu * t))

    def u(x, t):
        if t == 0:
            return -np.sin(np.pi * x)
        else:
            return -quad(integrand1, -np.inf, np.inf, args=(x, t))[0] / quad(integrand2, -np.inf, np.inf, args=(x, t))[
                0]

    solution = []
    for point in grid:
        solution.append(u(point[0].item(), point[1].item()))

    return torch.tensor(solution)


def u_net(net, x):
    net = net.to('cpu')
    x = x.to('cpu')
    return net(x).detach()


def l2_norm(net, x):
    x = x.to('cpu')
    net = net.to('cpu')
    predict = net(x).detach().cpu().reshape(-1)
    exact = u(x).detach().cpu().reshape(-1)
    l2_norm = torch.sqrt(sum((predict-exact)**2))
    return l2_norm.detach().cpu().numpy()

def l2_norm_mat(net, x):
    x = x.to('cpu')
    net = net.to('cpu')
    predict = net.detach().cpu().reshape(-1)
    exact = u(x).detach().cpu().reshape(-1)
    l2_norm = torch.sqrt(sum((predict-exact)**2))
    return l2_norm.detach().cpu().numpy()

def l2_norm_fourier(net, x):
    x = x.to(torch.device('cuda:0'))
    predict = net(x).detach().cpu().reshape(-1)
    exact = u(x).detach().cpu().reshape(-1)
    l2_norm = torch.sqrt(sum((predict-exact)**2))
    return l2_norm.detach().cpu().numpy()


def burgers1d_problem_formulation(grid_res):
    
    domain = Domain()
    domain.variable('x', [-1, 1], grid_res)
    domain.variable('t', [0, 1], grid_res)

    boundaries = Conditions()
    x = domain.variable_dict['x']
    boundaries.dirichlet({'x': [-1, 1], 't': 0}, value=-torch.sin(np.pi * x))

    boundaries.dirichlet({'x': -1, 't': [0, 1]}, value=0)

    boundaries.dirichlet({'x': 1, 't': [0, 1]}, value=0)

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

    grid = domain.build('autograd')

    return grid,domain,equation,boundaries



def experiment_state_burgers_1d_adam_lbfgs(grid_res, exp_name='burgers1d_adam_lbfgs_nncg', save_plot=True):
    # solver_device('cuda')
    exp_dict_list = []

    grid,domain,equation,boundaries = burgers1d_problem_formulation(grid_res)

    net = torch.nn.Sequential(
        torch.nn.Linear(2, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 1)
    )
    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=1/2, lambda_bound=1/2)

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                        loss_window=100,
                                        no_improvement_patience=100,
                                        patience=2,
                                        randomize_parameter=1e-5,
                                        verbose=False)

    optim = Optimizer('Adam', {'lr':  1e-3})
    
    start=time.time()
    path_to_parent_folder = r"C:\Users\Рустам\Documents\GitHub\torch_DE_solver_local\landscape_visualization\trajectories\burgers\adam_state_test"
    path_to_folder = path_to_parent_folder + r"\optimizer_Adam"
    os.makedirs(path_to_folder, exist_ok=True)
    
    cb_sm = save_model.SaveModel(path_to_folder, every_step = 50)
    model.train(optim, 2000, save_model=False,  callbacks=[cb_es, cb_sm], info_string_every=500 )
    end = time.time()

    time_adam = end - start


    grid_test = torch.cartesian_prod(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100))

    u_exact_train = u(grid).reshape(-1)

    u_exact_test = u(grid_test).reshape(-1)

    # error_adam_train = torch.sqrt(torch.mean((u_exact_train - net(grid).reshape(-1)) ** 2))

    error_adam_test = torch.sqrt(torch.mean((u_exact_test - net(grid_test).reshape(-1)) ** 2))

    # loss_adam = model.solution_cls.evaluate()[0].detach().cpu().numpy()

    # lu_f = model.solution_cls.operator.operator_compute()

    # lu_f, gr = integration(lu_f, grid)

    # lu_f_adam, _ = integration(lu_f, gr)

    print('Time taken {}= {}'.format(grid_res, time_adam))
    print('RMSE_adam {}= {}'.format(grid_res, error_adam_test))

    path_to_plot_model = r"test\landscape_visualization\saved_models\PINN_burgers_adam_state_test\model.py"
    path_to_trajectories =  path_to_parent_folder
    raw_states = make_loss_surface(path_to_plot_model, path_to_trajectories, grid_res)
    print(raw_states)

    ########
    # Choosing some action based on raw_states. For example, training with LBFGS
    ########

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                        loss_window=100,
                                        no_improvement_patience=200,
                                        patience=2,
                                        randomize_parameter=1e-5,
                                        verbose=False)

    optim = Optimizer('LBFGS', {'history_size': 100,
                                "line_search_fn": 'strong_wolfe'})

    start = time.time()
    path_to_folder = path_to_parent_folder + r"\optimizer_LBFGS"
    os.makedirs(path_to_folder, exist_ok=True)
    cb_sm = save_model.SaveModel(path_to_folder, every_step = 50)
    model.train(optim, 3000, save_model=False, callbacks=[cb_es, cb_sm], info_string_every=100)
    end = time.time()
    time_LBFGS = end - start

    # error_LBFGS_train = torch.sqrt(torch.mean((u_exact_train - net(grid).reshape(-1)) ** 2))

    error_LBFGS_test = torch.sqrt(torch.mean((u_exact_test - net(grid_test).reshape(-1)) ** 2))

    # loss_LBFGS = model.solution_cls.evaluate()[0].detach().cpu().numpy()

    # lu_f = model.solution_cls.operator.operator_compute()

    # grid = domain.build('autograd')

    # lu_f, gr = integration(lu_f, grid)

    # lu_f_LBFGS, _ = integration(lu_f, gr)

    print('Time taken {}= {}'.format(grid_res, time_LBFGS))
    print('RMSE_LBFGS{}= {}'.format(grid_res, error_LBFGS_test))

    path_to_plot_model = r"test\landscape_visualization\saved_models\PINN_burgers_adam_state_test\model.py"
    path_to_trajectories =  path_to_parent_folder
    raw_states = make_loss_surface(path_to_plot_model, path_to_trajectories, grid_res, resume=True)
    print(raw_states)



def make_loss_surface(path_to_plot_model, path_to_trajectories, grid_res=100, resume=False):

    model_args = {
    "mode": "NN",
    "num_of_layers": 3,
    "layers_AE": [
        991,
        125,
        15
    ],
    "path_to_plot_model": path_to_plot_model,
    "num_models": None,
    "from_last": False,
    "prefix": "model-",
    "path_to_trajectories": path_to_trajectories,
    "every_nth": 1,
    "grid_step": 0.1,
    "d_max_latent": 2,
    "anchor_mode": "circle",
    "rec_weight": 1.0,
    "anchor_weight": 0.0,
    "lastzero_weight": 0.0,
    "polars_weight": 0.0,
    "wellspacedtrajectory_weight": 0.0,
    "gridscaling_weight": 0.0,}   
    
    batch_size = 16
    epochs = 20000
    patience_scheduler = 400000
    every_epoch = 100
    cosine_scheduler_patience = 2000
    learning_rate = 0.0005
    resume = resume


    model = VisualizationModel(**model_args)
    optim = Optimizer('RMSprop', {'lr': learning_rate,}, cosine_scheduler_patience=cosine_scheduler_patience)
    cb_es = plot_early_stopping(patience=patience_scheduler)
    model.train(optim, epochs, every_epoch, batch_size, resume, callbacks=[cb_es])


    nth = 1
    # key_models, key_modelnames = generate_key_lists(path_to_trajectories, nth)
    # print(key_models, key_modelnames)

    model_layers = [2, 32, 32, 1]  # PINN layers
    grid_test = torch.cartesian_prod(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100))
    u_exact_test = u(grid_test).reshape(-1)
    loss_type = "loss_total" #loss_total, loss_oper, loss_bnd, u
    raw_states = {}
    for loss_type in ["loss_total", "loss_oper", "loss_bnd"]:
        plot_args = {
                "loss_type":loss_type,
                "every_nth": nth,
                "num_of_layers": 3,
                "layers_AE": [
                    991,
                    125,
                    15
                ],
                "batch_size": 32,
                "path_to_plot_model": path_to_plot_model,
                "num_models": None,
                "from_last": False,
                "prefix": "model-",
                "path_to_trajectories": path_to_trajectories,
                "loss_name": loss_type,
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
                "colorFromGridOnly": True
                }
        
        
        plotter = PlotLossSurface(**plot_args)
        grid, domain, equation, boundaries = burgers1d_problem_formulation(grid_res)
        raw_state = plotter.save_equation_loss_surface(u_exact_test, grid_test,  grid, domain, equation, boundaries, model_layers)
        raw_states[loss_type] = raw_state
    return raw_states


if __name__ == '__main__':


    grid_res = 100

    experiment_state_burgers_1d_adam_lbfgs(grid_res)

    