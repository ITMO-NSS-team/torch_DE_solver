import torch
import numpy as np
import os
import sys
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append('../')
sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.input_preprocessing import Equation
from tedeous.solver import Solver, grid_format_prepare
from tedeous.device import solver_device, check_device
from tedeous.optimizers import PSO


def exact_solution(grid):
    grid = grid.to('cpu').detach()
    test_data = scipy.io.loadmat(os.path.abspath(
        os.path.join(os.path.dirname( __file__ ), 'wolfram_sln/buckley_exact.mat')))
    u = torch.from_numpy(test_data['u']).reshape(-1, 1)

    # grid_test
    x = torch.from_numpy(test_data['x']).reshape(-1, 1)
    t = torch.from_numpy(test_data['t']).reshape(-1, 1)

    grid_data = torch.cat((x, t), dim=1)

    exact = scipy.interpolate.griddata(grid_data, u, grid, method='nearest').reshape(-1)

    return torch.from_numpy(exact)
    

solver_device('cuda')

m = 0.2
L = 1
Q = -0.1
Sq = 1
mu_w = 0.89e-3
mu_o = 4.62e-3
Swi0 = 0.
Sk = 1.
t_end = 1.


def experiment(grid_res, mode):
    
    x = torch.linspace(0, 1, grid_res+1)
    t = torch.linspace(0, t_end, grid_res+1)

    grid = grid_format_prepare([x,t], mode=mode).float()

    ##initial cond
    bnd1 = torch.cartesian_prod(x, torch.tensor([0.])).float()
    bndval1 = torch.zeros_like(x) + Swi0

    ##boundary cond
    bnd2 = torch.cartesian_prod(torch.tensor([0.]), t).float()
    bndval2 = torch.zeros_like(t) + Sk

    bconds = [[bnd1, bndval1, 'dirichlet'],
            [bnd2, bndval2, 'dirichlet']]

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 20),
        torch.nn.Tanh(),
        torch.nn.Linear(20, 20),
        torch.nn.Tanh(),
        torch.nn.Linear(20, 20),
        torch.nn.Tanh(),
        torch.nn.Linear(20, 1)
    )

    def k_o(x):
        return (1-model(x))**2

    def k_w(x):
        return (model(x))**2

    def dk_w(x):
        return 2*model(x)

    def dk_o(x):
        return -2*(1-model(x))

    def df(x):
        return (dk_w(x)*(k_w(x)+mu_w/mu_o*k_o(x))-
                k_w(x)*(dk_w(x)+mu_w/mu_o*dk_o(x)))/(k_w(x)+mu_w/mu_o*k_o(x))**2

    def coef_model(x):
        return -Q/Sq*df(x)

    buckley_eq = {
        'm*ds/dt**1':
            {
                'coeff': m,
                'ds/dt': [1],
                'pow': 1
            },
        '-Q/Sq*df*ds/dx**1':
            {
                'coeff': coef_model,
                'ds/dx': [0],
                'pow':1
            }
    }

    equation = Equation(grid, buckley_eq, bconds).set_strategy(mode)

    img_dir=os.path.join(os.path.dirname( __file__ ), 'Buckley_img')

    model = Solver(grid, equation, model, mode).solve(lambda_bound=10,
                                                            verbose=1,
                                                            learning_rate=1e-3,
                                                            eps=1e-6,
                                                            optimizer_mode='Adam',
                                                            tmax=10000,
                                                            print_every=1000,
                                                            step_plot_save=True,
                                                            image_save_dir=img_dir)

    u_exact = exact_solution(grid).to('cuda')

    u_exact = check_device(u_exact).reshape(-1)

    u_pred = check_device(model(grid)).reshape(-1)

    error_rmse = torch.sqrt(torch.sum((u_exact - u_pred)**2)) / torch.sqrt(torch.sum(u_exact**2))

    print('RMSE_adam= ', error_rmse.item())

    pso = PSO(
        pop_size=100,
        b=0.5,
        c2=0.05,
        variance=5e-2,
        c_decrease=True,
        lr=5e-3
    )

    model = Solver(grid, equation, model, mode).solve(lambda_bound=10,
                                                            verbose=1,
                                                            use_cache=False,
                                                            optimizer_mode=pso,
                                                            tmin=3000,
                                                            print_every=100,
                                                            step_plot_save=True,
                                                            image_save_dir=img_dir)

    u_pred = check_device(model(grid)).reshape(-1)

    error_rmse = torch.sqrt(torch.sum((u_exact - u_pred)**2)) / torch.sqrt(torch.sum(u_exact**2))

    print('RMSE_pso= ', error_rmse.item())

    return model

for i in range(2):
    model = experiment(20, 'autograd')

## After experiment, RMSE_adam ~ 0.23, RMSE_pso ~ 0.19 or less.