import numpy as np
from scipy.io import loadmat

import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy
import time
import pandas as pd
from scipy.integrate import quad
import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append('../')
sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.input_preprocessing import Equation
from tedeous.solver import Solver
from tedeous.solution import Solution
from tedeous.device import solver_device, device_type


def sln_ac(grid):
    data = scipy.io.loadmat(os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'wolfram_sln/AC.mat')))
    usol = data['uu'].reshape(-1)
    t_star = data['tt'][0]
    x_star = data['x'][0]
    grid_data = torch.cartesian_prod(torch.from_numpy(x_star),
                                     torch.from_numpy(t_star)).float()
    u = scipy.interpolate.griddata(grid_data, usol, grid.to('cpu'), method='nearest')

    return torch.from_numpy(u.reshape(-1))

solver_device('cuda')

device = device_type()

grid_res = 20

data_res = 20

x = torch.from_numpy(np.linspace(-1, 1, grid_res + 1))
t = torch.from_numpy(np.linspace(0, 1, grid_res + 1))

grid = torch.cartesian_prod(x, t).float()

bnd1 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()

bndval1 = bnd1[:,0:1]**2*torch.cos(np.pi*bnd1[:,0:1])
    
bnd2_l = torch.cartesian_prod(torch.from_numpy(np.array([-1], dtype=np.float64)), t).float()

bnd2_r = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)), t).float()

bnd2 = [bnd2_l, bnd2_r]

bnd3_l = torch.cartesian_prod(torch.from_numpy(np.array([-1], dtype=np.float64)), t).float()

bnd3_r = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)), t).float()

bop3= {
        'du/dx':
            {
                'coeff': 1,
                'du/dx': [0],
                'pow': 1,
                'var': 0
            }
}

bnd3 = [bnd3_l, bnd3_r]

if data_res != 0:
    x_data = torch.FloatTensor(data_res).uniform_(-1, 1)
    t_data = torch.FloatTensor(data_res).uniform_(0, 1)

    bnd4 = torch.cartesian_prod(x_data, t_data).float()

    bndval4 = sln_ac(bnd4)
    
    bconds = [[bnd1, bndval1, 'dirichlet'],
                [bnd2, 'periodic'],
                [bnd3, bop3, 'periodic'],
                [bnd4, bndval4, 'data']]

else:
    bconds = [[bnd1, bndval1, 'dirichlet'],
                [bnd2, 'periodic'],
                [bnd3, bop3, 'periodic']]


AC = {
    '1*du/dt**1':
        {
            'coeff': 1,
            'du/dt': [1],
            'pow': 1,
            'var': 0
        },
    '-0.0001*d2u/dx2**1':
        {
            'coeff': -0.0001,
            'd2u/dx2': [0,0],
            'pow': 1,
            'var': 0
        },
    '+5u**3':
        {
            'coeff': 5,
            'u': [None],
            'pow': 3,
            'var': 0
        },
    '-5u**1':
        {
            'coeff': -5,
            'u': [None],
            'pow': 1,
            'var': 0
        }
}
    

model = torch.nn.Sequential(
        torch.nn.Linear(2, 128),
        torch.nn.Tanh(),
        torch.nn.Linear(128, 128),
        torch.nn.Tanh(),
        torch.nn.Linear(128, 128),
        torch.nn.Tanh(),
        torch.nn.Linear(128, 128),
        torch.nn.Tanh(),
        torch.nn.Linear(128, 1)
    )

img_dir=os.path.join(os.path.dirname( __file__ ), 'AC_img_data')

if not(os.path.isdir(img_dir)):
    os.mkdir(img_dir)

start = time.time()

equation = Equation(grid, AC, bconds).set_strategy('autograd')

model = Solver(grid, equation, model, 'autograd').solve(lambda_bound=100, verbose=1, learning_rate=1e-3, lambda_update=True,
                                                                eps=1e-6, tmin=10, tmax=5e6, use_cache=False, print_every=100,
                                                                cache_dir='../cache/', patience=5, step_plot_save=True,
                                                                optimizer_mode='Adam', image_save_dir=img_dir)

end = time.time()
time_part = end - start

x1 = torch.from_numpy(np.linspace(-1, 1, grid_res + 1))
t1 = torch.from_numpy(np.linspace(0, 1, grid_res + 1))
grid1 = torch.cartesian_prod(x1, t1).float().to(device)

u_exact = sln_ac(grid1).to(device)

predict = model(grid1).reshape(-1)

error_rmse = torch.sqrt(torch.mean((u_exact - predict) ** 2))


print('Time taken {}= {}'.format(grid_res, end - start))
print('RMSE {}= {}'.format(data_res, error_rmse))



