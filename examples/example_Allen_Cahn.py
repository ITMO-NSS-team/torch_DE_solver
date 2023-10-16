import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
import pandas as pd
import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append('../')
sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.input_preprocessing import Equation
from tedeous.solver import Solver
from tedeous.solution import Solution
from tedeous.device import solver_device
from tedeous.models import FourierNN
from tedeous.models import Fourier_embedding

solver_device('gpu')

x = torch.from_numpy(np.linspace(-1, 1, 51))
t = torch.from_numpy(np.linspace(0, 1, 51))

# if the casual_loss is used the time parameter must be
# at the first place in the grid

grid = torch.cartesian_prod(t, x).float()

# Initial conditions at t=0
bnd1 = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), x).float()
    
# u(0,x)=sin(pi*x)
bndval1 = bnd1[:,1:]**2*torch.cos(np.pi*bnd1[:,1:])
    
# Initial conditions at t=1
bnd2_l = torch.cartesian_prod(t, torch.from_numpy(np.array([-1], dtype=np.float64))).float()

bnd2_r = torch.cartesian_prod(t, torch.from_numpy(np.array([1], dtype=np.float64))).float()

bnd2 = [bnd2_l, bnd2_r]

bnd3_l = torch.cartesian_prod(t, torch.from_numpy(np.array([-1], dtype=np.float64))).float()

bnd3_r = torch.cartesian_prod(t, torch.from_numpy(np.array([1], dtype=np.float64))).float()

bop3= {
        'du/dx':
            {
                'coeff': 1,
                'du/dx': [1],
                'pow': 1,
                'var': 0
            }
}

bnd3 = [bnd2_l, bnd2_r]
    
# Putting all bconds together
bconds = [[bnd1, bndval1, 'dirichlet'],
        [bnd2, 'periodic'],
        [bnd3, bop3, 'periodic']]


   
AC = {
    '1*du/dt**1':
        {
            'coeff': 1,
            'du/dt': [0],
            'pow': 1,
            'var': 0
        },
    '-0.0001*d2u/dx2**1':
        {
            'coeff': -0.0001,
            'd2u/dx2': [1,1],
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

FFL = Fourier_embedding(L=[None, 2], M=[None, 10])

out = FFL.out_features

model = torch.nn.Sequential(
    FFL,
    torch.nn.Linear(out, 128),
    torch.nn.Tanh(),
    torch.nn.Linear(128,128),
    torch.nn.Tanh(),
    torch.nn.Linear(128,128),
    torch.nn.Tanh(),
    torch.nn.Linear(128,1)
)
    
#model = Modified_MLP([128, 128, 128, 1], L=[None, 2], M=[None, 10]) # NN structure for more accurate solution

img_dir=os.path.join(os.path.dirname( __file__ ), 'AC_eq_img')

if not(os.path.isdir(img_dir)):
    os.mkdir(img_dir)

equation = Equation(grid, AC, bconds).set_strategy('autograd')

model = Solver(grid, equation, model, 'autograd').solve(lambda_bound=100,
                                         verbose=True, learning_rate=1e-3, eps=1e-7, tmin=1000, tmax=1e5,
                                         use_cache=True,cache_dir='../cache/',cache_verbose=False,
                                         save_always=False,print_every=1000,
                                         patience=6,loss_oscillation_window=100,no_improvement_patience=1000,
                                         model_randomize_parameter=1e-5,optimizer_mode='Adam',cache_model=None,
                                         step_plot_print=False, step_plot_save=True, tol=50, image_save_dir=img_dir)
