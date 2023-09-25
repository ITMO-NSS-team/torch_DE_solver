import torch
import torchtext
import SALib
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.input_preprocessing import Equation
from tedeous.solver import Solver
from tedeous.solution import Solution
from tedeous.device import solver_device


solver_device("cuda")

x_grid = np.linspace(-5,5,41)
t_grid = np.linspace(0,np.pi/2,41)

x = torch.from_numpy(x_grid)
t = torch.from_numpy(t_grid)

grid = torch.cartesian_prod(x, t).float()


"""
To solve schrodinger equation we have to solve system because of its complexity. 
Both for the operator and for boundary and initial conditions.
The system of boundary and initial conditions is written as follows:
bnd1 = torch.stack((bnd1_real, bnd1_imag),dim = 1)
etc...
For periodic bconds you need to set parameter bnd_type = 'periodic'.
For 'periodic' you don't need to set bnd_val.
In this case condition will be written as follows:
bnd1_left = ...
bnd1_right = ...
bnd1 = [bnd1_left, bnd1_right]
Each term of bconds support up to 4 parameters, such as: bnd, bnd_val, bnd_op, bnd_type.
bnd_type is not necessary, default = 'boundary values'.
bnd, bnd_val are essentials for setting parameters bconds.
Eventually, whole list of bconds will be written:
bconds = [[bnd1,...,...], etc...]
"""
## BOUNDARY AND INITIAL CONDITIONS
fun = lambda x: 2/np.cosh(x)

# u(x,0) = 2sech(x), v(x,0) = 0
bnd1_real = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()
bnd1_imag = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()


# u(x,0) = 2sech(x)
bndval1_real = fun(bnd1_real[:,0])

#  v(x,0) = 0
bndval1_imag = torch.from_numpy(np.zeros_like(bnd1_imag[:,0]))


# u(-5,t) = u(5,t)
bnd2_real_left = torch.cartesian_prod(torch.from_numpy(np.array([-5], dtype=np.float64)), t).float()
bnd2_real_right = torch.cartesian_prod(torch.from_numpy(np.array([5], dtype=np.float64)), t).float()
bnd2_real = [bnd2_real_left,bnd2_real_right]


# v(-5,t) = v(5,t)
bnd2_imag_left = torch.cartesian_prod(torch.from_numpy(np.array([-5], dtype=np.float64)), t).float()
bnd2_imag_right = torch.cartesian_prod(torch.from_numpy(np.array([5], dtype=np.float64)), t).float()
bnd2_imag = [bnd2_imag_left,bnd2_imag_right]


# du/dx (-5,t) = du/dx (5,t)
bnd3_real_left = torch.cartesian_prod(torch.from_numpy(np.array([-5], dtype=np.float64)), t).float()
bnd3_real_right = torch.cartesian_prod(torch.from_numpy(np.array([5], dtype=np.float64)), t).float()
bnd3_real = [bnd3_real_left, bnd3_real_right]



bop3_real = {
            'du/dx':
                {
                    'coeff': 1,
                    'du/dx': [0],
                    'pow': 1,
                    'var': 0
                }
}
# dv/dx (-5,t) = dv/dx (5,t)
bnd3_imag_left = torch.cartesian_prod(torch.from_numpy(np.array([-5], dtype=np.float64)), t).float()
bnd3_imag_right = torch.cartesian_prod(torch.from_numpy(np.array([5], dtype=np.float64)), t).float()
bnd3_imag = [bnd3_imag_left,bnd3_imag_right]


bop3_imag = {
            'dv/dx':
                {
                    'coeff': 1,
                    'dv/dx': [0],
                    'pow': 1,
                    'var': 1
                }
}


bcond_type = 'periodic'

bconds = [[bnd1_real, bndval1_real, 0, 'dirichlet'],
          [bnd1_imag, bndval1_imag, 1, 'dirichlet'],
          [bnd2_real, 0, bcond_type],
          [bnd2_imag, 1, bcond_type],
          [bnd3_real, bop3_real, bcond_type],
          [bnd3_imag, bop3_imag, bcond_type]]

'''
schrodinger equation:
i * dh/dt + 1/2 * d2h/dx2 + abs(h)**2 * h = 0 
real part: 
du/dt + 1/2 * d2v/dx2 + (u**2 + v**2) * v
imag part:
dv/dt - 1/2 * d2u/dx2 - (u**2 + v**2) * u
u = var:0
v = var:1
'''

schrodinger_eq_real = {
    'du/dt':
        {
            'coeff': 1,
            'term': [1],
            'pow': 1,
            'var': 0
        },
    '1/2*d2v/dx2':
        {
            'coeff': 1 / 2,
            'term': [0, 0],
            'pow': 1,
            'var': 1
        },
    'v * u**2':
        {
            'coeff': 1,
            'term': [[None], [None]],
            'pow': [1, 2],
            'var': [1, 0]
        },
    'v**3':
        {
            'coeff': 1,
            'term': [None],
            'pow': 3,
            'var': 1
        }

}
schrodinger_eq_imag = {
    'dv/dt':
        {
            'coeff': 1,
            'term': [1],
            'pow': 1,
            'var': 1
        },
    '-1/2*d2u/dx2':
        {
            'coeff': - 1 / 2,
            'term': [0, 0],
            'pow': 1,
            'var': 0
        },
    '-u * v ** 2':
        {
            'coeff': -1,
            'term': [[None], [None]],
            'pow': [1, 2],
            'var': [0, 1]
        },
    '-u ** 3':
        {
            'coeff': -1,
            'term': [None],
            'pow': 3,
            'var': 0
        }

}

schrodinger_eq = [schrodinger_eq_real,schrodinger_eq_imag]

model = torch.nn.Sequential(
        torch.nn.Linear(2, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 2)
    )


equation = Equation(grid, schrodinger_eq, bconds).set_strategy('autograd')

img_dir=os.path.join(os.path.dirname( __file__ ), 'schrodinger_img')

if not(os.path.isdir(img_dir)):
    os.mkdir(img_dir)

model = Solver(grid, equation, model, 'autograd').solve(lambda_bound=1, verbose=1, learning_rate=0.8,
                                    eps=1e-6, tmin=1000, tmax=1e5,use_cache=False,cache_dir='../cache/',cache_verbose=True,
                                    save_always=False,no_improvement_patience=500,print_every = 100, optimizer_mode='LBFGS',step_plot_print=True,step_plot_save=True,image_save_dir=img_dir)
                                    


