import torch
import torchtext
import SALib
import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt
import scipy

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))


from tedeous.solver import Solver
from tedeous.input_preprocessing import Equation
from tedeous.device import solver_device

solver_device('cpu')
# Grid
x_grid = np.linspace(0,1,21)
t_grid = np.linspace(0,1,21)

x = torch.from_numpy(x_grid)
t = torch.from_numpy(t_grid)

grid = torch.cartesian_prod(x, t).float()

# Boundary and initial conditions

# u(x,0)=1e4*sin^2(x(x-1)/10)

func_bnd1 = lambda x: 10 ** 4 * np.sin((1/10) * x * (x-1)) ** 2
bnd1 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()
bndval1 = func_bnd1(bnd1[:,0])

# du/dx (x,0) = 1e3*sin^2(x(x-1)/10)
func_bnd2 = lambda x: 10 ** 3 * np.sin((1/10) * x * (x-1)) ** 2
bnd2 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()
bop2 = {
        'du/dt':
            {
                'coeff': 1,
                'du/dt': [1],
                'pow': 1,
                'var': 0
            }
}
bndval2 = func_bnd2(bnd2[:,0])

# u(0,t) = u(1,t)
bnd3_left = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)),t).float()
bnd3_right = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)),t).float()
bnd3 = [bnd3_left,bnd3_right]

# du/dt(0,t) = du/dt(1,t)
bnd4_left = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)),t).float()
bnd4_right = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)),t).float()
bnd4 = [bnd4_left,bnd4_right]

bop4= {
        'du/dx':
            {
                'coeff': 1,
                'du/dx': [0],
                'pow': 1,
                'var': 0
            }
}
bcond_type = 'periodic'

bconds = [[bnd1,bndval1, 'dirichlet'],
          [bnd2, bop2, bndval2, 'operator'],
          [bnd3, bcond_type],
          [bnd4,bop4,bcond_type]]

# wave equation is d2u/dt2-(1/4)*d2u/dx2=0
C = 4
wave_eq = {
    'd2u/dt2':
        {
            'coeff': 1,
            'd2u/dt2': [1, 1],
            'pow': 1,
            'var': 0
        },
        '-1/C*d2u/dx2':
        {
            'coeff': -1/C,
            'd2u/dx2': [0, 0],
            'pow': 1,
            'var': 0
        }
}

# NN
model = torch.nn.Sequential(
        torch.nn.Linear(2, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 1))


def v(grid):
    return torch.cos(grid[:,0])+grid[:,1]

weak_form = [v]

start = time.time()

equation = Equation(grid, wave_eq, bconds, h=0.01).set_strategy('NN')

img_dir=os.path.join(os.path.dirname( __file__ ), 'wave_periodic_weak_img')

if not(os.path.isdir(img_dir)):
    os.mkdir(img_dir)


model = Solver(grid, equation, model, 'NN', weak_form=weak_form).solve(lambda_bound=1000, verbose=1, learning_rate=1e-2,
                                    eps=1e-6, tmin=1000, tmax=1e5,use_cache=False,cache_dir='../cache/',cache_verbose=True,
                                    save_always=False,no_improvement_patience=500,print_every=100,step_plot_print=False,step_plot_save=True,image_save_dir=img_dir)

end = time.time()
print('Time taken 10= ', end - start)