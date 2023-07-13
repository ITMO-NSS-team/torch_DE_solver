import torch
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.append('../')

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.input_preprocessing import Equation
from tedeous.solver import Solver
from tedeous.metrics import Solution
from tedeous.device import solver_device


solver_device('cuda')

t = torch.from_numpy(np.linspace(0, 0.25, 300))
grid = t.reshape(-1, 1).float()

h = (t[1]-t[0]).item()

# Initial conditions at t=0
bnd1 = torch.from_numpy(np.array([[0]], dtype=np.float64)).float()
bndval1 = torch.from_numpy(np.array([[-1]], dtype=np.float64))

bnd2 = torch.from_numpy(np.array([[0]], dtype=np.float64)).float()
bop2 = {
        'dy/dt':
            { 'coeff': 1,
              'dy/dt': [0],
              'pow': 1
            }
        }
bndval2 = torch.from_numpy(np.array([[2]], dtype=np.float64))

bconds = [[bnd1, bndval1, 'dirichlet'],
          [bnd2, bop2, bndval2, 'operator']]


def t_func(grid):
    return grid

# y_tt - 10*y_t + 9*y - 5*t = 0

ode = {
        'd2y/dt2':
            {
              'coeff': 1,
              'd2y/dt2': [0,0],
              'pow': 1
            },
        '10*dy/dt':
            {
              'coeff': -10,
              'dy/dt': [0],
              'pow': 1
            },
        '9*y':
            {
              'coeff': 9,
              'dy/dt': [None],
              'pow': 1
            },
        '-5*t':
            {
              'coeff': -5*t_func(grid),
              'dy/dt': [None],
              'pow': 0
            }
      }

model = torch.nn.Sequential(
        torch.nn.Linear(1, 50),
        torch.nn.Tanh(),
        torch.nn.Linear(50, 50),
        torch.nn.Tanh(),
        torch.nn.Linear(50, 50),
        torch.nn.Tanh(),
        torch.nn.Linear(50, 50),
        torch.nn.Tanh(),
        torch.nn.Linear(50, 1)
        )

start = time.time()
equation = Equation(grid, ode, bconds).set_strategy('autograd')

img_dir=os.path.join(os.path.dirname( __file__ ), 'poisson_img')

model = Solver(grid, equation, model, 'autograd').solve(lambda_bound=100,
                                         verbose=1, learning_rate=1e-3, eps=1e-5, tmin=1000, tmax=5e6,
                                         use_cache=False,cache_dir='../cache/',cache_verbose=True,
                                         save_always=True,print_every=2000, gamma=None, lr_decay=1000,
                                         patience=5,loss_oscillation_window=100,no_improvement_patience=1000,
                                         model_randomize_parameter=1e-5,optimizer_mode='Adam',cache_model=None,
                                         step_plot_print=False, step_plot_save=True, image_save_dir=img_dir)
end = time.time()

print(end-start)
def sln(t):
    return 50/81 + (5/9) * t + (31/81) * torch.exp(9*t) - 2 * torch.exp(t)

plt.plot(grid.detach().numpy(), sln(grid).detach().numpy(), label='Exact')
plt.plot(grid.detach().numpy(), model(grid).detach().numpy(), '--', label='Predicted')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper right')
plt.show()