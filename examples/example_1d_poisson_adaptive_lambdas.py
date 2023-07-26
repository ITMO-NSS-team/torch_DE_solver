import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.append('../')

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.input_preprocessing import Equation
from tedeous.solver import Solver
from tedeous.solution import Solution
from tedeous.device import solver_device

solver_device('cpu')
a = 4

def u(x, a):
  return torch.sin(torch.pi * a * x)

def u_xx(x, a):
  return (torch.pi * a) ** 2 * torch.sin(torch.pi * a * x)

t0 = 0
tmax = 1
Nt = 100

x = torch.from_numpy(np.linspace(t0, tmax, Nt))
grid = x.reshape(-1, 1).float()

h = (x[1]-x[0]).item()

bnd1 = torch.from_numpy(np.array([[0]], dtype=np.float64)).float()
bndval1 = torch.from_numpy(np.array([[0]], dtype=np.float64))
bnd2 = torch.from_numpy(np.array([[1]], dtype=np.float64)).float()
bndval2  = torch.from_numpy(np.array([[0]], dtype=np.float64))

bconds = [[bnd1, bndval1, 'dirichlet'],
          [bnd2, bndval2, 'dirichlet']]

# equation: d2u/dx2 = -16*pi^2*sin(4*pi*x)

poisson = {
    'd2u/dx2':
        {
        'coeff': 1,
        'term': [0, 0],
        'pow': 1,
        },

    '16*pi^2*sin(4*pi*x)':
        {
        'coeff': u_xx(grid, a),
        'term': [None],
        'pow': 0,
        }
}

model = torch.nn.Sequential(
        torch.nn.Linear(1, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 1)
    )
start = time.time()
equation = Equation(grid, poisson, bconds, h=h).set_strategy('autograd')


img_dir=os.path.join(os.path.dirname( __file__ ), 'poisson_img')

model = Solver(grid, equation, model, 'autograd').solve( lambda_bound=100, lambda_update=True,
                                         verbose=1, learning_rate=1e-3, eps=1e-9, tmin=1000, tmax=5e6,
                                         use_cache=True,cache_dir='../cache/',cache_verbose=True,
                                         save_always=True,print_every=1000, gamma=0.9, lr_decay=1000,
                                         patience=5,loss_oscillation_window=100,no_improvement_patience=1000,
                                         model_randomize_parameter=1e-5,optimizer_mode='Adam',cache_model=None,
                                         step_plot_print=False, step_plot_save=True, image_save_dir=img_dir)

end = time.time()
print(end-start)

plt.plot(grid.detach().numpy(), u(grid,a).detach().numpy(), label='Exact')
plt.plot(grid.detach().numpy(), model(grid).detach().numpy(), '--', label='Predicted')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper right')
plt.show()