import torch
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
import pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.append('../')

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from input_preprocessing import Equation
from solver import Solver
from metrics import Solution
from device import solver_device

solver_device('cpu')

p_l = 1
v_l = 0
Ro_l = 1
gam_l = 1.4

p_r = 0.1
v_r = 0
Ro_r = 0.125
gam_r = 1.4

x0 = 0.5
h = 0.05
x_grid=np.linspace(0,1,15)
t_grid=np.linspace(0,0.2,15)

x = torch.from_numpy(x_grid)
t = torch.from_numpy(t_grid)

grid = torch.cartesian_prod(x, t).float()

## BOUNDARY AND INITIAL CONDITIONS
# p:0, v:1, Ro:2

def u0(x,x0):
  if x>x0:
    return [p_r, v_r, Ro_r]
  else:
    return [p_l, v_l, Ro_l]

# Initial conditions at t=0
bnd1 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()

u_init0 = np.zeros(bnd1.shape[0], dtype=np.float64)
u_init1 = np.zeros(bnd1.shape[0], dtype=np.float64)
u_init2 = np.zeros(bnd1.shape[0], dtype=np.float64)
j=0
for i in bnd1:
  u_init0[j] = u0(i[0], x0)[0]
  u_init1[j] = u0(i[0], x0)[1]
  u_init2[j] = u0(i[0], x0)[2]
  j +=1

bndval1_0 = torch.from_numpy(u_init0)
bndval1_1 = torch.from_numpy(u_init1)
bndval1_2 = torch.from_numpy(u_init2)

bndval1 = torch.stack((bndval1_0, bndval1_1, bndval1_2),dim=1)

#  Boundary conditions at x=0
bnd2 = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), t).float()

bndval2_0 = torch.from_numpy(np.asarray([p_l for i in bnd2[:, 0]], dtype=np.float64))
bndval2_1 = torch.from_numpy(np.asarray([v_l for i in bnd2[:, 0]], dtype=np.float64))
bndval2_2 = torch.from_numpy(np.asarray([Ro_l for i in bnd2[:, 0]], dtype=np.float64))
bndval2 = torch.stack((bndval2_0, bndval2_1, bndval2_2),dim=1)


# Boundary conditions at x=1
bnd3 = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)), t).float()

# u(1,t)=0
bndval3_0 = torch.from_numpy(np.asarray([p_r for i in bnd3[:, 0]], dtype=np.float64))
bndval3_1 = torch.from_numpy(np.asarray([v_r for i in bnd3[:, 0]], dtype=np.float64))
bndval3_2 = torch.from_numpy(np.asarray([Ro_r for i in bnd3[:, 0]], dtype=np.float64))
bndval3 = torch.stack((bndval3_0, bndval3_1, bndval3_2),dim=1)

# Putting all bconds together
bconds = [[bnd1, bndval1_0, 0, 'dirichlet'],
          [bnd1, bndval1_1, 1, 'dirichlet'],
          [bnd1, bndval1_2, 2, 'dirichlet'],
          [bnd2, bndval2_0, 0, 'dirichlet'],
          [bnd2, bndval2_1, 1, 'dirichlet'],
          [bnd2, bndval2_2, 2, 'dirichlet'],
          [bnd3, bndval3_0, 0, 'dirichlet'],
          [bnd3, bndval3_1, 1, 'dirichlet'],
          [bnd3, bndval3_2, 2, 'dirichlet']]


'''
gas dynamic system equations:
Eiler's equations system for Sod test in shock tube

'''
gas_eq1={
        'dro/dt':
        {
            'coeff': 1,
            'term': [1],
            'pow': 1,
            'var': 2
        },
        'v*dro/dx':
        {
            'coeff': 1,
            'term': [[None], [0]],
            'pow': [1, 1],
            'var': [1, 2]
        },
        'ro*dv/dx':
        {
            'coeff': 1,
            'term': [[None], [0]],
            'pow': [1, 1],
            'var': [2, 1]
        }
     }
gas_eq2 = {
        'ro*dv/dt':
        {
            'coeff': 1,
            'term': [[None], [1]],
            'pow': [1, 1],
            'var': [2, 1]
        },
        'ro*v*dv/dx':
        {
            'coeff': 1,
            'term': [[None],[None], [0]],
            'pow': [1, 1, 1],
            'var': [2, 1, 1]
        },
        'dp/dx':
        {
            'coeff': 1,
            'term': [0],
            'pow': 1,
            'var': 0
        }
     }
gas_eq3 =  {
        'dp/dt':
        {
            'coeff': 1,
            'term': [1],
            'pow': 1,
            'var': 0
        },
        'gam*p*dv/dx':
        {
            'coeff': gam_l,
            'term': [[None], [0]],
            'pow': [1, 1],
            'var': [0, 1]
        },
        'v*dp/dx':
        {
            'coeff': 1,
            'term': [[None], [0]],
            'pow': [1, 1],
            'var': [1, 0]
        }

     }

gas_eq = [gas_eq1, gas_eq2, gas_eq3]

model = torch.nn.Sequential(
        torch.nn.Linear(2, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 3)
    )
def v(grid):
    return torch.sin(grid[:,0])+torch.sin(2*grid[:,0])+grid[:,1]


weak_form = [v]

start = time.time()

equation = Equation(grid, gas_eq, bconds, h=h).set_strategy('NN')

img_dir=os.path.join(os.path.dirname( __file__ ), 'SOD_NN_weak_img')

if not(os.path.isdir(img_dir)):
    os.mkdir(img_dir)


model = Solver(grid, equation, model, 'NN', weak_form=weak_form).solve(
                                lambda_bound=100, verbose=True, learning_rate=1e-3,
                                eps=1e-6, tmin=1000, tmax=1e5, use_cache=False, cache_dir='../cache/', cache_verbose=False,
                                save_always=False,no_improvement_patience=500,print_every=100,step_plot_print=False,step_plot_save=True,image_save_dir=img_dir)

end = time.time()