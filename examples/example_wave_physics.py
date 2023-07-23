# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:33:44 2021

@author: user
"""
import torch
import numpy as np
import os
import sys
import time


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.append('../')

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.input_preprocessing import Equation
from tedeous.solver import Solver
from tedeous.solution import Solution
from tedeous.device import solver_device

"""
Preparing grid

Grid is an essentially torch.Tensor  of a n-D points where n is the problem
dimensionality
"""

solver_device('gpu')



def func(grid):
    x, t = grid[:,0],grid[:,1]
    sln=np.cos(2*np.pi*t)*np.sin(np.pi*x)
    return sln



x_grid=np.linspace(0,1,20+1)
t_grid=np.linspace(0,1,20+1)

x = torch.from_numpy(x_grid)
t = torch.from_numpy(t_grid)

#h=abs((t[1]-t[0]).item())

grid = torch.cartesian_prod(x, t).float()


"""
Preparing boundary conditions (BC)

For every boundary we define three items

bnd=torch.Tensor of a boundary n-D points where n is the problem
dimensionality

bop=dict in form {'term1':term1,'term2':term2}-> term1+term2+...=0

NB! dictionary keys at the current time serve only for user-frienly 
description/comments and are not used in model directly thus order of
items must be preserved as (coeff,op,pow)

term is a dict term={coefficient:c1,[sterm1,sterm2],'pow': power}

Meaning c1*u*d2u/dx2 has the form

{'coefficient':c1,
 'u*d2u/dx2': [[None],[0,0]],
 'pow':[1,1]}

None is for function without derivatives


bval=torch.Tensor prescribed values at every point in the boundary
"""

# Initial conditions at t=0
bnd1 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()

# u(0,x)=sin(pi*x)
bndval1 = func(bnd1)



## Initial conditions at t=1
#bnd2 = torch.cartesian_prod(x, torch.from_numpy(np.array([1], dtype=np.float64))).float()

## u(1,x)=sin(pi*x)
#bndval2 = func(bnd2)

bnd2 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()

bop2 = {
'du/dt':
    {
        'coeff': 1,
        'du/dx': [1],
        'pow': 1,
        'var':0
    }
    }

bndval2=torch.from_numpy(np.zeros(len(bnd2), dtype=np.float64))

# Boundary conditions at x=0
bnd3 = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), t).float()

# u(0,t)=0
bndval3 = func(bnd3)

# Boundary conditions at x=1
bnd4 = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)), t).float()


# u(1,t)=0
bndval4 = func(bnd4)





# Putting all bconds together
bconds = [[bnd1, bndval1, 'dirichlet'],
          #[bnd2, bndval2, 'dirichlet'],
          [bnd2,bop2, bndval2, 'operator'],
          [bnd3, bndval3, 'dirichlet'],
          [bnd4, bndval4, 'dirichlet'],
          ]


"""
Defining wave equation

Operator has the form

op=dict in form {'term1':term1,'term2':term2}-> term1+term2+...=0

NB! dictionary keys at the current time serve only for user-frienly 
description/comments and are not used in model directly thus order of
items must be preserved as (coeff,op,pow)



term is a dict term={coefficient:c1,[sterm1,sterm2],'pow': power}

c1 may be integer, function of grid or tensor of dimension of grid

Meaning c1*u*d2u/dx2 has the form

{'coefficient':c1,
 'u*d2u/dx2': [[None],[0,0]],
 'pow':[1,1]}

None is for function without derivatives


"""
# operator is 4*d2u/dx2-1*d2u/dt2=0
wave_eq = {
    'd2u/dt2**1':
        {
            'coeff': 1.,
            'd2u/dt2': [1,1],
            'pow':1
        },
        '-C*d2u/dx2**1':
        {
            'coeff': -4,
            'd2u/dx2': [0, 0],
            'pow': 1
        }
}


model = torch.nn.Sequential(
     torch.nn.Linear(2, 100),
     torch.nn.Tanh(),
     torch.nn.Linear(100, 100),
     torch.nn.Tanh(),
     torch.nn.Linear(100, 100),
     torch.nn.Tanh(),
     torch.nn.Linear(100, 100),
     torch.nn.Tanh(),
     torch.nn.Linear(100, 1)
 )






start = time.time()

equation = Equation(grid, wave_eq, bconds).set_strategy('autograd')

img_dir=os.path.join(os.path.dirname( __file__ ), 'wave_example_physics_img')

if not(os.path.isdir(img_dir)):
    os.mkdir(img_dir)



model = Solver(grid, equation, model, 'autograd').solve(lambda_bound=100,verbose=1, learning_rate=1e-3,
                                            eps=1e-4, tmin=1000, tmax=1e5,use_cache=True,cache_verbose=True,
                                            save_always=True,print_every=500,model_randomize_parameter=1e-6,
                                            optimizer_mode='Adam',no_improvement_patience=1000,step_plot_print=False,step_plot_save=True,image_save_dir=img_dir)


end = time.time()
