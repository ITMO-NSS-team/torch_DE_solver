# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:33:44 2021

@author: user
"""
import torch
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys

sys.path.append('../')

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from solver import *
import time

device = torch.device('cpu')

"""
Preparing grid

Grid is an essentially torch.Tensor of a n-D points where n is the problem
dimensionality
"""

t = torch.from_numpy(np.linspace(0, 1, 100))

grid = t.reshape(-1, 1).float()

grid.to(device)

"""
Preparing boundary conditions (BC)

For every boundary we define three items

bnd=torch.Tensor of a boundary n-D points where n is the problem
dimensionality

bop=list in form [[term1],[term2],...] -> term1+term2+...=0

term is a list term=[coefficient,[sterm1,sterm2],power]

Meaning c1*u*d2u/dx2 has the form

[c1,[[None],[0,0]],[1,1]]

None is for function without derivatives

bval=torch.Tensor prescribed values at every point in the boundary
"""

# point t=0
bnd1 = torch.from_numpy(np.array([[0]], dtype=np.float64))

# May be None as well
bop1 = [[1, [None], 1]]

#  So u(0)=-1/2
bndval1 = torch.from_numpy(np.array([[-1 / 2]], dtype=np.float64))

# point t=1
bnd2 = torch.from_numpy(np.array([[1]], dtype=np.float64))

# d/dt
bop2 = [[1, [0], 1]]

# So, du/dt |_{x=1}=3
bndval2 = torch.from_numpy(np.array([[3]], dtype=np.float64))

# Putting all bconds together
bconds = [[bnd1, bop1, bndval1], [bnd2, bop2, bndval2]]

"""
Defining Legendre polynomials generating equations

Operator has the form

op=list in form [[term1],[term2],...] -> term1+term2+...=0

term is a list term=[coefficient,[sterm1,sterm2],power]

c1 may be function of grid or tensor of dimension of grid.

Meaning c1*u*d2u/dt2 has the form

[c1,[[None],[0,0]],[1,1]]

None is for function without derivatives


"""


# 1-t^2
def c1(grid):
    return 1 - grid ** 2


# -2t
def c2(grid):
    return -2 * grid


operator = [[c1(grid), [0, 0], 1], [c2(grid), [0], 1], [6, [None], 1]]

operator = [[c1, [0, 0], 1], [c2, [0], 1], [6, [None], 1]]

"""
Let's decipher this two (both are equal)

[c1,[0,0],1] -> c1*d2u/dt2 = (1-t^2)*d2u/dt2

[c2,[0],1] -> c2*du/dt = -2t*du/dt


[6,[None],1] -> 6*u=n*(n-1)*u (n=3)


So, operator is  (1-t^2)*d2u/dt2-2t*du/dt+n*(n-1)*u=0 (n=3)

"""

for _ in range(1):
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 1),
        torch.nn.Tanh()
    )

    start = time.time()
    model = point_sort_shift_solver(grid, model, operator, bconds, lambda_bound=10, verbose=True, learning_rate=1e-3,
                                    eps=0.01, tmin=1000, tmax=1e5)
    end = time.time()

    print('Time taken 10= ', end - start)

    fig = plt.figure()
    plt.scatter(grid.reshape(-1), model(grid).detach().numpy().reshape(-1))
    # analytical sln is 1/2*(-1 + 3*t**2)
    plt.scatter(grid.reshape(-1), 1 / 2 * (-1 + 3 * grid ** 2).reshape(-1))
    plt.show()
