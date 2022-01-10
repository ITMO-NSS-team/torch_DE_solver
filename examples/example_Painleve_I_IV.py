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

# sys.path.append('../')
sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from TEDEouS.solver import *
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

# point t=0
bnd1 = torch.from_numpy(np.array([[0]], dtype=np.float64))

bop1 = None

#  So u(0)=-1/2
bndval1 = torch.from_numpy(np.array([[0]], dtype=np.float64))

# point t=1
bnd2 = torch.from_numpy(np.array([[float(t[0])]], dtype=np.float64))

# d/dt
bop2 ={
        '1*du/dt**1':
            {
                'coeff': 1,
                'du/dt': [0],
                'pow': 1
                }
            
    }
    
    

# So, du/dt |_{x=1}=3
bndval2 = torch.from_numpy(np.array([[0]], dtype=np.float64))

# Putting all bconds together
bconds = [[bnd1, bop1, bndval1], [bnd2, bop2, bndval2]]

"""
Defining Legendre polynomials generating equations

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


# t
def p1_c1(grid):
    return grid



n=3

# P_I operator is  d2u/dt2-6*u^2-t=0 
p_1= {
    '1*d2u/dt2**1':
        {
            'coeff': 1, #coefficient is a torch.Tensor
            'du/dt': [0, 0],
            'pow': 1
        },
    '-6*u**2':
        {
            'coeff': -6,
            'u':  [None],
            'pow': 2
        },
    '-t':
        {
            'coeff': -p1_c1(grid),
            'u':  [None],
            'pow': 0
        }
}



for _ in range(1):
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 1)
        # torch.nn.Tanh()
    )

    start = time.time()
    model = point_sort_shift_solver(grid, model, p_1, bconds, lambda_bound=100, verbose=1, learning_rate=1e-4,
                                    eps=1e-7, tmin=1000, tmax=1e5,use_cache=False,cache_dir='../cache/',cache_verbose=True
                                    ,batch_size=None, save_always=False)
    end = time.time()

    print('Time taken P_I= ', end - start)

a=1
# P_II operator is  d2u/dt2-2*u^3-tu-a=0 
p_2= {
    '1*d2u/dt2**1':
        {
            'coeff': 1, #coefficient is a torch.Tensor
            'du/dt': [0, 0],
            'pow': 1
        },
    '-6*u**2':
        {
            'coeff': -2,
            'u':  [None],
            'pow': 3
        },
    '-t u':
        {
            'coeff': -p1_c1(grid),
            'u':  [None],
            'pow': 1
        },
    '-a':
        {
            'coeff': -a,
            'u':  [None],
            'pow': 0
        }
}



for _ in range(1):
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 1)
        # torch.nn.Tanh()
    )

    start = time.time()
    model = point_sort_shift_solver(grid, model, p_2, bconds, lambda_bound=100, verbose=1, learning_rate=1e-4,
                                    eps=1e-7, tmin=1000, tmax=1e5,use_cache=False,cache_dir='../cache/',cache_verbose=True
                                    ,batch_size=None, save_always=False)
    end = time.time()

    print('Time taken P_II= ', end - start)


def chebgrid(a,b,n):
    k=np.arange(n)+1
    k1=2*k-1
    n1=2*n
    cos_vec=np.cos(k1/n1*np.pi)
    grid=(a+b)/2+(b-a)/2*cos_vec
    grid=np.flip(grid)
    return grid

t = torch.from_numpy(np.linspace(1/4, 2.1, 100))

# t = torch.from_numpy(chebgrid(1/4,2.1,100).copy())

grid = t.reshape(-1, 1).float()

grid.to(device)


# point t=0
bnd1 = torch.from_numpy(np.array([[1]], dtype=np.float64))

bop1 = None

#  So u(0)=-1/2
bndval1 = torch.from_numpy(np.array([[1]], dtype=np.float64))

# point t=1
bnd2 = torch.from_numpy(np.array([[1]], dtype=np.float64))

# d/dt
bop2 ={
        '1*du/dt**1':
            {
                'coeff': 1,
                'du/dt': [0],
                'pow': 1
                }
            
    }
    
    

# So, du/dt |_{x=1}=3
bndval2 = torch.from_numpy(np.array([[0]], dtype=np.float64))


# # point t=0
# bnd1 = torch.from_numpy(np.array([[t[0]]], dtype=np.float64))

# bop1 = None

# #  So u(0)=-1/2
# bndval1 = torch.from_numpy(np.array([[21.8832]], dtype=np.float64))

# # point t=1
# bnd2 = torch.from_numpy(np.array([[t[-1]],[t[-2]]], dtype=np.float64))

# # d/dt
# bop2 =None
    

# # So, du/dt |_{x=1}=3
# bndval2 = torch.from_numpy(np.array([[27.7769],[27.07]], dtype=np.float64))



# Putting all bconds together
bconds = [[bnd1, bop1, bndval1], [bnd2, bop2, bndval2]]


a=1
b=1
g=1
d=1

# P_III operator is  t*u*d2u/dt2-t*(du/dt)^2+u*du/dt-d*t-b*u-a*u^3-g*t*u^4=0 
p_3= {
    't*u*d2u/dt2**1':
        {
            'coeff': p1_c1(grid), #coefficient is a torch.Tensor
            'du/dt': [[None],[0, 0]],
            'pow': [1,1]
        },
    '-t*(du/dt)^2':
        {
            'coeff': -p1_c1(grid),
            'u':  [0],
            'pow': 2
        },
    'u*du/dt':
        {
            'coeff': 1,
            'u':  [[None],[0]],
            'pow': [1,1]
        },
    '-d*t':
        {
            'coeff': -d*p1_c1(grid),
            'u':  [None],
            'pow': 0
        },
    '-b*u':
        {
            'coeff': -b,
            'u':  [None],
            'pow': 1
        },
    '-a*u^3':
        {
            'coeff': -a,
            'u':  [None],
            'pow': 3
        },
    '-g*t*u^4':
        {
            'coeff': -g*p1_c1(grid),
            'u':  [None],
            'pow': 4
        }
        
}



for _ in range(1):
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 1)
        # torch.nn.Tanh()
    )

    start = time.time()
    # model = point_sort_shift_solver(grid, model, p_3, bconds, lambda_bound=1000, verbose=2, h=0.01,learning_rate=1e-4,
    #                                 eps=1e-7, tmin=1000, tmax=1e6,use_cache=False,cache_dir='../cache/',cache_verbose=True
    #                                 ,batch_size=None, save_always=False)
    model = point_sort_shift_solver(grid, model, p_3, bconds, lambda_bound=100, verbose=2, learning_rate=1e-4,
                                    eps=1e-7, tmin=1000, tmax=1e6,use_cache=False,cache_dir='../cache/',cache_verbose=True
                                    ,batch_size=None, save_always=False)
    end = time.time()

    print('Time taken P_III= ', end - start)
