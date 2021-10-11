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
from solver import *
import time

device = torch.device('cpu')



def chebgrid(a,b,n):
    k=np.arange(n)+1
    k1=2*k-1
    n1=2*n
    cos_vec=np.cos(k1/n1*np.pi)
    grid=(a+b)/2+(b-a)/2*cos_vec
    grid=np.flip(grid)
    return grid

# t = torch.from_numpy(chebgrid(1/4,2.1,100).copy())

t = torch.from_numpy(np.linspace(1/4, 7/4, 100))



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

# t
def p4_c1(grid):
    return grid


# t^2-a
def p4_c2(grid):
    return grid**2-a


# P_IV operator is  u*d2u/dt2-1/2 (du/dt)^2-b-2*(t^2-a)*u^2-4t*u^3-3/2*u^4
p_4= {
    'u*d2u/dt2**1':
        {
            'coeff': 1, #coefficient is a torch.Tensor
            'du/dt': [[None],[0, 0]],
            'pow': [1,1]
        },
    '-1/2 (du/dt)^2':
        {
            'coeff': -1/2,
            'u':  [0],
            'pow': 2
        },
    '-b':
        {
            'coeff': -b,
            'u':  [None],
            'pow': 0
        },
    '-2*(t^2-a)*u^2':
        {
            'coeff': -2*p4_c2(grid),
            'u':  [None],
            'pow': 2
        },
    '-4t*u^3':
        {
            'coeff': -4*p4_c1(grid),
            'u':  [None],
            'pow': 3
        },
    '-3/2*u^4':
        {
            'coeff': -3/2,
            'u':  [None],
            'pow': 4
        }
        
}



for lr in [1e-3,1e-4]:
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 100),
        torch.nn.Mish(),
        torch.nn.Linear(100, 100),
        torch.nn.Mish(),
        torch.nn.Linear(100, 100),
        torch.nn.Mish(),
        torch.nn.Linear(100, 1)
        # torch.nn.Tanh()
    )

    start = time.time()

    model = point_sort_shift_solver(grid, model, p_4, bconds, lambda_bound=100, verbose=2, learning_rate=lr,
                                    eps=1e-7, tmin=1000, tmax=1e5,use_cache=True,cache_dir='../cache/',cache_verbose=True
                                    ,batch_size=None, save_always=False)
    end = time.time()

    print('Time taken P_IV= ', end - start)
