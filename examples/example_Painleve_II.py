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


t = torch.from_numpy(np.linspace(0, 1, 100))

grid = t.reshape(-1, 1).float()

grid.to(device)


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




# t
def p1_c1(grid):
    return grid



a=1 #alpha coefficient

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
