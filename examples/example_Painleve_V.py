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

device = torch.device('cuda')



def chebgrid(a,b,n):
    k=np.arange(n)+1
    k1=2*k-1
    n1=2*n
    cos_vec=np.cos(k1/n1*np.pi)
    grid=(a+b)/2+(b-a)/2*cos_vec
    grid=np.flip(grid)
    return grid

# t = torch.from_numpy(chebgrid(1/4, 7/4,500).copy())

t = torch.from_numpy(np.linspace(0.9, 1.2, 100))



grid = t.reshape(-1, 1).float()

grid.to(device)


# point t=0
bnd1 = torch.from_numpy(np.array([[0.9]], dtype=np.float64))

bop1 = None

#  So u(0)=-1/2
bndval1 = torch.from_numpy(np.array([[2.979325765542575]], dtype=np.float64))

# point t=0
bnd2 = torch.from_numpy(np.array([[1.2]], dtype=np.float64))

bop2 = None

#  So u(0)=-1/2
bndval2 = torch.from_numpy(np.array([[4.117945909429169]], dtype=np.float64))




# # point t=0
# bnd1 = torch.from_numpy(np.array([[t[0]]], dtype=np.float64))

# bop1 = None

# #  So u(0)=-1/2
# bndval1 = torch.from_numpy(np.array([[4.63678]], dtype=np.float64))

# # point t=1
# bnd2 = torch.from_numpy(np.array([[t[-1]]], dtype=np.float64))

# # d/dt
# bop2 =None
    

# # So, du/dt |_{x=1}=3
# bndval2 = torch.from_numpy(np.array([[167.894]], dtype=np.float64))



# Putting all bconds together
bconds = [[bnd1, bop1, bndval1], [bnd2, bop2, bndval2]]


a=1
b=1
g=1
d=1



# t^2
def p5_c1(grid):
    return grid**2

# 2*(a+3b+t(g-td))
def p5_c2(grid):
    return 2*(a+3*b+grid*(g-d*grid))

# 2*(3a+3+t(g-td))
def p5_c3(grid):
    return -2*(3*a+b+grid*(g+d*grid))

# t
def p5_c4(grid):
    return grid

# P_V operator is  u*d2u/dt2-1/2 (du/dt)^2-b-2*(t^2-a)*u^2-4t*u^3-3/2*u^4
p_5= {
    '-2*t^2*u*d2u/dt2**1':
        {
            'coeff': -2*p5_c1(grid),
            'du/dt': [[None],[0, 0]],
            'pow': [1,1]
        },
    '2*t^2*u**2*d2u/dt2**1':
        {
            'coeff': 2*p5_c1(grid),
            'du/dt': [[None],[0, 0]],
            'pow': [2,1]
        },
    '2*b':
        {
            'coeff': 2*b,
            'u':  [None],
            'pow': 0
        },
    '-6*b*u':
        {
            'coeff': -6*b,
            'u':  [None],
            'pow': 1
        },
    '2*(a+3b+t(g-td))*u**2':
        {
            'coeff': p5_c2(grid),
            'u':  [None],
            'pow': 2
        },
    '-2*(3a+b+t(g+td))*u**3':
        {
            'coeff': p5_c3(grid),
            'u':  [None],
            'pow': 3
        },
    '6*a*u**4':
        {
            'coeff': 6*a,
            'u':  [None],
            'pow': 4
        },
    '-2*a*u**5':
        {
            'coeff': -2*a,
            'u':  [None],
            'pow': 5
        },
    '-2*t*u*du/dt':
        {
            'coeff': -2*p5_c4(grid),
            'du/dt': [[None],[0]],
            'pow': [1,1]
        }, 
    '2*t*u**2*du/dt':
        {
            'coeff': 2*p5_c4(grid),
            'du/dt': [[None],[0]],
            'pow': [2,1]
        }, 
    't**2*du/dt**2':
        {
            'coeff': p5_c1(grid),
            'du/dt': [0],
            'pow': 2
        }, 
    '-3*t**2*u*du/dt**2':
        {
            'coeff': -3*p5_c1(grid),
            'du/dt': [[None],[0]],
            'pow': [1,2]
        }
}




# model = torch.nn.Sequential(
#     torch.nn.Linear(1, 100),
#     torch.nn.LayerNorm(100,100),
#     torch.nn.Mish(),
#     torch.nn.Linear(100, 100),
#     torch.nn.Mish(),
#     torch.nn.Linear(100, 100),
#     torch.nn.Mish(),
#     torch.nn.Linear(100, 1)
# )

model = torch.nn.Sequential(
torch.nn.Linear(1, 100),
torch.nn.Tanh(),
torch.nn.Linear(100, 100),
torch.nn.Tanh(),
torch.nn.Linear(100, 100),
torch.nn.Tanh(),
torch.nn.Linear(100, 1)
)

# lp_par={'operator_p':2,
#     'operator_weighted':False,
#     'operator_normalized':False,
#     'boundary_p':2,
#     'boundary_weighted':False,
#     'boundary_normalized':False}


lp_par={'operator_p':2,
    'operator_weighted':True,
    'operator_normalized':True,
    'boundary_p':2,
    'boundary_weighted':True,
    'boundary_normalized':True}

start = time.time()

model = point_sort_shift_solver(grid, model, p_5, bconds, lambda_bound=1000, verbose=2,h=abs((t[1]-t[0]).item()), learning_rate=1e-5,
                                eps=1e-7, tmin=1000, tmax=1e5,use_cache=True,cache_dir='../cache/',cache_verbose=True
                                ,batch_size=None, save_always=True,lp_par=lp_par)
end = time.time()

print('Time taken P_V= ', end - start)
