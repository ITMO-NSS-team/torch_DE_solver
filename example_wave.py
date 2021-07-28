# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:33:44 2021

@author: user
"""
import torch
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from solver import *
import time


device = torch.device('cpu')

x=torch.from_numpy(np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]))
t=torch.from_numpy(np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]))

grid=torch.cartesian_prod(x,t).float()


grid.to(device)

        

bnd1=torch.cartesian_prod(x,torch.from_numpy(np.array([0],dtype=np.float64))).float()


# bop1=[[1,[None],1]]

bndval1=torch.sin(np.pi*bnd1[:,0])

bnd2=torch.cartesian_prod(x,torch.from_numpy(np.array([1],dtype=np.float64))).float()

# bop2=[[1,[None],1]]

bndval2=torch.sin(np.pi*bnd2[:,0])

bnd3=torch.cartesian_prod(torch.from_numpy(np.array([0],dtype=np.float64)),t).float()

# bop3=[[1,[None],1]]

bndval3=torch.from_numpy(np.zeros(len(bnd3),dtype=np.float64))

bnd4=torch.cartesian_prod(torch.from_numpy(np.array([1],dtype=np.float64)),t).float()

# bop4=[[1,[None],1]]

bndval4=torch.from_numpy(np.zeros(len(bnd4),dtype=np.float64))

bconds=[[bnd1,bndval1],[bnd2,bndval2],[bnd3,bndval3],[bnd4,bndval4]]

# bconds=[[bndpos1,bop12,bndval1],[bndpos2,bop22,bndval2],[bndpos3,bop32,bndval3],[bndpos4,bop42,bndval4]]


def c1(grid):
    return torch.sin(grid[:,0])*torch.cos(grid[:,1])

# operator=[[4,[[0],[None]],[1,1]],[-1,[1,1],1]]

operator=[[4,[0,0],1],[-1,[1,1],1]]

# operator=[[4,[0,0],1],[c1,[1,1],1]]

# operator=[[4,[None],1],[-1,[1,1],1]]


# torch.autograd.set_detect_anomaly(False)

for _ in range(1):

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 1)
        )
    
    
    start=time.time()
    model=point_sort_shift_solver(grid,model,operator,bconds,lambda_bound=100,verbose=True,learning_rate = 1e-3,eps=0.01,tmin=1000,tmax=1e5)
    end=time.time()
    
    print('Time taken 10= ',end-start)




