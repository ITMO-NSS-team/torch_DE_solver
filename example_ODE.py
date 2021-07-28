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

t=torch.from_numpy(np.linspace(0,1,100))

grid=t.reshape(-1,1).float()


grid.to(device)

        

bnd1=torch.from_numpy(np.array([[0]],dtype=np.float64))


bop1=[[1,[None],1]]

bndval1=torch.from_numpy(np.array([[-1/2]],dtype=np.float64))

bnd2=torch.from_numpy(np.array([[1]],dtype=np.float64))

bop2=[[1,[0],1]]

bndval2=torch.from_numpy(np.array([[3]],dtype=np.float64))


# bconds=[[bnd1,bndval1],[bnd2,bndval2]]

bconds=[[bnd1,bop1,bndval1],[bnd2,bop2,bndval2]]


def c1(grid):
    return 1-grid**2

def c2(grid):
    return -2*grid



operator=[[c1(grid),[0,0],1],[c2(grid),[0],1],[6,[None],1]]

operator=[[c1,[0,0],1],[c2,[0],1],[6,[None],1]]


for _ in range(1):

    model = torch.nn.Sequential(
        torch.nn.Linear(1, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 1),
        torch.nn.Tanh()
        )
    
    
    start=time.time()
    model=point_sort_shift_solver(grid,model,operator,bconds,lambda_bound=10,verbose=True,learning_rate = 1e-3,eps=0.01,tmin=1000,tmax=1e5)
    end=time.time()
    
    print('Time taken 10= ',end-start)

    fig = plt.figure()
    plt.scatter(grid.reshape(-1),model(grid).detach().numpy().reshape(-1))
    plt.scatter(grid.reshape(-1),1/2*(-1 + 3*grid**2).reshape(-1))
    plt.show()


