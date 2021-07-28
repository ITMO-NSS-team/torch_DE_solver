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

x=torch.from_numpy(np.linspace(0,1,10))
t=torch.from_numpy(np.linspace(0,1,10))

grid=torch.cartesian_prod(x,t).float()


grid.to(device)

a1,a2,a3=[1,2,1]

b1,b2,b3=[2,1,3]

r1,r2=[5,5]    


bnd1=torch.cartesian_prod(torch.from_numpy(np.array([0],dtype=np.float64)),t).float()


bop1=[[a1,[0,0],1],[a2,[0],1],[a3,[None],1]]

bndval1=torch.zeros(len(bnd1))

bnd2=torch.cartesian_prod(torch.from_numpy(np.array([1],dtype=np.float64)),t).float()


bop2=[[b1,[0,0],1],[b2,[0],1],[b3,[None],1]]

bndval2=torch.zeros(len(bnd2))


bnd3=torch.cartesian_prod(torch.from_numpy(np.array([1],dtype=np.float64)),t).float()


bop3=[[r1,[0],1],[r2,[None],1]]

bndval3=torch.zeros(len(bnd3))

bndval3=torch.from_numpy(np.zeros(len(bnd3),dtype=np.float64))

bnd4=torch.cartesian_prod(x,torch.from_numpy(np.array([0],dtype=np.float64))).float()

bop4=[[1,[None],1]]

bndval4=torch.zeros(len(bnd4))

bconds=[[bnd1,bop1,bndval1],[bnd2,bop2,bndval2],[bnd3,bop3,bndval3],[bnd4,bop4,bndval4]]



'''
bnd1=torch.cartesian_prod(torch.from_numpy(np.array([0],dtype=np.float64)),t).float()


bop1=[[1,[None],1]]

bndval1=torch.from_numpy(np.array([0., -0.0370864, -0.0679946, -0.0869588, -0.096891, -0.101156, -0.101887, -0.100256, -0.0968864, -0.0921156],dtype=np.float64))

bnd2=torch.cartesian_prod(torch.from_numpy(np.array([1],dtype=np.float64)),t).float()


bop2=[[1,[None],1]]

bndval2=torch.from_numpy(np.array([0., 0.0370053, 0.0373262, 0.0265625, 0.0150148, 0.00553037,-0.00180545, -0.00758799, -0.0123609, -0.0164849],dtype=np.float64))


bnd3=torch.cartesian_prod(x,torch.from_numpy(np.array([1],dtype=np.float64))).float()


bop3=[[1,[None],1]]

bndval3=torch.from_numpy(np.array([-0.0921156, -0.0733226, -0.0578285, -0.0454475, -0.0359152,-0.0288834, -0.0239469, -0.0206109, -0.0183252, -0.0164849],dtype=np.float64))

# bop3=[[1,[None],1]]

bndval3=torch.from_numpy(np.zeros(len(bnd3),dtype=np.float64))

bnd4=torch.cartesian_prod(x,torch.from_numpy(np.array([0],dtype=np.float64))).float()

bop4=[[1,[None],1]]

bndval4=torch.zeros(len(bnd4))



bconds=[[bnd1,bndval1],[bnd2,bndval2],[bnd3,bndval3],[bnd4,bndval4]]
'''

def c1(grid):
    return (-1)*torch.sin(grid[:,0])*torch.cos(grid[:,1])

# operator=[[4,[[0],[None]],[1,1]],[-1,[1,1],1]]

operator=[[1,[1],1],[6,[[None],[0]],[1,1]],[1,[0,0,0],1],[c1,[None],0]]


# operator=[[1,[1],1],[6,[[None],[0]],[1,1]],[c1,[None],0]]

# operator=[[1,[1],1],[6,[[None],[0]],[1,1]],[1,[0,0,0],1]]

# operator=[[4,[0,0],1],[c1,[1,1],1]]

# operator=[[4,[None],1],[-1,[1,1],1]]


torch.autograd.set_detect_anomaly(False)




for _ in range(1):

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 100),
        # torch.nn.Dropout(0.1),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        # torch.nn.Dropout(0.1),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        # torch.nn.Dropout(0.1),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 1)
        )
    
    
    
    start=time.time()
    model=point_sort_shift_solver(grid,model,operator,bconds,lambda_bound=1000,verbose=True,learning_rate = 1e-3,eps=0.001,tmin=1000,tmax=1e5,h=0.01)
    end=time.time()
    
    print('Time taken 10= ',end-start)


