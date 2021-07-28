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


'''
Preparing grid

Grid is a essentially torch.Tensor of a n-D points where n is the problem
dimensionality
'''

device = torch.device('cpu')

x=torch.from_numpy(np.linspace(0,1,10))
t=torch.from_numpy(np.linspace(0,1,10))

grid=torch.cartesian_prod(x,t).float()


grid.to(device)


'''
Preparing boundary conditions (BC)

For every boundary we define three items

bnd=torch.Tensor of a boundary n-D points where n is the problem
dimensionality

bop=list in form [[term1],[term2],...] -> term1+term2+...=0

term is a list term=[coefficient,[sterm1,sterm2],power]

Meaning c1*u*d2u/dx2 has the form

[c1,[[None],[0,0]],[1,1]]

None is for function without derivatives


'''

#coefficients for BC

a1,a2,a3=[1,2,1]

b1,b2,b3=[2,1,3]

r1,r2=[5,5]    

'''
Boundary x=0
'''

# points
bnd1=torch.cartesian_prod(torch.from_numpy(np.array([0],dtype=np.float64)),t).float()


#operator a1*d2u/dx2+a2*du/dx+a3*u
bop1=[[a1,[0,0],1],[a2,[0],1],[a3,[None],1]]

#equal to zero
bndval1=torch.zeros(len(bnd1))


'''
Boundary x=1
'''

# points
bnd2=torch.cartesian_prod(torch.from_numpy(np.array([1],dtype=np.float64)),t).float()

#operator b1*d2u/dx2+b2*du/dx+b3*u
bop2=[[b1,[0,0],1],[b2,[0],1],[b3,[None],1]]

#equal to zero
bndval2=torch.zeros(len(bnd2))


'''
Another boundary x=1
'''
# points
bnd3=torch.cartesian_prod(torch.from_numpy(np.array([1],dtype=np.float64)),t).float()

#operator r1*du/dx+r2*u
bop3=[[r1,[0],1],[r2,[None],1]]

#equal to zero
bndval3=torch.zeros(len(bnd3))

'''
Initial conditions at t=0
'''

bnd4=torch.cartesian_prod(x,torch.from_numpy(np.array([0],dtype=np.float64))).float()

#No operator applied,i.e. u(x,0)=0, may be as well None
bop4=[[1,[None],1]]


#equal to zero
bndval4=torch.zeros(len(bnd4))


#Putting all bconds together
bconds=[[bnd1,bop1,bndval1],[bnd2,bop2,bndval2],[bnd3,bop3,bndval3],[bnd4,bop4,bndval4]]



'''
Defining kdv equation

Operator has the form

op=list in form [[term1],[term2],...] -> term1+term2+...=0

term is a list term=[coefficient,[sterm1,sterm2],power]

c1 may be function of grid or tensor of dimension of grid.

Meaning c1*u*d2u/dx2 has the form

[c1,[[None],[0,0]],[1,1]]

None is for function without derivatives


'''


def c1(grid):
    return (-1)*torch.sin(grid[:,0])*torch.cos(grid[:,1])


operator=[[1,[1],1],[6,[[None],[0]],[1,1]],[1,[0,0,0],1],[c1,[None],0]]

'''
Let's decipher this one

[1,[1],1] -> du/dt


[6,[[None],[0]],[1,1]] -> 6*u**1*du/dx**1=6u*(du/dx)


[1,[0,0,0],1]-> d3u/dx3


[c1,[None],0] -> -sin(x)*cos(t)

So, operator is du/dt+6u*(du/dx)+d3u/dx3-sin(x)*cos(t)=0

'''

'''
Solving equation
'''
for _ in range(1):

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 1)
        )
    
    
    
    start=time.time()
    model=point_sort_shift_solver(grid,model,operator,bconds,lambda_bound=1000,verbose=True,learning_rate = 1e-3,eps=0.001,tmin=1000,tmax=1e5,h=0.01)
    end=time.time()
    
    print('Time taken 10= ',end-start)


