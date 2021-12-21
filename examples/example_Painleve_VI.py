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

t = torch.from_numpy(np.linspace(1.2, 1.4, 101))



grid = t.reshape(-1, 1).float()

grid.to(device)


# point t=0
bnd1 = torch.from_numpy(np.array([[1.2]], dtype=np.float64))

bop1 = None

#  So u(0)=-1/2
bndval1 = torch.from_numpy(np.array([[2]], dtype=np.float64))

# point t=0
bnd2 = torch.from_numpy(np.array([[1.4]], dtype=np.float64))

bop2 = None

#  So u(0)=-1/2
bndval2 = torch.from_numpy(np.array([[2]], dtype=np.float64))




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



p6coeff={0:-grid**3*b,
         1:2*grid**2*(1 + grid)*b,
         2:-grid*(b-d+grid*(a+(4+grid)*b+(-1+grid)*g+d)),
         3:2*grid*((1+grid)*a+(1+grid)*b+(-1+grid)*(g+d)),
         4:-(a+grid*(4+grid)*a-g+grid*(b+g+(-1+grid)*d)),
         5:2*(1+grid)*a,
         6:(-1+grid)*grid**3,
         7:-grid*(1+grid*(-3+grid+grid**2)),
         8:(-1+grid)*grid*(-1+2*grid),
         9:-(1/2)*(-1+grid)**2*grid**3,
         10:(-1+grid)**2*grid**2*(1+grid),
         11:-(3/2)*(-1+grid)**2*grid**2,
         12:(-1+grid)**2*grid**3,
         13:-(-1+grid)**2*grid**2*(1+grid),
         14:(-1+grid)**2*grid**2}


# P_VI operator is  
p_6= {
    '-t**3*b':
        {
            'coeff': p6coeff[0],
            'u':  [None],
            'pow': 0
        },
    'u':
        {
            'coeff': p6coeff[1],
            'u':  [None],
            'pow': 1
        },
    'u**2':
        {
            'coeff': p6coeff[2],
            'u':  [None],
            'pow': 2
        },
    'u**3':
        {
            'coeff': p6coeff[3],
            'u':  [None],
            'pow': 3
        },
    'u**4':
        {
            'coeff': p6coeff[4],
            'u':  [None],
            'pow': 4
        },
    'u**5':
        {
            'coeff': p6coeff[5],
            'u':  [None],
            'pow': 5
        },
    'u**6':
        {
            'coeff': -a,
            'u':  [None],
            'pow': 6
        },
    'u*du/dt':
        {
            'coeff': p6coeff[6],
            'du/dt': [[None],[0]],
            'pow': [1,1]
        }, 
    'u**2*du/dt':
        {
            'coeff': p6coeff[7],
            'du/dt': [[None],[0]],
            'pow': [2,1]
        }, 
    'u**3*du/dt':
        {
            'coeff': p6coeff[8],
            'du/dt': [[None],[0]],
            'pow': [3,1]
        }, 
    'du/dt**2':
        {
            'coeff': p6coeff[9],
            'du/dt': [0],
            'pow': 2
        },
    'u*du/dt**2':
        {
            'coeff': p6coeff[10],
            'du/dt': [[None],[0]],
            'pow': [1,2]
        }, 
    'u**2*du/dt**2':
        {
            'coeff': p6coeff[11],
            'du/dt': [[None],[0]],
            'pow': [2,2]
        }, 
    'u*d2u/dt2':
        {
            'coeff': p6coeff[12],
            'du/dt': [[None],[0,0]],
            'pow': [1,1]
        },
    'u**2*d2u/dt2':
        {
            'coeff': p6coeff[13],
            'du/dt': [[None],[0,0]],
            'pow': [2,1]
        },
    'u**3*d2u/dt2':
        {
            'coeff': p6coeff[14],
            'du/dt': [[None],[0,0]],
            'pow': [3,1]
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

model = point_sort_shift_solver(grid, model, p_6, bconds, lambda_bound=100, verbose=2,h=abs((t[1]-t[0]).item()), learning_rate=1e-5,
                                eps=1e-7, tmin=1000, tmax=1e5,use_cache=True,cache_dir='../cache/',cache_verbose=True
                                ,batch_size=None, save_always=True,lp_par=lp_par)
end = time.time()

print('Time taken P_VI= ', end - start)

sln=np.genfromtxt('wolfram_sln/P_VI_sln_'+str(100)+'.csv',delimiter=',')
sln_torch=torch.from_numpy(sln)
sln_torch1=sln_torch.reshape(-1,1)

fig = plt.figure()
plt.scatter(grid.reshape(-1), model(grid).detach().numpy().reshape(-1),label='NN solution')
plt.scatter(grid.reshape(-1), sln_torch1.detach().numpy().reshape(-1),label='Wolfram Solution')
plt.legend()
plt.show()


