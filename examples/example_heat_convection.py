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
from scipy import interpolate

device = torch.device('cuda')

dat=np.load('convection_.npy')

dat=np.transpose(dat)

conv_velocity=np.load('conv_velocity.npy')

conv_velocity=np.transpose(conv_velocity)


train_data=torch.from_numpy(dat.reshape(-1,1))

r = torch.from_numpy(np.array([0.012, 0.017, 0.022]))




# model = torch.nn.Sequential(
# torch.nn.Linear(2, 100),
# torch.nn.Tanh(),
# torch.nn.Linear(100, 100),
# torch.nn.Tanh(),
# torch.nn.Linear(100, 100),
# torch.nn.Tanh(),
# torch.nn.Linear(100, 1)
# # torch.nn.Tanh()
# )

# params = list(model.parameters())


# optimizer = torch.optim.Adam(params, lr=0.0001)
# batch_size = 128 # or whatever

# t=0

# loss_mean=1000
# min_loss=np.inf

# t_init=t
# t_end=t

# while min_loss>1e-5 and t<1e4:
    
#         # X is a torch Variable
#         permutation = torch.randperm(data_grid.size()[0])
        
#         loss_list=[]
        
#         for i in range(0,data_grid.size()[0], batch_size):
#             optimizer.zero_grad()
    
#             indices = permutation[i:i+batch_size]
#             batch_x, batch_y = data_grid[indices], train_data[indices]
    
#             # in case you wanted a semi-full example
#             # outputs = model.forward(batch_x)
#             loss = torch.mean((batch_y-model(batch_x))**2)
#             # loss = torch.mean((batch_y-model(batch_x))**2)
#             loss.backward()
#             optimizer.step()
#             loss_list.append(loss.item())
#         loss_mean=np.mean(loss_list)
#         if loss_mean<min_loss:
#             best_model=model
#             min_loss=loss_mean
#             t_end=t    
#             l2_norm=torch.mean((best_model(data_grid)-train_data)**2)
#             print('t={} steps_taken={}, min_loss={}, l2_norm={}'.format(t, t_end-t_init,min_loss,l2_norm))
#         t_init=t_end
#         t+=1




dat1=dat[:,0:15000:75]

conv_velocity1=conv_velocity[:,0:15000:75]

t_dat=np.array([0+0.002*i for i in range(15000)])

t =torch.from_numpy(t_dat[0:15000:75])


conv_velocity_interp=interpolate.interp2d(r,t,np.transpose(conv_velocity1))

data_grid = torch.cartesian_prod(r, t).float()

data_grid.to(device)


interpt0 = interpolate.interp1d(r, dat1[:,0])

interpr0 = interpolate.interp1d(t, dat1[0,:])

interpr1 = interpolate.interp1d(t, dat1[-1,:])


r = torch.from_numpy(np.arange(0.012, 0.022,0.01/100))
t = t[0:-1:int(len(t)/100)]

conv_velocity_new=np.transpose(conv_velocity_interp(r,t))

conv_velocity_new_1=conv_velocity_new.reshape(-1)

grid = torch.cartesian_prod(r, t).float()

grid.to(device)


"""
Preparing boundary conditions (BC)

Unlike KdV example there is optional possibility to define only two items
when boundary operator is not needed

bnd=torch.Tensor of a boundary n-D points where n is the problem
dimensionality

bval=torch.Tensor prescribed values at every point in the boundary

"""

# Initial conditions at t=0
bnd1 = torch.cartesian_prod(r, torch.from_numpy(np.array([0], dtype=np.float64))).float()

# u(0,x)=sin(pi*x)
bndval1 =torch.from_numpy( interpt0(r))

# # Initial conditions at t=1
# bnd2 = torch.cartesian_prod(r, torch.from_numpy(np.array([1], dtype=np.float64))).float()

# # u(1,x)=sin(pi*x)
# bndval2 = torch.sin(np.pi * bnd2[:, 0])

# Boundary conditions at x=0
bnd3 = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), t).float()

# u(0,t)=0
bndval3 = torch.from_numpy( interpr0(t))

# Boundary conditions at x=1
bnd4 = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)), t).float()

# u(1,t)=0
bndval4 = torch.from_numpy( interpr1(t))

# Putting all bconds together
bconds = [[bnd1, bndval1], [bnd3, bndval3], [bnd4, bndval4]]

def c1(grid):
    return 1/grid[:,0]


def c2(grid):
    return 1/grid[:,1]

# P_VI operator is  
heat= {
    'du/dt':
        {
            'coeff': 1,
            'du/dt':  [1],
            'pow': 1
        },
    'l/t*du/dr':
        {
            'coeff': -6.25e-8*c1(grid),
            'du/dr':  [0],
            'pow': 1
        },
    'l*d2u/dr2':
        {
            'coeff': -5.79e-8,
            'du/dr':  [0,0],
            'pow': 1
        },
    'v*crl*d2u/dr2':
        {
            'coeff': 0.98*conv_velocity_new_1,
            'du/dr':  [0],
            'pow': 1
        },
    '0.0125':
        {
            'coeff': 0.0125,
            'du/dr':  [None],
            'pow': 0
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
torch.nn.Linear(2, 100),
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
    'boundary_p':1,
    'boundary_weighted':True,
    'boundary_normalized':True}

start = time.time()

model = point_sort_shift_solver(grid, model, heat, bconds, lambda_bound=1000, verbose=2,h=0.02/100, learning_rate=1e-5,
                                eps=1e-7, tmin=1000, tmax=1e5,use_cache=True,cache_dir='../cache/',cache_verbose=True
                                ,batch_size=None, save_always=True,lp_par=lp_par)
end = time.time()

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(data_grid[:, 0].reshape(-1), data_grid[:, 1].reshape(-1),
                model(data_grid).detach().numpy().reshape(-1), cmap=cm.jet, linewidth=0.2, alpha=1)
ax.set_xlabel("r")
ax.set_ylabel("t")
ax.set_title('Convection equation solution')
plt.show()

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(data_grid[:, 0].reshape(-1), data_grid[:, 1].reshape(-1),
                dat1.reshape(-1), cmap=cm.jet, linewidth=0.2, alpha=1)
ax.set_xlabel("r")
ax.set_ylabel("t")
ax.set_title('Convection data')
plt.show()


fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(data_grid[:, 0].reshape(-1), data_grid[:, 1].reshape(-1),
                np.abs(model(data_grid).detach().numpy().reshape(-1)-dat1.reshape(-1)), cmap=cm.jet, linewidth=0.2, alpha=1)
ax.set_xlabel("r")
ax.set_ylabel("t")
ax.set_title('absolute error')
plt.show()

