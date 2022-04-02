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

sys.path.append('../')

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from solver import lbfgs_solution,matrix_optimizer,grid_format_prepare
import time
from scipy.special import legendre
from solver import matrix_cache_lookup
device = torch.device('cpu')
from cache import save_model
from solver import optimization_solver
from config import read_config
from metrics import derivative
import warnings
"""
Preparing grid

Grid is an essentially torch.Tensor of a n-D points where n is the problem
dimensionality
"""

grid=torch.from_numpy(np.linspace(0,1,100))

grid=grid.reshape(-1,1).float()

data_norm=grid**3


data_norm=data_norm.reshape(-1,1).float()

model = torch.nn.Sequential(
    torch.nn.Linear(1, 256),
    torch.nn.Tanh(),
    # torch.nn.Dropout(0.1),
    # torch.nn.ReLU(),
    torch.nn.Linear(256, 64),
    # # torch.nn.Dropout(0.1),
    torch.nn.Tanh(),
    torch.nn.Linear(64, 1024),
    # torch.nn.Dropout(0.1),
    torch.nn.Tanh(),
    torch.nn.Linear(1024, 1)
    # torch.nn.Tanh()
)



optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# l1_lambda = 0.001
# l1_norm =sum(p.abs().sum() for p in model.parameters())




def train_surface(model,optimizer,grid,data_norm,batch_size=128,eps=1e-6,tmax=1e5):
    t=0

    loss_mean=1000
    min_loss=np.inf


    while loss_mean>eps and t<tmax:

        # X is a torch Variable
        permutation = torch.randperm(grid.size()[0])
        
        loss_list=[]
        
        for i in range(0,grid.size()[0], batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i+batch_size]
            batch_x, batch_y = grid[indices], data_norm[indices]

            # in case you wanted a semi-full example
            # outputs = model.forward(batch_x)
            loss = torch.mean(torch.abs(batch_y-model(batch_x)))

            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        loss_mean=np.mean(loss_list)
        if loss_mean<min_loss:
            best_model=model
            min_loss=loss_mean
        print('Surface trainig t={}, loss={}'.format(t,loss_mean))
        t+=1
    return best_model

# grid_test=grid_test.reshape(-1,1).float()



def nn_autograd_simple(model, points, order,axis=0):
    points.requires_grad=True
    f = model(points).sum()
    for i in range(order):
        grads, = torch.autograd.grad(f, points, create_graph=True)
        f = grads[:,axis].sum()
    return grads[:,axis]


def nn_autograd_mixed(model, points,axis=[0]):
    points.requires_grad=True
    f = model(points).sum()
    for ax in axis:
        grads, = torch.autograd.grad(f, points, create_graph=True)
        f = grads[:,ax].sum()
    return grads[:,axis[-1]]



def nn_autograd(*args,axis=0):
    model=args[0]
    points=args[1]
    if len(args)==3:
        order=args[2]
        grads=nn_autograd_simple(model, points, order,axis=axis)
    else:
        grads=nn_autograd_mixed(model, points,axis=axis)
    return grads



model=train_surface(model,optimizer,grid,data_norm)



d1=nn_autograd(model, grid, 1)

plt.plot(grid.detach().numpy(),model(grid).detach().numpy())

plt.figure()

plt.plot(grid.detach().numpy(),d1.detach().numpy())

plt.plot(grid.detach().numpy(),(3*grid**2).detach().numpy())

d2=nn_autograd(model, grid, 2)

plt.figure()

plt.plot(grid.detach().numpy(),d2.detach().numpy())

plt.plot(grid.detach().numpy(),(6*grid).detach().numpy())


x = torch.from_numpy(np.linspace(0, 1, 100+1))
t = torch.from_numpy(np.linspace(0, 1, 100+1))

grid_2d = torch.cartesian_prod(x, t).float()

data_norm_2d=(grid_2d[:,0]**2*grid_2d[:,1]**3).reshape(-1,1)

model2d = torch.nn.Sequential(
    torch.nn.Linear(2, 256),
    torch.nn.Tanh(),
    # torch.nn.Dropout(0.1),
    # torch.nn.ReLU(),
    torch.nn.Linear(256, 64),
    # # torch.nn.Dropout(0.1),
    torch.nn.Tanh(),
    torch.nn.Linear(64, 1024),
    # torch.nn.Dropout(0.1),
    torch.nn.Tanh(),
    torch.nn.Linear(1024, 1)
    # torch.nn.Tanh()
)



optimizer2d = torch.optim.SGD(model2d.parameters(), lr=0.0001)


model2d=train_surface(model2d,optimizer2d,grid_2d,data_norm_2d,tmax=1e4)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(grid_2d[:, 0].detach().numpy().reshape(-1), grid_2d[:, 1].detach().numpy().reshape(-1),
                model2d(grid_2d).detach().numpy().reshape(-1), cmap=cm.jet, linewidth=0.2, alpha=1)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
plt.show()


d1_2d_0=nn_autograd(model2d, grid_2d, 1,axis=0)

print(torch.mean((d1_2d_0-2*grid_2d[:,0]*grid_2d[:,1]**3)**2))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(grid_2d[:, 0].detach().numpy().reshape(-1), grid_2d[:, 1].detach().numpy().reshape(-1),
                d1_2d_0.detach().numpy().reshape(-1), cmap=cm.jet, linewidth=0.2, alpha=1)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(grid_2d[:, 0].detach().numpy().reshape(-1), grid_2d[:, 1].detach().numpy().reshape(-1),
                (2*grid_2d[:,0]*grid_2d[:,1]**3).detach().numpy().reshape(-1), cmap=cm.jet, linewidth=0.2, alpha=1)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
plt.show()


d1_2d_1=nn_autograd(model2d, grid_2d, 1,axis=1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(grid_2d[:, 0].detach().numpy().reshape(-1), grid_2d[:, 1].detach().numpy().reshape(-1),
                d1_2d_1.detach().numpy().reshape(-1), cmap=cm.jet, linewidth=0.2, alpha=1)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(grid_2d[:, 0].detach().numpy().reshape(-1), grid_2d[:, 1].detach().numpy().reshape(-1),
                (3*grid_2d[:,0]**2*grid_2d[:,1]**2).detach().numpy().reshape(-1), cmap=cm.jet, linewidth=0.2, alpha=1)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
plt.show()


print(torch.mean((d1_2d_1-3*grid_2d[:,0]**2*grid_2d[:,1]**2)**2))

d1_2d_1_1=nn_autograd(model2d, grid_2d, 2,axis=1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(grid_2d[:, 0].detach().numpy().reshape(-1), grid_2d[:, 1].detach().numpy().reshape(-1),
                (d1_2d_1_1).detach().numpy().reshape(-1), cmap=cm.jet, linewidth=0.2, alpha=1)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
plt.show()


realdata={'xx':(2*grid_2d[:,1]**3),'xt':(6*grid_2d[:,0]*grid_2d[:,1]**2),'tx':(6*grid_2d[:,0]*grid_2d[:,1]**2),
         'tt':6*grid_2d[:,0]**2*grid_2d[:,1]}

for der in [[0,0],[0,1],[1,0],[1,1]]:
    minerr=10000000
    for rd in ['xx','xt','tx','tt']:
        err=torch.mean((nn_autograd(model2d, grid_2d,axis=der)-realdata[rd])**2)
        if err<minerr:
            minrd=rd
            minerr=err
    print('der={},rd={},minerr={}'.format(der,minrd,minerr))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(grid_2d[:, 0].detach().numpy().reshape(-1), grid_2d[:, 1].detach().numpy().reshape(-1),
                (6*grid_2d[:,0]**2*grid_2d[:,1]).detach().numpy().reshape(-1), cmap=cm.jet, linewidth=0.2, alpha=1)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
plt.show()





