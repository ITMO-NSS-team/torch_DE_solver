# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:25:13 2022

@author: user
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from input_preprocessing import op_dict_to_list,operator_unify





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


wave_eq = {
    '4*d2u/dx2**1':
        {
            'coeff': 4,
            'd2u/dx2': [0, 0],
            'pow': 1
        },
    '-d2u/dt2**1':
        {
            'coeff': -1,
            'd2u/dt2': [1,1],
            'pow':1
        }
}



    
if type(wave_eq)==dict:
    wave_eq=op_dict_to_list(wave_eq)
wave_eq1 = operator_unify(wave_eq)
    
    

x_grid=np.linspace(0,1,100+1)
t_grid=np.linspace(0,1,100+1)

x = torch.from_numpy(x_grid)
t = torch.from_numpy(t_grid)

grid = torch.cartesian_prod(x, t).float()


"""
Preparing boundary conditions (BC)

Unlike KdV example there is optional possibility to define only two items
when boundary operator is not needed

bnd=torch.Tensor of a boundary n-D points where n is the problem
dimensionality

bval=torch.Tensor prescribed values at every point in the boundary

"""

# Initial conditions at t=0
bnd1 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()

# u(0,x)=sin(pi*x)
bndval1 = torch.sin(np.pi * bnd1[:, 0])

# Initial conditions at t=1
bnd2 = torch.cartesian_prod(x, torch.from_numpy(np.array([1], dtype=np.float64))).float()

# u(1,x)=sin(pi*x)
bndval2 = torch.sin(np.pi * bnd2[:, 0])

# Boundary conditions at x=0
bnd3 = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), t).float()

# u(0,t)=0
bndval3 = torch.from_numpy(np.zeros(len(bnd3), dtype=np.float64))

# Boundary conditions at x=1
bnd4 = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)), t).float()

# u(1,t)=0
bndval4 = torch.from_numpy(np.zeros(len(bnd4), dtype=np.float64))

# Putting all bconds together
bconds = [[bnd1, bndval1], [bnd2, bndval2], [bnd3, bndval3], [bnd4, bndval4]]

"""
Defining wave equation

Operator has the form

op=dict in form {'term1':term1,'term2':term2}-> term1+term2+...=0

NB! dictionary keys at the current time serve only for user-frienly 
description/comments and are not used in model directly thus order of
items must be preserved as (coeff,op,pow)
 

term is a dict term={coefficient:c1,[sterm1,sterm2],'pow': power}

c1 may be integer, function of grid or tensor of dimension of grid

Meaning c1*u*d2u/dx2 has the form

{'coefficient':c1,
 'u*d2u/dx2': [[None],[0,0]],
 'pow':[1,1]}

None is for function without derivatives


"""


model = torch.nn.Sequential(
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



optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


lam=1

min_loss=10000000

# cur_loss=min_loss

def closure():
    # nonlocal cur_loss
    optimizer.zero_grad()
    op_part=torch.mean((4*nn_autograd(model, grid, 2,axis=[0,0])-1*nn_autograd(model, grid, 2,axis=[1,1]))**2)
    bcond_part=torch.mean(torch.abs(torch.cat((model(bnd1),model(bnd2),model(bnd3),model(bnd4)))- \
                          torch.cat((bndval1.reshape(-1,1),bndval2.reshape(-1,1),
                                     bndval3.reshape(-1,1),bndval4.reshape(-1,1)))))
    print('op_part={}, bcond_part={}'.format(op_part,bcond_part))
    loss=op_part+lam*bcond_part
    loss.backward()
    # cur_loss = loss.item()
    return loss

# cur_loss=min_loss

t=0

loss=optimizer.step(closure)

while loss>1e-3 and t<1e4:
    loss=optimizer.step(closure)
    if t%100==0:
        print('t={}, loss={}'.format(t,loss.item()))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(grid[:, 0].detach().numpy().reshape(-1), grid[:, 1].detach().numpy().reshape(-1),
                        model(grid).detach().numpy().reshape(-1), cmap=plt.cm.jet, linewidth=0.2, alpha=1)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        plt.show()
    t+=1



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(grid[:, 0].detach().numpy().reshape(-1), grid[:, 1].detach().numpy().reshape(-1),
                model(grid).detach().numpy().reshape(-1), cmap=plt.cm.jet, linewidth=0.2, alpha=1)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
plt.show()
