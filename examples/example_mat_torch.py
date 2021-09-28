# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 12:59:47 2021

@author: user
"""
import sys

sys.path.append('../')


import torch
import numpy as np
from input_preprocessing import *

device = torch.device('cpu')

x = torch.from_numpy(np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]))
t = torch.from_numpy(np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]))

new_grid = []

new_grid.append(x)
new_grid.append(t)

grid = np.meshgrid(*new_grid)
grid = torch.tensor(grid, device=device)

grid.to(device)

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
# operator is 4*d2u/dx2-1*d2u/dt2=0
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


model=grid[0]**3*grid[1]**2

def derivative(u_tensor, h_tensor, axis, scheme_order=1, boundary_order=1):

    u_tensor = torch.transpose(u_tensor, 0, axis)
    h_tensor = torch.transpose(h_tensor, 0, axis)

    du_forward = (-torch.roll(u_tensor, -1) + u_tensor) / \
                 (-torch.roll(h_tensor, -1) + h_tensor)
    du_backward = (torch.roll(u_tensor, 1) - u_tensor) / \
                  (torch.roll(h_tensor, 1) - h_tensor)

    du = (1/2) * (du_forward + du_backward)

    # ind = torch.zeros(du.shape, dtype=torch.long, device=device)
    # values = torch.gather(du_forward, 1, ind)
    du[:, 0] = du_forward[:, 0]
    # du = du.scatter_(1, ind, values)

    # ind = (du.shape[axis] - 1) * torch.ones(du.shape, dtype=torch.long, device=device)
    # values = torch.gather(du_backward, 1, ind)
    # du = du.scatter_(1, ind, values)
    du[:, -1] = du_backward[:, -1]

    du = torch.transpose(du, 0, axis)

    return du


def take_matrix_diff_term(model, grid, term):
    """
    Axiluary function serves for single differential operator resulting field
    derivation

    Parameters
    ----------
    model : torch.Sequential
        Neural network.
    term : TYPE
        differential operator in conventional form.

    Returns
    -------
    der_term : torch.Tensor
        resulting field, computed on a grid.

    """
    # it is may be int, function of grid or torch.Tensor
    coeff = term[0]
    # this one contains product of differential operator
    operator_product = term[1]
    # float that represents power of the differential term
    power = term[2]
    # initially it is an ones field
    der_term = torch.zeros_like(model) + 1
    for j, scheme in enumerate(operator_product):
        prod=model
        if scheme!=None:
            for axis in scheme:
                h = grid[axis]
                prod=derivative(prod, h, axis, scheme_order=1, boundary_order=1)
        der_term = der_term * prod ** power[j]
    der_term = coeff * der_term
    return der_term

def apply_matrix_diff_operator(model,grid, operator):
    """
    Deciphers equation in a single grid subset to a field.

    Parameters
    ----------
    model : torch.Sequential
        Neural network.
    operator : list
        Single (len(subset)==1) operator in input form. See 
        input_preprocessing.operator_prepare()

    Returns
    -------
    total : torch.Tensor

    """
    for term in operator:
        dif = take_matrix_diff_term(model, grid, term)
        try:
            total += dif
        except NameError:
            total = dif
    return total

applied_op=apply_matrix_diff_operator(model,grid,wave_eq1)


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

def bnd_prepare_matrix(bconds,grid):
    for bnd in bconds:
        if len(bnd)==2:
            bpts=bnd[0]
            bop=None
            bval=bnd[1]
        else:
            bpts=bnd[0]
            bop=bnd[1]
            bval=bnd[2]
        for pt in bpts:
            t1=torch.isclose(bnd1[1][1],grid[1].float())
            t0=torch.isclose(bnd1[1][0],grid[0].float())
            torch.where(t0*t1==True)
        return prepared_bconds

def apply_matrix_bcond_operator(model,grid,bconds):
    true_b_val_list = []
    b_val_list = []
    b_pos_list = []

    # we apply no  boundary conditions operators if they are all None

    simpleform = False
    for bcond in bconds:
        if bcond[1] == None:
            simpleform = True
        if bcond[1] != None:
            simpleform = False
            break
    if simpleform:
        for bcond in bconds:
            b_pos_list.append(bcond[0])
            true_boundary_val = bcond[2].reshape(-1, 1)
            true_b_val_list.append(true_boundary_val)
        # print(flatten_list(b_pos_list))
        # b_pos=torch.cat(b_pos_list)
        true_b_val = torch.cat(true_b_val_list)
        b_op_val = model(grid)
        b_val = b_op_val[flatten_list(b_pos_list)]
    # or apply differential operatorn first to compute corresponding field and
    else:
        for bcond in bconds:
            b_pos = bcond[0]
            b_cond_operator = bcond[1]
            true_boundary_val = bcond[2].reshape(-1, 1)
            true_b_val_list.append(true_boundary_val)
            if b_cond_operator == None or b_cond_operator == [[1, [None], 1]]:
                b_op_val = model(grid)
            else:
                b_op_val = apply_operator_set(model, b_cond_operator)
            # take boundary values
            b_val_list.append(b_op_val[b_pos])
        true_b_val = torch.cat(true_b_val_list)
        b_val = torch.cat(b_val_list)
    
    return bcond_val,true_val
