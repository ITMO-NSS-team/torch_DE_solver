from sympy import symbols,Matrix,diff
import numpy as np
import itertools
import torch
import os
import time


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'



def create_poly_terms(variables,degree=1):
    if degree==1:
        return [v for v in variables]
    else:
        terms_list=[]
        for v in create_poly_terms(variables,degree=degree-1):
            for v1 in variables:
                terms_list.append(v*v1)
        return terms_list


#list_flatten =lambda nested_list: [item for sublist in nested_list for item in sublist]

def delete_duplicates(x):
  return list(dict.fromkeys(x))



def poly_model(dim=2,order=2):
    x=[symbols('x'+str(i)) for i in range(dim)]
    terms=[]
    for n in range(1,order+1):
        terms.append(create_poly_terms(x,degree=n))
    terms.reverse()
    terms.append([x[1]**0])
    flattened_list=list(itertools.chain(*terms))
    return delete_duplicates(flattened_list)


def create_subs(symbol,point):
    subs=[]
    for i,symb in enumerate(symbol):
        subs.append((symb,point[i]))
    return subs



def main_system(model,der_grid):
    dim=len(der_grid[1])
    x=[symbols('x'+str(i)) for i in range(dim)]
    system=[]
    for point in der_grid:
        subs_model=[term.subs(create_subs(x,point)) for term in model]
        system.append(subs_model)
    return system


def distances(der_grid,x0):
    return np.linalg.norm(np.array(der_grid)-np.array(x0),axis=1)



def pick_points(der_grid,x0,npoints):
    dist=distances(der_grid,x0)
    picked_points_number=np.argsort(dist)[:npoints]
    return picked_points_number




def lagrange_interp_weights_n(model,interp_grid):
    npts=len(interp_grid)
    sys=np.array(main_system(model,interp_grid),dtype=np.float64)
    #mat=Matrix(sys)
    #det=mat.det()
    #print('cond=',np.linalg.cond(sys))
    if (np.linalg.det(sys)==0):
        print('Warning: We were unable to interpolate surfce at point {} with the current model. Please consider changing either model (order) or adding points to der_grid. Zeros are returned.'.format(interp_grid[0]))
        return [0 for _ in range(npts)]
    weights=[]
    for i in range(npts):
        point_characteristic=np.zeros(npts)
        point_characteristic[i]=1
        coeffs=np.linalg.solve(sys,point_characteristic)
        weights.append(sum(model*coeffs))
    return weights


def compute_weigths(model,der_grid,comp_grid):
    weights=[]
    npoints=len(model)
    if npoints>len(der_grid):
        print('Number of differentiation points less than model parameters')
        return None
    for x0 in comp_grid:
        picked_points=pick_points(der_grid,x0,npoints)
        interp_grid=[der_grid[i] for i in picked_points]
        w=lagrange_interp_weights_n(model,interp_grid)
        weights.append(w)
    return weights


def compute_derivative(weights, axes=[0]):
    x=[symbols('x'+str(i)) for i in range(max(axes)+1)]
    deriv=weights
    for axis in axes:
        deriv=[[diff(model,x[axis]) for model in point] for point in deriv]
    return deriv

interp_model=poly_model(dim=2,order=3)

der_grid=[[1/10*i,1/10*j] for i in range(10) for j in range(10)]

from copy import copy

comp_grid=copy(der_grid)

weights=compute_weigths(interp_model,der_grid,comp_grid)


der1=compute_derivative(weights,axes=[0,0])

der2=compute_derivative(weights,axes=[1,1])

def substitute_points(weights,points):
    dim=len(points[1])
    x=[symbols('x'+str(i)) for i in range(dim)]
    for i in range(len(points)):
        weights[i]=[term.subs(create_subs(x,points[i])) for term in weights[i]]
    return weights


der11=substitute_points(der1,comp_grid)

der21=substitute_points(der2,comp_grid)

picked_points=[pick_points(der_grid,point,len(interp_model)) for point in comp_grid]




def op_loss(NN_model,grid):
    values=NN_model(grid)
    op=torch.zeros_like(values)
    for i in range(grid.shape[0]):
        op[i]=torch.dot(torch.tensor(np.array(der11[i],dtype=np.float32)),values[picked_points[i]].reshape(-1))+torch.dot(torch.tensor(np.array(der21[i],dtype=np.float32)),values[picked_points[i]].reshape(-1))
    return op


NN_model = torch.nn.Sequential(
        torch.nn.Linear(2, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 1))

grid_NN = torch.tensor(der_grid)

loss_op=op_loss(NN_model,grid_NN)

#print(NN_model(grid_NN))

