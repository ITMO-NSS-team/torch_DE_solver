# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 15:06:41 2021

@author: Sashka
"""
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from input_preprocessing import grid_prepare, bnd_prepare, operator_prepare,batch_bconds_transform
from points_type import grid_sort
import numpy as np
from cache import cache_lookup,cache_retrain,save_model



def take_derivative_shift_op(model, term):
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
    # this one contains shifted grids (see input_preprocessing module)
    shift_grid_list = term[1]
    # signs corresponding to a grid
    s_order_norm_list = term[2]
    # float that represents power of the differential term
    power = term[3]
    # initially it is an ones field
    der_term = torch.zeros_like(model(shift_grid_list[0][0])) + 1
    for j, scheme in enumerate(shift_grid_list):
        # every shift in grid we should add with correspoiding sign, so we start
        # from zeros
        grid_sum = torch.zeros_like(model(scheme[0]))
        for k, grid in enumerate(scheme):
            # and add grid sequentially
            grid_sum += model(grid) * s_order_norm_list[j][k]
        # Here we want to apply differential operators for every term in the product
        der_term = der_term * grid_sum ** power[j]
    der_term = coeff * der_term
    return der_term


def apply_const_shift_operator(model, operator):
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
        dif = take_derivative_shift_op(model, term)
        try:
            total += dif
        except NameError:
            total = dif
    return total


def apply_operator_set(model, operator_set):
    """
    Deciphers equation in a whole grid to a field.

    Parameters
    ----------
    model : torch.Sequential
        Neural network.
    operator : list
        Multiple (len(subset)>=1) operators in input form. See 
        input_preprocessing.operator_prepare()

    Returns
    -------
    total : torch.Tensor

    """
    field_part = []
    for operator in operator_set:
        field_part.append(apply_const_shift_operator(model, operator))
    field_part = torch.cat(field_part)
    return field_part


flatten_list = lambda t: [item for sublist in t for item in sublist]


def point_sort_shift_loss(model, grid, operator_set, bconds, lambda_bound=10):

    op = apply_operator_set(model, operator_set)
    
    if bconds==None:
        loss = torch.mean((op) ** 2)
        return loss
    
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

    """
    actually, we can use L2 norm for the operator and L1 for boundary
    since L2>L1 and thus, boundary values become not so signifnicant, 
    so the NN converges faster. On the other hand - boundary conditions is the
    crucial thing for all that stuff, so we should increase significance of the
    coundary conditions
    """
    # l1_lambda = 0.001
    # l1_norm =sum(p.abs().sum() for p in model.parameters())
    # loss = torch.mean((op) ** 2) + lambda_bound * torch.mean((b_val - true_b_val) ** 2)+ l1_lambda * l1_norm
    
    loss = torch.mean((op) ** 2) + lambda_bound * torch.mean((b_val - true_b_val) ** 2)
    
    return loss

def point_sort_shift_loss_batch(model, prepared_grid, point_type, operator, bconds,subset=['central'], lambda_bound=10,batch_size=32,h=0.001):
    permutation = torch.randperm(prepared_grid.size()[0])
    loss=0
    batch_num=0
    for i in range(0,prepared_grid.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        if len(indices)<5:
            continue
        # batch= grid[indices]
        
        # batch_grid = grid_prepare(batch)
        batch_grid=prepared_grid[indices]


        batch_types=np.array(list(point_type.values()))[indices.tolist()]
        
        batch_type=dict(zip(batch_grid, batch_types))
        
        batch_dict=grid_sort(batch_type)
        batch_bconds=batch_bconds_transform(batch_grid,bconds)
        batch_bconds = bnd_prepare(batch_bconds, batch_grid, h=h)

        batch_operator = operator_prepare(operator, batch_dict, subset=subset, true_grid=prepared_grid[indices], h=h)
        
        
        loss+= point_sort_shift_loss(model, batch_grid, batch_operator, batch_bconds, lambda_bound=lambda_bound)
        batch_num+=1
    loss=1/batch_num*loss
    return loss


def compute_operator_loss(grid, model, operator, bconds, grid_point_subset=['central'], lambda_bound=10,h=0.001):
    prepared_grid = grid_prepare(grid)
    bconds = bnd_prepare(bconds, prepared_grid, h=h)
    operator = operator_prepare(operator, prepared_grid, subset=grid_point_subset, true_grid=grid, h=h)
    loss = point_sort_shift_loss(model, prepared_grid, operator, bconds, lambda_bound=lambda_bound)
    loss=float(loss.float())
    return loss


def solution_print(prepared_grid,model,title=None):
    if prepared_grid.shape[1] == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if title!=None:
            ax.set_title(title)
        ax.plot_trisurf(prepared_grid[:, 0].reshape(-1), prepared_grid[:, 1].reshape(-1),
                        model(prepared_grid).detach().numpy().reshape(-1), cmap=cm.jet, linewidth=0.2, alpha=1)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        plt.show()
    if prepared_grid.shape[1] == 1:
        fig = plt.figure()
        plt.scatter(prepared_grid.reshape(-1), model(prepared_grid).detach().numpy().reshape(-1))
        plt.show()





def point_sort_shift_solver(grid, model, operator, bconds, grid_point_subset=['central'], lambda_bound=10,
                            verbose=False, learning_rate=1e-4, eps=0.1, tmin=1000, tmax=1e5, h=0.001,
                            use_cache=True,cache_dir='../cache/',cache_verbose=False,batch_size=None,save_always=False):
    # prepare input data to uniform format 
    
    prepared_grid,grid_dict,point_type = grid_prepare(grid)
    prepared_bconds = bnd_prepare(bconds, prepared_grid, h=h)
    full_prepared_operator = operator_prepare(operator, grid_dict, subset=grid_point_subset, true_grid=grid, h=h)
    
    

    #  use cache if needed
    if use_cache:
        cache_checkpoint,min_loss=cache_lookup(prepared_grid, full_prepared_operator, prepared_bconds,cache_dir=cache_dir
                                               ,nmodels=None,verbose=cache_verbose,lambda_bound=0.001)
        model, optimizer_state= cache_retrain(model,cache_checkpoint,grid,verbose=cache_verbose)
    
        
    # model is not saved if cache model good enough

    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.LBFGS(model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # if optimizer_state is not None:
    #     try:
    #         optimizer.load_state_dict(optimizer_state)
    #     except Exception:
    #         optimizer_state=None
    #     tmin=100
    loss = point_sort_shift_loss(model, prepared_grid, full_prepared_operator, prepared_bconds, lambda_bound=lambda_bound)
    
    save_cache=False
    
    if loss>0.1 or save_always:
        save_cache=True
    
    
    # standard NN stuff
    if verbose:
        print('-1 {}'.format(loss))
    
    t = 0
    
    last_loss=np.zeros(100)+float(loss)
    line=np.polyfit(range(100),last_loss,1)
    
    # def closure():
    #     optimizer.zero_grad()
    #     loss = point_sort_shift_loss(model, prepared_grid, operator, bconds, lambda_bound=lambda_bound)
    #     loss.backward()
    #     return loss
    
    stop_dings=0
    
    # to stop train proceduce we fit the line in the loss data
    #if line is flat enough 5 times, we stop the procedure
    while stop_dings<=5:
        optimizer.zero_grad()
        if batch_size==None:
            loss = point_sort_shift_loss(model, prepared_grid, full_prepared_operator, prepared_bconds, lambda_bound=lambda_bound)
        else:
            loss=point_sort_shift_loss_batch(model, prepared_grid, point_type, operator, bconds,subset=grid_point_subset, lambda_bound=lambda_bound,batch_size=batch_size,h=h)
        last_loss[t%100]=loss.item()
        
        if t%100==0:
            line=np.polyfit(range(100),last_loss,1)
            if abs(line[0]) < eps:
                stop_dings+=1
        
        if (t % 100 == 0) and verbose:

            print(t, loss.item(), line,line[0]/line[1])
            solution_print(prepared_grid,model,title='Iteration = ' + str(t))

        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t += 1
        if t > tmax:
            break
    if save_cache:
        save_model(model,model.state_dict(),optimizer.state_dict(),cache_dir=cache_dir,name=None)
    return model


