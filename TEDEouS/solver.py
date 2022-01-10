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
from TEDEouS.input_preprocessing import grid_prepare, bnd_prepare, operator_prepare
from TEDEouS.metrics import point_sort_shift_loss,point_sort_shift_loss_batch
import numpy as np
from TEDEouS.cache import cache_lookup,cache_retrain,save_model





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
                            use_cache=True,cache_dir='../cache/',cache_verbose=False,
                            batch_size=None,save_always=False,lp_par=None):
    # prepare input data to uniform format 
    
    prepared_grid,grid_dict,point_type = grid_prepare(grid)
    prepared_bconds = bnd_prepare(bconds, prepared_grid,grid_dict, h=h)
    full_prepared_operator = operator_prepare(operator, grid_dict, subset=grid_point_subset, true_grid=grid, h=h)
    
    

    #  use cache if needed
    if use_cache:
        cache_checkpoint,min_loss=cache_lookup(prepared_grid, full_prepared_operator, prepared_bconds,cache_dir=cache_dir
                                               ,nmodels=None,verbose=cache_verbose,lambda_bound=0.001,norm=lp_par)
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
            loss = point_sort_shift_loss(model, prepared_grid, full_prepared_operator, prepared_bconds, lambda_bound=lambda_bound,norm=lp_par)
        else:
            loss=point_sort_shift_loss_batch(model, prepared_grid, point_type, operator, bconds,subset=grid_point_subset, lambda_bound=lambda_bound,batch_size=batch_size,h=h,norm=lp_par)
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


