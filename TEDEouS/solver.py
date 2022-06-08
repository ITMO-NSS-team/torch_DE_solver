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
from TEDEouS.input_preprocessing import bnd_prepare_matrix,operator_prepare_matrix
from TEDEouS.input_preprocessing import operator_prepare_autograd,bnd_prepare_autograd
from TEDEouS.input_preprocessing import op_dict_to_list
from TEDEouS.metrics import matrix_loss,autograd_loss
import numpy as np
from TEDEouS.cache import cache_lookup,cache_retrain,save_model,cache_lookup_autograd







def solution_print(prepared_grid,model,title=None):
    if prepared_grid.shape[1] == 2:
        nvars_model = model(prepared_grid).shape[1]
        plt.ion()
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(projection='3d')

        if title!=None:
            ax1.set_title(title)
            # ax2.set_title(title)
        if nvars_model == 1:
            ax1.plot_trisurf(prepared_grid[:, 0].detach().numpy().reshape(-1), prepared_grid[:, 1].detach().numpy().reshape(-1),
                        model(prepared_grid).detach().numpy().reshape(-1), cmap=cm.jet, linewidth=0.2, alpha=1)
            ax1.set_xlabel("x1")
            ax1.set_ylabel("x2")

        elif nvars_model == 2:
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(projection='3d')
            ax1.plot_trisurf(prepared_grid[:, 0].reshape(-1), prepared_grid[:, 1].reshape(-1),
                        model(prepared_grid)[:,0].detach().numpy().reshape(-1), cmap=cm.jet, linewidth=0.2, alpha=1)
            ax2.plot_trisurf(prepared_grid[:, 0].reshape(-1), prepared_grid[:, 1].reshape(-1),
                        model(prepared_grid)[:,1].detach().numpy().reshape(-1), cmap=cm.jet, linewidth=0.2, alpha=1)
            ax2.set_xlabel("x1")
            ax2.set_ylabel("x2")
        
        
        #plt.show(block=False)
        #plt.show()
        plt.pause(0.001)

    if prepared_grid.shape[1] == 1:
        fig = plt.figure()
        plt.scatter(prepared_grid.detach().numpy().reshape(-1), model(prepared_grid).detach().numpy().reshape(-1))
        plt.show(block=False)
        plt.show()


def solution_print_mat(grid,model,title=None):
    if grid.shape[0] == 1:
        fig = plt.figure()
        plt.scatter(grid.reshape(-1), model.detach().numpy().reshape(-1))
        plt.show()
    if grid.shape[0] == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if title!=None:
            ax.set_title(title)
        ax.plot_trisurf(grid[0].reshape(-1), grid[1].reshape(-1),
                        model.reshape(-1).detach().numpy(), cmap=cm.jet, linewidth=0.2, alpha=1)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        plt.show()



def create_random_fn(eps):
    def randomize_params(m):
      if type(m)==torch.nn.Linear or type(m)==torch.nn.Conv2d:
        m.weight.data=m.weight.data+(2*torch.randn(m.weight.size())-1)*eps#Random weight initialisation
        m.bias.data=m.bias.data+(2*torch.randn(m.bias.size())-1)*eps
    return randomize_params




def point_sort_shift_solver(grid, model, operator, bconds, grid_point_subset=['central'], lambda_bound=10,
                            verbose=False, learning_rate=1e-4, eps=1e-5, tmin=1000, tmax=1e5, h=0.001,
                            use_cache=True,cache_dir='../cache/',cache_verbose=False,
                            batch_size=None,save_always=False,lp_par=None,print_every=100,
                            patience=5,loss_oscillation_window=100,no_improvement_patience=1000,
                            model_randomize_parameter=0,optimizer='Adam',print_plot = True):
    # prepare input data to uniform format 
    
    prepared_grid,grid_dict,point_type = grid_prepare(grid)
    prepared_bconds = bnd_prepare(bconds, prepared_grid,grid_dict, h=h)
    full_prepared_operator = operator_prepare(operator, grid_dict, subset=grid_point_subset, true_grid=grid, h=h)
    
    r=create_random_fn(model_randomize_parameter)   
    #  use cache if needed
    if use_cache:
        cache_checkpoint,min_loss=cache_lookup(prepared_grid, full_prepared_operator, prepared_bconds,cache_dir=cache_dir
                                               ,nmodels=None,verbose=cache_verbose,lambda_bound=lambda_bound,norm=lp_par)
        model, optimizer_state= cache_retrain(model,cache_checkpoint,grid,verbose=cache_verbose)
  
        model.apply(r)


    if optimizer=='Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer=='SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer=='LBFGS':
        optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate)
    else:
        print('Wrong optimizer chosen, optimization was not performed')
        return model

    if not use_cache:
        min_loss = point_sort_shift_loss(model, prepared_grid, full_prepared_operator, prepared_bconds, lambda_bound=lambda_bound)
    
    save_cache=False
    
    if min_loss>0.1 or save_always:
        save_cache=True
    
    
    # standard NN stuff
    if verbose:
        print('-1 {}'.format(min_loss))
    
    t = 0
    
    last_loss=np.zeros(loss_oscillation_window)+float(min_loss)
    line=np.polyfit(range(loss_oscillation_window),last_loss,1)
    

    def closure():
        nonlocal cur_loss
        optimizer.zero_grad()
        if batch_size==None:
            loss = point_sort_shift_loss(model, prepared_grid, full_prepared_operator, prepared_bconds, lambda_bound=lambda_bound,norm=lp_par)
        else:
            loss=point_sort_shift_loss_batch(model, prepared_grid, point_type, operator, bconds,subset=grid_point_subset, lambda_bound=lambda_bound,batch_size=batch_size,h=h,norm=lp_par)
        loss.backward()
        cur_loss = loss.item()
        return loss
    
    stop_dings=0
    t_imp_start=0
    # to stop train proceduce we fit the line in the loss data
    #if line is flat enough 5 times, we stop the procedure
    cur_loss=min_loss
    while stop_dings<=patience:
        optimizer.step(closure)

        last_loss[t%loss_oscillation_window]=cur_loss

        
        if cur_loss<min_loss:
            min_loss=cur_loss
            t_imp_start=t
        if t%loss_oscillation_window==0:
            line=np.polyfit(range(loss_oscillation_window),last_loss,1)
            if abs(line[0]/cur_loss) < eps and t>0:
                stop_dings+=1
                model.apply(r)
                if verbose:
                    print('Oscillation near the same loss')
                    print(t, cur_loss, line,line[0]/cur_loss, stop_dings)
                    solution_print(prepared_grid,model,title='Iteration = ' + str(t))
        

        if (t-t_imp_start==no_improvement_patience) and verbose:
            print('No improvement in '+str(no_improvement_patience)+' steps')
            t_imp_start=t
            stop_dings+=1
            print(t, cur_loss, line,line[0]/cur_loss, stop_dings)
            solution_print(prepared_grid,model,title='Iteration = ' + str(t))
            
        if print_every!=None and (t % print_every == 0) and verbose:
            print(t, cur_loss, line,line[0]/cur_loss, stop_dings)
            solution_print(prepared_grid,model,title='Iteration = ' + str(t))

        t += 1
        if t > tmax:
            break
    if (save_cache and use_cache) or save_always:
        save_model(model,model.state_dict(),optimizer.state_dict(),cache_dir=cache_dir,name=None)
    return model


def nn_optimizer(grid, model, operator, bconds, grid_point_subset=['central'], lambda_bound=10,
                            verbose=False, learning_rate=1e-4, eps=1e-5, tmin=1000, tmax=1e5, h=0.001,
                            use_cache=True,cache_dir='../cache/',cache_verbose=False,
                            batch_size=None,save_always=False,lp_par=None,print_every=100,
                            patience=5,loss_oscillation_window=100,no_improvement_patience=1000,
                            model_randomize_parameter=0,optimizer='Adam'):
    # prepare input data to uniform format 
    
    prepared_grid,grid_dict,point_type = grid_prepare(grid)
    prepared_bconds = bnd_prepare(bconds, prepared_grid,grid_dict, h=h)
    full_prepared_operator = operator_prepare(operator, grid_dict, subset=grid_point_subset, true_grid=grid, h=h)
    
    r=create_random_fn(model_randomize_parameter)   
    #  use cache if needed
    if use_cache:
        cache_checkpoint,min_loss=cache_lookup(prepared_grid, full_prepared_operator, prepared_bconds,cache_dir=cache_dir
                                               ,nmodels=None,verbose=cache_verbose,lambda_bound=lambda_bound,norm=lp_par)
        model, optimizer_state= cache_retrain(model,cache_checkpoint,grid,verbose=cache_verbose)
  
        model.apply(r)

    if optimizer=='Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer=='SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer=='LBFGS':
        optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate)
    else:
        print('Wrong optimizer chosen, optimization was not performed')
        return model
    

    if not use_cache:
        min_loss = point_sort_shift_loss(model, prepared_grid, full_prepared_operator, prepared_bconds, lambda_bound=lambda_bound)
    
    save_cache=False
    
    if min_loss>0.1 or save_always:
        save_cache=True
    
    
    # standard NN stuff
    if verbose:
        print('-1 {}'.format(min_loss))
    
    t = 0
    
    last_loss=np.zeros(loss_oscillation_window)+float(min_loss)
    line=np.polyfit(range(loss_oscillation_window),last_loss,1)
    

    def closure():
        nonlocal cur_loss
        optimizer.zero_grad()
        if batch_size==None:
            loss = point_sort_shift_loss(model, prepared_grid, full_prepared_operator, prepared_bconds, lambda_bound=lambda_bound,norm=lp_par)
        else:
            loss=point_sort_shift_loss_batch(model, prepared_grid, point_type, operator, bconds,subset=grid_point_subset, lambda_bound=lambda_bound,batch_size=batch_size,h=h,norm=lp_par)
        loss.backward()
        cur_loss = loss.item()
        return loss
    
    stop_dings=0
    t_imp_start=0
    # to stop train proceduce we fit the line in the loss data
    #if line is flat enough 5 times, we stop the procedure
    cur_loss=min_loss
    while stop_dings<=patience:
        optimizer.step(closure)

        last_loss[t%loss_oscillation_window]=cur_loss
        
        if cur_loss<min_loss:
            min_loss=cur_loss
            t_imp_start=t
        if t%loss_oscillation_window==0:
            line=np.polyfit(range(loss_oscillation_window),last_loss,1)

            if abs(line[0]/cur_loss) < eps and t>0:
                stop_dings+=1
                model.apply(r)
                if verbose:
                    print('Oscillation near the same loss')
                    print(t, cur_loss, line,line[0]/cur_loss, stop_dings)
                    if print_plot:
                        solution_print(prepared_grid,model,title='Iteration = ' + str(t))

        if (t-t_imp_start==no_improvement_patience) and verbose:
            print('No improvement in '+str(no_improvement_patience)+' steps')
            t_imp_start=t
            stop_dings+=1
            print(t, cur_loss, line,line[0]/cur_loss, stop_dings)
            if print_plot:
                solution_print(prepared_grid,model,title='Iteration = ' + str(t))
            
        if print_every!=None and (t % print_every == 0) and verbose:
            print(t, cur_loss, line,line[0]/cur_loss, stop_dings)
            if print_plot:
                solution_print(prepared_grid,model,title='Iteration = ' + str(t))


        t += 1
        if t > tmax:
            break
    if (save_cache and use_cache) or save_always:
        save_model(model,model.state_dict(),optimizer.state_dict(),cache_dir=cache_dir,name=None)
    return model





def lbfgs_solution(model, grid, operator, norm_lambda, bcond, rtol=1e-6,atol=0.01,nsteps=10000):
    
    unified_operator = operator_prepare_matrix(operator)

    b_prepared = bnd_prepare_matrix(bcond, grid)

    # optimizer = torch.optim.Adam([model.requires_grad_()], lr=1e-4)
    optimizer = torch.optim.LBFGS([model.requires_grad_()], lr=1e-3)
   
    
    def closure():
        nonlocal cur_loss
        optimizer.zero_grad()

        loss = matrix_loss(model, grid, unified_operator, b_prepared, lambda_bound=norm_lambda)
        loss.backward()
        cur_loss = loss.item()

        return loss

    cur_loss = float('inf')
    # tol = 1e-20

    for i in range(nsteps):
        optimizer.zero_grad()
        past_loss = cur_loss
        
        # loss = matrix_loss(model, grid, unified_operator, b_prepared, lambda_bound=norm_lambda)
        optimizer.step(closure)
        
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # cur_loss = loss.item()
        
        
        if i % 1000 == 0:
            print('i={} loss={}'.format(i, cur_loss))

        if abs(cur_loss - past_loss) / abs(cur_loss) < rtol:
        #     # print("sosholsya")
            break
        if abs(cur_loss) < atol:
        #     # print("sosholsya")
            break

    return model.detach()



def matrix_cache_lookup(grid, model, operator, bconds,h=0.001,model_randomize_parameter=0,cache_dir="../cache",
                            lambda_bound=10):
    # prepare input data to uniform format 
    op_list=op_dict_to_list(operator)

    for term in op_list:
        if type(term[0])==torch.Tensor:
            term[0]=term[0].reshape(-1)
        if callable(term[0]):
            print("Warning: coefficient is callable, it may lead to wrong cache item choice")

    prepared_grid,grid_dict,point_type = grid_prepare(grid)
    prepared_bconds = bnd_prepare(bconds, prepared_grid,grid_dict, h=h)
    full_prepared_operator = operator_prepare(op_list, grid_dict, true_grid=grid, h=h)
    
    r=create_random_fn(model_randomize_parameter)   
    #  use cache if needed

    cache_checkpoint,min_loss=cache_lookup(prepared_grid, full_prepared_operator, prepared_bconds,cache_dir=cache_dir
                                           ,nmodels=None,verbose=True,lambda_bound=lambda_bound)
    model, optimizer_state= cache_retrain(model,cache_checkpoint,grid,verbose=False)
  
    model.apply(r)
    
    return model



def matrix_optimizer(grid, model, operator, bconds, lambda_bound=10,
                            verbose=False, learning_rate=1e-4, eps=1e-5, tmin=1000, tmax=1e5,
                            use_cache=True,cache_dir='../cache/',cache_verbose=False,
                            batch_size=None,save_always=False,lp_par=None,print_every=100,
                            patience=5,loss_oscillation_window=100,no_improvement_patience=1000,
                            model_randomize_parameter=1e-5,optimizer='LBFGS',cache_model=None, coord_list=None):
    # prepare input data to uniform format 
    
    #if coord_list==None:
     #   print("Error: No coord_list supplied, using cache is not possible")
      #  use_cache=False

    if model==None:
        if len(grid.shape)==2:
            model=torch.rand(grid.shape)
        else:
            model= torch.rand(grid[0].shape)
   


    if use_cache:
        NN_grid=torch.from_numpy(np.vstack([grid[i].reshape(-1) for i in range(grid.shape[0])]).T).float()
        if cache_model==None:
            cache_model = torch.nn.Sequential(
                torch.nn.Linear(grid.shape[0], 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 1)
            )
        
        cache_model=matrix_cache_lookup(NN_grid, cache_model, operator, bconds,cache_dir=cache_dir,lambda_bound=100)
        r=create_random_fn(model_randomize_parameter)
        
        cache_model.apply(r)
        
        if len(grid.shape)==2:
            model=cache_model(NN_grid).reshape(grid.shape).detach()
        else:
            model=cache_model(NN_grid).reshape(grid[0].shape).detach()
        
    
    unified_operator = operator_prepare_matrix(operator)

    b_prepared = bnd_prepare_matrix(bconds, grid)
    
    if optimizer=='Adam':
        optimizer = torch.optim.Adam([model.requires_grad_()], lr=learning_rate)
    elif optimizer=='SGD':
        optimizer = torch.optim.SGD([model.requires_grad_()], lr=learning_rate)
    elif optimizer=='LBFGS':
        optimizer = torch.optim.LBFGS([model.requires_grad_()], lr=learning_rate)
    else:
        print('Wrong optimizer chosen, optimization was not performed')
        return model
    
    min_loss = matrix_loss(model, grid, unified_operator, b_prepared, lambda_bound=lambda_bound)
    
    # standard NN stuff
    if verbose:
        print('-1 {}'.format(min_loss))
    
    t = 0
    
    last_loss=np.zeros(loss_oscillation_window)+float(min_loss)
    line=np.polyfit(range(loss_oscillation_window),last_loss,1)
    

    def closure():
        nonlocal cur_loss
        optimizer.zero_grad()
        loss =matrix_loss(model, grid, unified_operator, b_prepared, lambda_bound=lambda_bound)
        loss.backward()
        cur_loss = loss.item()
        return loss
    
    stop_dings=0
    t_imp_start=0
    # to stop train proceduce we fit the line in the loss data
    #if line is flat enough 5 times, we stop the procedure
    cur_loss=min_loss
    while stop_dings<=patience:
        optimizer.step(closure)

        last_loss[t%loss_oscillation_window]=cur_loss
        
        if cur_loss<min_loss:
            min_loss=cur_loss
            t_imp_start=t
        if t%loss_oscillation_window==0:
            line=np.polyfit(range(loss_oscillation_window),last_loss,1)
            if abs(line[0]/cur_loss) < eps and t>0:
                stop_dings+=1
                if verbose:
                    print('Oscillation near the same loss')
                    print(t, cur_loss, line,line[0]/cur_loss, stop_dings)
                    solution_print_mat(grid,model,title='Iteration = ' + str(t))
        
        if (t-t_imp_start==no_improvement_patience) and verbose:
            print('No improvement in '+str(no_improvement_patience)+' steps, min_loss = '+str(min_loss))
            t_imp_start=t
            stop_dings+=1
            print(t, cur_loss, line,line[0]/cur_loss, stop_dings)
            solution_print_mat(grid,model,title='Iteration = ' + str(t))
            
        if print_every!=None and (t % print_every == 0) and verbose:
            print(t, cur_loss, line,line[0]/cur_loss, stop_dings)
            solution_print_mat(grid,model,title='Iteration = ' + str(t))

        t += 1
        if t > tmax:
            break

        
    if use_cache or save_always:
        if cache_model==None:
            cache_model = torch.nn.Sequential(
                torch.nn.Linear(grid.shape[0], 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 1)
            )
        
        NN_grid=torch.from_numpy(np.vstack([grid[i].reshape(-1) for i in range(grid.shape[0])]).T).float()
        optimizer = torch.optim.Adam(cache_model.parameters(), lr=0.001)
        
        model_res=model.reshape(-1,1)
            
        def closure():
            optimizer.zero_grad()
            loss = torch.mean((cache_model(NN_grid)-model_res)**2)
            loss.backward()
            return loss
        
        loss=np.inf
        t=1
        while loss>1e-5 and t<1e5:
            loss = optimizer.step(closure)
            t+=1
            if False:
                print('Retrain from cache t={}, loss={}'.format(t,loss))


        save_model(cache_model,cache_model.state_dict(),optimizer.state_dict(),cache_dir=cache_dir,name=None)
    
        
    return model



def nn_autograd_optimizer(grid, model, operator, bconds, lambda_bound=10,
                            verbose=False, learning_rate=1e-4, eps=1e-5, tmin=1000, tmax=1e5, h=0.001,
                            use_cache=True,cache_dir='../cache/',cache_verbose=False,
                            batch_size=None,save_always=False,lp_par=None,print_every=100,
                            patience=5,loss_oscillation_window=100,no_improvement_patience=1000,
                            model_randomize_parameter=0,optimizer='Adam',abs_loss=None):
    # prepare input data to uniform format 
    
    grid=torch.clone(grid)
    
    full_prepared_operator = operator_prepare_autograd(operator,grid)
    
    prepared_bconds=bnd_prepare_autograd(bconds, grid)
    
    r=create_random_fn(model_randomize_parameter)   
     # use cache if needed
    if use_cache:
        cache_checkpoint,min_loss=cache_lookup_autograd(grid, full_prepared_operator, prepared_bconds,cache_dir=cache_dir
                                                ,nmodels=None,verbose=cache_verbose,lambda_bound=lambda_bound)
        model, optimizer_state= cache_retrain(model,cache_checkpoint,grid,verbose=cache_verbose)
  
        model.apply(r)

    if optimizer=='Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer=='SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer=='LBFGS':
        optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate)
    else:
        print('Wrong optimizer chosen, optimization was not performed')
        return model
    

    if not use_cache:
        min_loss = autograd_loss(model, grid, full_prepared_operator, prepared_bconds, lambda_bound=lambda_bound)
    
    save_cache=False
    
    if min_loss>0.1 or save_always:
        save_cache=True
    
    
    # standard NN stuff
    if verbose:
        print('-1 {}'.format(min_loss))
    
    t = 0
    
    last_loss=np.zeros(loss_oscillation_window)+float(min_loss)
    line=np.polyfit(range(loss_oscillation_window),last_loss,1)
    


    def closure():
        nonlocal cur_loss
        optimizer.zero_grad()
        loss =autograd_loss(model, grid, full_prepared_operator, prepared_bconds, lambda_bound=lambda_bound)
        loss.backward()
        cur_loss = loss.item()
        return loss
    
    stop_dings=0
    t_imp_start=0
    # to stop train proceduce we fit the line in the loss data
    #if line is flat enough 5 times, we stop the procedure
    cur_loss=min_loss
    while stop_dings<=patience:
        optimizer.step(closure)

        last_loss[t%loss_oscillation_window]=cur_loss
        
        if cur_loss<min_loss:
            min_loss=cur_loss
            t_imp_start=t
        if t%loss_oscillation_window==0:
            line=np.polyfit(range(loss_oscillation_window),last_loss,1)
            if abs(line[0]/cur_loss) < eps and t>0:
                stop_dings+=1
                model.apply(r)
                if verbose:
                    print('Oscillation near the same loss')
                    print(t, cur_loss, line,line[0]/cur_loss, stop_dings)
                    solution_print(grid,model,title='Iteration = ' + str(t))
        
        if (t-t_imp_start==no_improvement_patience) and verbose:
            print('No improvement in '+str(no_improvement_patience)+' steps')
            t_imp_start=t
            stop_dings+=1
            print(t, cur_loss, line,line[0]/cur_loss, stop_dings)
            solution_print(grid,model,title='Iteration = ' + str(t))
        
        if abs_loss!=None:
            if cur_loss<abs_loss and verbose:
                print('Absolute value of loss is lower than threshold')
                stop_dings+=1
                print(t, cur_loss, line,line[0]/cur_loss, stop_dings)
                solution_print(grid,model,title='Iteration = ' + str(t))  
     
         
            
        if print_every!=None and (t % print_every == 0) and verbose:
            print(t, cur_loss, line,line[0]/cur_loss, stop_dings)
            solution_print(grid,model,title='Iteration = ' + str(t))

        t += 1
        if t > tmax:
            break
    if (save_cache and use_cache) or save_always:
        save_model(model,model.state_dict(),optimizer.state_dict(),cache_dir=cache_dir,name=None)
    return model



def grid_format_prepare(coord_list, mode='NN'):
    if type(coord_list)==torch.Tensor:
        print('Grid is a tensor, assuming old format, no action performed')
        return coord_list
    if mode=='NN':
        # coord_list=torch.tensor(coord_list)
        grid=torch.cartesian_prod(*coord_list).float()
        # grid=grid.reshape(-1,1)
    elif mode=='mat':
        grid = np.meshgrid(*coord_list)
        grid = torch.tensor(grid)
    return grid






def optimization_solver(coord_list, model, operator,bconds,config,mode='NN'):
    grid=grid_format_prepare(coord_list, mode=mode)
    if mode=='NN':
        model=point_sort_shift_solver(grid, model, operator, bconds,
            learning_rate=config.params['Optimizer']['learning_rate'],
            lambda_bound=config.params['Optimizer']['lambda_bound'],
            optimizer=config.params['Optimizer']['optimizer'],
            grid_point_subset=config.params['NN']['grid_point_subset'],
            h=config.params['NN']['h'],
            use_cache=config.params['Cache']['use_cache'],
            cache_dir=config.params['Cache']['cache_dir'],
            cache_verbose=config.params['Cache']['cache_verbose'],
            model_randomize_parameter=config.params['Cache']['model_randomize_parameter'],
            save_always=config.params['Cache']['save_always'],
            batch_size=config.params['NN']['batch_size'],
            lp_par=config.params['NN']['lp_par'],
            verbose=config.params['Verbose']['verbose'],
            print_every=config.params['Verbose']['print_every'],
            eps=config.params['StopCriterion']['eps'], 
            tmin=config.params['StopCriterion']['tmin'], 
            tmax=config.params['StopCriterion']['tmax'], 
            patience=config.params['StopCriterion']['patience'],
            loss_oscillation_window=config.params['StopCriterion']['loss_oscillation_window'],
            no_improvement_patience=config.params['StopCriterion']['no_improvement_patience'])
    if mode=='mat':
        model=matrix_optimizer(grid, model, operator, bconds,
            learning_rate=config.params['Optimizer']['learning_rate'],
            lambda_bound=config.params['Optimizer']['lambda_bound'],
            optimizer=config.params['Optimizer']['optimizer'],
            use_cache=config.params['Cache']['use_cache'],
            cache_dir=config.params['Cache']['cache_dir'],
            cache_verbose=config.params['Cache']['cache_verbose'],
            model_randomize_parameter=config.params['Cache']['model_randomize_parameter'],
            save_always=config.params['Cache']['save_always'],
            lp_par=config.params['Matrix']['lp_par'],
            cache_model=config.params['Matrix']['cache_model'],
            verbose=config.params['Verbose']['verbose'],
            print_every=config.params['Verbose']['print_every'],
            eps=config.params['StopCriterion']['eps'], 
            tmin=config.params['StopCriterion']['tmin'], 
            tmax=config.params['StopCriterion']['tmax'], 
            patience=config.params['StopCriterion']['patience'],
            loss_oscillation_window=config.params['StopCriterion']['loss_oscillation_window'],
            no_improvement_patience=config.params['StopCriterion']['no_improvement_patience'])             
    return model
