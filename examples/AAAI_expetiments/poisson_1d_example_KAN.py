
import torch
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('AAAI_expetiments'))))


from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.models import mat_model, Fourier_embedding
from tedeous.callbacks import plot, early_stopping, adaptive_lambda
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.eval import integration

from tedeous.models import KAN

import pandas as pd

solver_device('cuda')

a = 4

def u(x):
  return torch.sin(torch.pi * a * x)

def u_x(x):
   return (torch.pi * a) * torch.cos(torch.pi * a * x)

def u_xx(x):
  return -(torch.pi * a) ** 2 * torch.sin(torch.pi * a * x)

def u_net(net, x):
    net = net.to('cpu')
    x = x.to('cpu')
    return net(x).detach()

def u_net_x(net, x):
    x = x.to('cpu')
    net = net.to('cpu')
    x.requires_grad_()
    u = net(x)
    u_x = torch.autograd.grad(sum(u), x)[0]
    return u_x.detach()

def u_net_xx(net, x):
    x = x.to('cpu')
    net = net.to('cpu')
    x.requires_grad_()
    u = net(x)
    u_x = torch.autograd.grad(sum(u), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(sum(u_x), x)[0]
    return u_xx.detach()

def c2_norm(net, x):
    norms = [(u, u_net), (u_x, u_net_x), (u_xx, u_net_xx)]
    norm = 0
    for exact, predict in norms:
        norm += torch.max(abs(exact(x).cpu().reshape(-1) - predict(net, x).cpu().reshape(-1)))
    return norm.detach().cpu().numpy()

def l2_norm(net, x):
    x = x.to('cpu')
    net = net.to('cpu')
    predict = net(x).detach().cpu().reshape(-1)
    exact = u(x).detach().cpu().reshape(-1)
    l2_norm = torch.sqrt(sum((predict-exact)**2))
    return l2_norm.detach().cpu().numpy()

def l2_norm_mat(net, x):
    x = x.to('cpu')
    net = net.to('cpu')
    predict = net.detach().cpu().reshape(-1)
    exact = u(x).detach().cpu().reshape(-1)
    l2_norm = torch.sqrt(sum((predict-exact)**2))
    return l2_norm.detach().cpu().numpy()

def l2_norm_fourier(net, x):
    x = x.to(torch.device('cuda:0'))
    predict = net(x).detach().cpu().reshape(-1)
    exact = u(x).detach().cpu().reshape(-1)
    l2_norm = torch.sqrt(sum((predict-exact)**2))
    return l2_norm.detach().cpu().numpy()




def poisson1d_problem_formulation(grid_res):
    x0 = 0
    xmax = 1

    domain = Domain()

    domain.variable('x', [x0, xmax], grid_res, dtype='float32')

    boundaries = Conditions()

    boundaries.dirichlet({'x': x0}, value=u)
    boundaries.dirichlet({'x': xmax}, value=u)

    grid = domain.variable_dict['x'].reshape(-1,1)

    # equation: d2u/dx2 = -16*pi^2*sin(4*pi*x)

    equation = Equation()

    poisson = {
        '-d2u/dx2':
            {
            'coeff': -1,
            'term': [0, 0],
            'pow': 1,
            },

        'f(x)':
            {
            'coeff': u_xx(grid),
            'term': [None],
            'pow': 0,
            }
    }

    equation.add(poisson)
    return grid,domain,equation,boundaries




def experiment_data_amount_poisson_1d_PINN(grid_res,exp_name='poisson1d_PINN',save_plot=True):
    solver_device('cuda')
    exp_dict_list = []

    grid,domain,equation,boundaries = poisson1d_problem_formulation(grid_res)


    net = KAN(layers_hidden=[1,50,50,1])

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=1, lambda_bound=1)
    
    cb_es = early_stopping.EarlyStopping(eps=1e-9, randomize_parameter=1e-6)

    optimizer = Optimizer('Adam', {'lr': 1e-4})
     
    start=time.time()

    model.train(optimizer, 5e5, save_model=False, callbacks=[cb_es])

    end=time.time()

    run_time=end-start

    end_loss = model.solution_cls.evaluate()[0].detach().cpu().numpy()

    lu_f = model.solution_cls.operator.operator_compute()

    lu_f, _ = integration(lu_f, grid)

    lu = u_net_xx(net, grid)

    lu, _ = integration(lu, grid)

    grid_test = torch.linspace(0, 1, 100).reshape(-1, 1).float()

    exp_dict={'grid_res': grid_res,
                        'l2_error_train': l2_norm(net, grid),
                        'l2_error_test': l2_norm(net, grid_test),
                        'c2_error_train': c2_norm(net, grid),
                        'c2_error_test': c2_norm(net, grid_test),
                        'loss': end_loss,
                        "lu_f": lu_f.item(),
                        "lu": lu.item(),
                        'run_time': run_time,
                        'type':exp_name}

    exp_dict_list.append(exp_dict)


    print('grid_res=', grid_res)
    print('c2_norm = ', c2_norm(net, grid))
    print('l2_norm = ', l2_norm(net, grid))
    print('lu_f = ', lu_f.item())
    print('lu = ', lu.item())

    if save_plot:
        if not os.path.isdir('examples\\AAAI_expetiments\\figures_{}'.format(exp_name)):
            os.mkdir('examples\\AAAI_expetiments\\figures_{}'.format(exp_name))
        plt.figure()
        plt.plot(grid.detach().cpu().numpy(), u(grid).detach().cpu().numpy(), label='Exact')
        plt.plot(grid.detach().cpu().numpy(), net(grid.cpu()).detach().cpu().numpy(), '--', label='Predicted')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='upper right')
        plt.savefig('examples\\AAAI_expetiments\\figures_{}\\img_{}.png'.format(exp_name,N))

    return exp_dict_list


def experiment_data_amount_poisson_1d_PSO_KAN(grid_res,exp_name='poisson1d_PSO_KAN',save_plot=True):
    solver_device('cuda')
    exp_dict_list = []

    grid,domain,equation,boundaries = poisson1d_problem_formulation(grid_res)

    net = KAN(layers_hidden=[1,50,50,1])

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=1, lambda_bound=1)
    
    cb_es = early_stopping.EarlyStopping(eps=1e-9, randomize_parameter=1e-6)

    optimizer = Optimizer('Adam', {'lr': 1e-4})

    start=time.time()

    model.train(optimizer, 5e5, save_model=False, callbacks=[cb_es])

    l2_pinn = l2_norm(net, grid)
    print('l2_norm_before_PSO = ', l2_pinn)

    grid_test = torch.linspace(0, 1, 100).reshape(-1, 1).float()

    l2_error_adam_train=l2_norm(net, grid),
    l2_error_adam_test=l2_norm(net, grid_test),
    c2_error_adam_train=c2_norm(net, grid),
    l2_error_adam_test=c2_norm(net, grid_test),


    net = net.to('cuda')

    ########

    cb_es = early_stopping.EarlyStopping(eps=1e-9, randomize_parameter=1e-6, info_string_every=1000)

    optimizer = Optimizer('PSO', {'pop_size': 50, #30
                                  'b': 0.4, #0.5
                                  'c2': 0.5, #0.05
                                  'c1': 0.5, 
                                  'variance': 5e-2,
                                  'lr': 1e-4})

    model.train(optimizer, 2e4, save_model=False, callbacks=[cb_es])


    end=time.time()

    run_time=end-start

    end_loss = model.solution_cls.evaluate()[0].detach().cpu().numpy()

    lu_f = model.solution_cls.operator.operator_compute()

    lu_f, _ = integration(lu_f, grid)

    lu = u_net_xx(net, grid)

    lu, _ = integration(lu, grid)

    grid_test = torch.linspace(0, 1, 100).reshape(-1, 1).float()

    exp_dict={'grid_res': grid_res,
                        'l2_error_train_adam': l2_error_adam_train,
                        'l2_error_test_adam': l2_error_adam_test,
                        'c2_error_train_adam': c2_error_adam_train,
                        'c2_error_test_adam': l2_error_adam_test,
                        'l2_error_train_PSO': l2_norm(net, grid),
                        'l2_error_test_PSO': l2_norm(net, grid_test),
                        'c2_error_train_PSO': c2_norm(net, grid),
                        'c2_error_test_PSO': c2_norm(net, grid_test),
                        'loss': end_loss,
                        "lu_f": lu_f.item(),
                        "lu": lu.item(),
                        'run_time': run_time,
                        'type':exp_name}

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE_adam {}= {}'.format(grid_res, l2_error_adam_test))
    print('RMSE_pso {}= {}'.format(grid_res, l2_norm(net, grid_test)))

    exp_dict_list.append(exp_dict)


    if save_plot:
        if not os.path.isdir('examples\\AAAI_expetiments\\figures_{}'.format(exp_name)):
            os.mkdir('examples\\AAAI_expetiments\\figures_{}'.format(exp_name))
        plt.figure()
        plt.plot(grid.detach().cpu().numpy(), u(grid).detach().cpu().numpy(), label='Exact')
        plt.plot(grid.detach().cpu().numpy(), net(grid.cpu()).detach().cpu().numpy(), '--', label='Predicted')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='upper right')
        plt.savefig('examples\\AAAI_expetiments\\figures_{}\\img_{}.png'.format(exp_name,N))

    return exp_dict_list




def experiment_data_amount_poisson_1d_lam_KAN(grid_res,exp_name='poisson1d_lam_KAN',save_plot=True):
    solver_device('cuda')
    exp_dict_list = []

    grid,domain,equation,boundaries = poisson1d_problem_formulation(grid_res)

    net = KAN(layers_hidden=[1,50,50,1])

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=1, lambda_bound=1)
    
    cb_es = early_stopping.EarlyStopping(eps=1e-9, randomize_parameter=1e-6, normalized_loss=True)
    cb_lam = adaptive_lambda.AdaptiveLambda()

    optimizer = Optimizer('Adam', {'lr': 1e-4})

    start=time.time()

    model.train(optimizer, 5e5, save_model=False, callbacks=[cb_es, cb_lam])

    end=time.time()

    run_time=end-start

    end_loss = model.solution_cls.evaluate()[0].detach().cpu().numpy()

    lu_f = model.solution_cls.operator.operator_compute()

    lu_f, _ = integration(lu_f, grid)

    lu = u_net_xx(net, grid)

    lu, _ = integration(lu, grid)

    grid_test = torch.linspace(0, 1, 100).reshape(-1, 1).float()

    exp_dict={'grid_res': grid_res,
                        'l2_error_train': l2_norm(net, grid),
                        'l2_error_test': l2_norm(net, grid_test),
                        'c2_error_train': c2_norm(net, grid),
                        'c2_error_test': c2_norm(net, grid_test),
                        'loss': end_loss,
                        "lu_f": lu_f.item(),
                        "lu": lu.item(),
                        'run_time': run_time,
                        'type':exp_name}

    exp_dict_list.append(exp_dict)

    print('grid_res=', grid_res)
    print('c2_norm = ', c2_norm(net, grid))
    print('l2_norm = ', l2_norm(net, grid))
    print('lu_f = ', lu_f.item())
    print('lu = ', lu.item())

    if save_plot:
        if not os.path.isdir('examples\\AAAI_expetiments\\figures_{}'.format(exp_name)):
            os.mkdir('examples\\AAAI_expetiments\\figures_{}'.format(exp_name))
        plt.figure()
        plt.plot(grid.detach().cpu().numpy(), u(grid).detach().cpu().numpy(), label='Exact')
        plt.plot(grid.detach().cpu().numpy(), net(grid.cpu()).detach().cpu().numpy(), '--', label='Predicted')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='upper right')
        plt.savefig('examples\\AAAI_expetiments\\figures_{}\\img_{}.png'.format(exp_name,N))

    return exp_dict_list





def replace_none_with_zero(tuple_data):
    if isinstance(tuple_data, torch.Tensor):
        tuple_data[tuple_data == None] = 0
    elif tuple_data is None:
        tuple_data = torch.tensor([0.])
    elif isinstance(tuple_data, tuple):
        new_tuple = tuple(replace_none_with_zero(item) for item in tuple_data)
        return new_tuple
    return tuple_data

def gramian(net, residuals):
        # Compute the jacobian on batched data
    def jacobian():
        jac = []
        loss = residuals
        for l in loss:
            j = torch.autograd.grad(l, net.parameters(), retain_graph=True, allow_unused=True)
            j = replace_none_with_zero(j)
            j = parameters_to_vector(j).reshape(1, -1)
            jac.append(j)
        return torch.cat(jac)

    J = jacobian()
    return 1.0 / len(residuals) * J.T @ J

def grid_line_search_factory(loss, steps):

    def loss_at_step(step, model, tangent_params):
        params = parameters_to_vector(model.parameters())
        new_params = params - step*tangent_params
        vector_to_parameters(new_params, model.parameters())
        loss_val, _ = loss()
        vector_to_parameters(params, model.parameters())
        return loss_val

    def grid_line_search_update(model, tangent_params):

        losses = []
        for step in steps:
            losses.append(loss_at_step(step, model, tangent_params).reshape(1))
        losses = torch.cat(losses)
        step_size = steps[torch.argmin(losses)]

        params = parameters_to_vector(model.parameters())
        new_params = params - step_size*tangent_params
        vector_to_parameters(new_params, model.parameters())

        return step_size

    return grid_line_search_update


def experiment_data_amount_poisson_1d_NGD_KAN(grid_res,NGD_info_string=True,exp_name='poisson1d_NGD_KAN',save_plot=True):
    solver_device('cuda')
    exp_dict_list = []
    
    l_op = 1
    l_bound = 1

    grid_steps = torch.linspace(0, 30, 31)
    steps = 0.5**grid_steps

    x0 = 0
    xmax = 1

    domain = Domain()

    domain.variable('x', [x0, xmax], grid_res, dtype='float64')

    boundaries = Conditions()

    boundaries.dirichlet({'x': x0}, value=u)
    boundaries.dirichlet({'x': xmax}, value=u)

    grid = domain.variable_dict['x'].reshape(-1,1)

    equation = Equation()

    poisson = {
        '-d2u/dx2':
            {
            'coeff': -1,
            'term': [0, 0],
            'pow': 1,
            },

        'f(x)':
            {
            'coeff': u_xx(grid),
            'term': [None],
            'pow': 0,
            }
    }

    equation.add(poisson)
    
    net = KAN(layers_hidden=[1,32,32,1])

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=l_op, lambda_bound=l_bound)

    ls_update = grid_line_search_factory(model.solution_cls.evaluate, steps)

    loss, _ = model.solution_cls.evaluate()

    start = time.time()

    iteration=0
    #for iteration in range(100):
    while loss.item()>1e-7:
        
        loss, _ = model.solution_cls.evaluate()
        grads = torch.autograd.grad(loss, model.net.parameters(), retain_graph=True, allow_unused=True)
        grads = replace_none_with_zero(grads)
        f_grads = parameters_to_vector(grads)

        int_res = model.solution_cls.operator._pde_compute()
        bval, true_bval, _, _ = model.solution_cls.boundary.apply_bcs()
        bound_res = bval-true_bval

        # assemble gramian
        G_int  = gramian(model.net, int_res)

        G_bdry = gramian(model.net, bound_res)
        G      = G_int + G_bdry

        # Marquardt-Levenberg
        Id = torch.eye(len(G))
        G = torch.min(torch.tensor([loss, 0.0])) * Id + G
        # compute natural gradient

        #G = G.detach().cpu().numpy()
        #f_grads =f_grads.detach().cpu().numpy()
        #f_nat_grad = np.linalg.lstsq(G, f_grads)[0] 
        #f_nat_grad = torch.from_numpy(np.array(f_nat_grad)).to(torch.float64).to("cuda:0")


        G = G.detach().cpu()
        f_grads =f_grads.detach().cpu()
        f_nat_grad=torch.linalg.lstsq(G, f_grads,driver='gelsd')[0] 
        
        f_nat_grad = f_nat_grad.to("cuda:0")

        

        # one step of NGD
        actual_step = ls_update(model.net, f_nat_grad)
        iteration+=1
        if iteration>1000:
            break
        #if iteration%10 == 0 and NGD_info_string:
        ##if NGD_info_string:
        #    print('iteration= ', iteration)
        #    print('step= ', actual_step.item())
        #    print('loss=' , model.solution_cls.evaluate()[0].item())
    if NGD_info_string:
        print('iteration= ', iteration)
        print('step= ', actual_step.item())
        print('loss=' , model.solution_cls.evaluate()[0].item())
    end = time.time()
    run_time = end-start
    
    grid = domain.build('autograd')
    grid_test = torch.linspace(0, 1, 100).reshape(-1, 1).double()
    error_train = torch.sqrt(torch.mean((u(grid).reshape(-1) - net(grid).reshape(-1)) ** 2))
    error_test = torch.sqrt(torch.mean((u(grid_test).reshape(-1) - net(grid_test).reshape(-1)) ** 2))
    loss = model.solution_cls.evaluate()[0].detach().cpu().numpy()

    lu_f = model.solution_cls.operator.operator_compute()

    lu_f, _ = integration(lu_f, grid)

    #########

    exp_dict={'grid_res': grid_res,
                        'l2_error_train': error_train.item(),
                        'l2_error_test': error_test.item(),
                        'c2_error_train': None,
                        'c2_error_test': None,
                        'loss': loss.item(),
                        "lu_f": lu_f.item(),
                        "lu": None,
                        'run_time': run_time,
                        'type':exp_name}

    exp_dict_list.append(exp_dict)



    print('grid_res=', grid_res)
    print('l2_norm = ', error_train.item())
    print('lu_f = ', lu_f.item())

    if save_plot:
        if not os.path.isdir('examples\\AAAI_expetiments\\figures_{}'.format(exp_name)):
            os.mkdir('examples\\AAAI_expetiments\\figures_{}'.format(exp_name))
        plt.figure()
        plt.plot(grid.detach().cpu().numpy(), u(grid).detach().cpu().numpy(), label='Exact')
        plt.plot(grid.detach().cpu().numpy(), net(grid).detach().cpu().numpy(), '--', label='Predicted')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='upper right')
        plt.savefig('examples\\AAAI_expetiments\\figures_{}\\img_{}.png'.format(exp_name,grid_res))


    return exp_dict_list




if __name__ == '__main__':

    if not os.path.isdir('examples\\AAAI_expetiments\\results'):
        os.mkdir('examples\\AAAI_expetiments\\results')
    

    

    nruns = 1


    #exp_dict_list=[]

    #for N in grid_n:
    #    for _ in range(nruns):
    #        exp_dict_list.append(experiment_data_amount_poisson_1d_PINN_KAN(N))

    

    #exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
    #df = pd.DataFrame(exp_dict_list_flatten)
    #df.to_csv('examples\\AAAI_expetiments\\results\\poisson_PINN_KAN.csv')

    #plt.close()

    exp_dict_list=[]


    for grid_res in range(10, 101, 10):
        for _ in range(nruns):
            exp_dict_list.append(experiment_data_amount_poisson_1d_PSO_KAN(grid_res))

    

    exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
    df = pd.DataFrame(exp_dict_list_flatten)
    df.to_csv('examples\\AAAI_expetiments\\results\\poisson_PSO_KAN.csv')


    plt.close()


    #exp_dict_list=[]

    #for grid_res in range(10, 101, 10):
    #    for _ in range(nruns):
    #        exp_dict_list.append(experiment_data_amount_poisson_1d_lam_KAN(grid_res))


    #exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
    #df = pd.DataFrame(exp_dict_list_flatten)
    #df.to_csv('examples\\AAAI_expetiments\\results\\poisson_lam_KAN.csv')

    #plt.close()


    #exp_dict_list=[]

    #for grid_res in range(10, 101, 10):
    #    for _ in range(nruns):
    #        exp_dict_list.append(experiment_data_amount_poisson_1d_NGD_KAN(grid_res,NGD_info_string=True))

  
    #exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
    #df = pd.DataFrame(exp_dict_list_flatten)
    #df.to_csv('examples\\AAAI_expetiments\\results\\poisson_NGD_KAN.csv')