
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
from tedeous.callbacks import plot, early_stopping, adaptive_lambda, cache
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




def experiment_data_amount_poisson_1d_adam_lbfgs_nncg(grid_res,exp_name='poisson1d_adam_lbfgs_nncg',save_plot=True):
    solver_device('cuda')
    exp_dict_list = []

    grid,domain,equation,boundaries = poisson1d_problem_formulation(grid_res)

    net = torch.nn.Sequential(
            torch.nn.Linear(1, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 1)
        )

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=1, lambda_bound=1)
    
    cb_es = early_stopping.EarlyStopping(eps=1e-6, randomize_parameter=1e-6)

    optimizer = Optimizer('Adam', {'lr': 1e-4})

    start=time.time()

    model.train(optimizer, 20000, callbacks=[cb_es], info_string_every=500)

    end = time.time()

    time_adam = end - start

    grid_test = torch.linspace(0, 1, 100).reshape(-1, 1).float()

    l2_error_adam_train=l2_norm(net, grid),
    l2_error_adam_test=l2_norm(net, grid_test),
    c2_error_adam_train=c2_norm(net, grid),
    c2_error_adam_test=c2_norm(net, grid_test),

    loss_adam = model.solution_cls.evaluate()[0].detach().cpu().numpy()

    lu_f = model.solution_cls.operator.operator_compute()

    lu_f_adam, _ = integration(lu_f, grid)

    lu = u_net_xx(net, grid)

    lu_adam, _ = integration(lu, grid)
    
    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE_adam {}= {}'.format(grid_res, l2_norm(net, grid_test)))

    ########

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                        loss_window=100,
                                        no_improvement_patience=200,
                                        patience=2,
                                        randomize_parameter=1e-5,
                                        verbose=False)

    optim = Optimizer('LBFGS', {'history_size': 100,
                                "line_search_fn": 'strong_wolfe'})

    start = time.time()
    model.train(optim, 2000, save_model=False, callbacks=[cb_es], info_string_every=100)
    end = time.time()
    time_LBFGS = end - start

    grid_test = torch.linspace(0, 1, 100).reshape(-1, 1).float()

    l2_error_LBFGS_train=l2_norm(net, grid),
    l2_error_LBFGS_test=l2_norm(net, grid_test),
    c2_error_LBFGS_train=c2_norm(net, grid),
    c2_error_LBFGS_test=c2_norm(net, grid_test),

    loss_LBFGS = model.solution_cls.evaluate()[0].detach().cpu().numpy()

    lu_f = model.solution_cls.operator.operator_compute()

    lu_f_LBFGS, _ = integration(lu_f, grid)

    lu = u_net_xx(net, grid)

    lu_LBFGS, _ = integration(lu, grid)

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE_LBFGS {}= {}'.format(grid_res, l2_norm(net, grid_test)))

    ########

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                        loss_window=100,
                                        no_improvement_patience=200,
                                        patience=2,
                                        randomize_parameter=1e-5,
                                        verbose=False)

    optim = Optimizer('NNCG', {'mu': 1e-1,
                               "rank": 60,
                               'line_search_fn': "armijo",
                               'verbose': False})

    start = time.time()
    model.train(optim, 3000, save_model=False, callbacks=[cb_es], info_string_every=100)
    end = time.time()
    time_NNCG = end - start

    grid_test = torch.linspace(0, 1, 100).reshape(-1, 1).float()

    l2_error_NNCG_train=l2_norm(net, grid),
    l2_error_NNCG_test=l2_norm(net, grid_test),
    c2_error_NNCG_train=c2_norm(net, grid),
    c2_error_NNCG_test=c2_norm(net, grid_test),

    loss_NNCG = model.solution_cls.evaluate()[0].detach().cpu().numpy()

    lu_f = model.solution_cls.operator.operator_compute()

    lu_f_NNCG, _ = integration(lu_f, grid)

    lu = u_net_xx(net, grid)

    lu_NNCG, _ = integration(lu, grid)


    exp_dict={'grid_res': grid_res,
                        'l2_error_train_adam': l2_error_adam_train,
                        'l2_error_test_adam': l2_error_adam_test,
                        'c2_error_train_adam': c2_error_adam_train,
                        'c2_error_test_adam': c2_error_adam_test,
                        'l2_error_train_LBFGS': l2_error_LBFGS_train,
                        'l2_error_test_LBFGS': l2_error_LBFGS_test,
                        'c2_error_train_LBFGS': c2_error_LBFGS_train,
                        'c2_error_test_LBFGS': c2_error_LBFGS_test,
                        'l2_error_train_NNCG': l2_error_NNCG_train,
                        'l2_error_test_NNCG': l2_error_NNCG_test,
                        'c2_error_train_NNCG': c2_error_NNCG_train,
                        'c2_error_test_NNCG': c2_error_NNCG_test,
                        'loss_adam': loss_adam,
                        'loss_LBFGS': loss_LBFGS,
                        'loss_NNCG': loss_NNCG,
                        "lu_f_adam": lu_f_adam.item(),
                        "lu_adam": lu_adam.item(),
                        "lu_f_LBFGS": lu_f_LBFGS.item(),
                        "lu_LBFGS": lu_LBFGS.item(),
                        "lu_f_NNCG": lu_f_NNCG.item(),
                        "lu_NNCG": lu_NNCG.item(),
                        'time_adam': time_adam,
                        'time_LBFGS': time_LBFGS,
                        'time_NNCG': time_NNCG,
                        'type':exp_name}

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE_NNCG {}= {}'.format(grid_res, l2_norm(net, grid_test)))

    exp_dict_list.append(exp_dict)

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




if __name__ == '__main__':

    if not os.path.isdir('examples\\AAAI_expetiments\\results'):
        os.mkdir('examples\\AAAI_expetiments\\results')
    
    part1 = np.arange(2, 10, 3)
    part2 = np.arange(10, 100, 30)
    part3 = np.arange(100, 1000, 300)
    #part4 = np.arange(1000, 11000, 3000)
    
    #grid_n = np.concatenate([part1, part2, part3, part4])
    grid_n = np.concatenate([part1, part2, part3])

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


    for grid_res in grid_n:
        for _ in range(nruns):
            exp_dict_list.append(experiment_data_amount_poisson_1d_adam_lbfgs_nncg(grid_res))
            exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
            df = pd.DataFrame(exp_dict_list_flatten)
            df.to_csv('examples\\AAAI_expetiments\\results\\poisson_adam_lbfgs_nncg_{}.csv'.format(grid_res))



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