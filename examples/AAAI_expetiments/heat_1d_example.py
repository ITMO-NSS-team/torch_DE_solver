
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



import pandas as pd

solver_device('cuda')

mu = 1 / 4

def u(grid):
  return torch.exp(-torch.pi**2*grid[:,1]*mu)*torch.sin(torch.pi*grid[:,0])


def u_net(net, x):
    net = net.to('cpu')
    x = x.to('cpu')
    return net(x).detach()


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




def heat1d_problem_formulation(grid_res):
    
    domain = Domain()
    domain.variable('x', [0, 1], grid_res, dtype='float64')
    domain.variable('t', [0, 1], grid_res, dtype='float64')

    boundaries = Conditions()
    x = domain.variable_dict['x']
    boundaries.dirichlet({'x': [0, 1], 't': 0}, value=torch.sin(np.pi * x))

    boundaries.dirichlet({'x': 0., 't': [0, 1]}, value=0)

    boundaries.dirichlet({'x': 1., 't': [0, 1]}, value=0)

    equation = Equation()

    heat_eq = {
        'du/dt**1':
            {
                'coeff': 1.,
                'du/dt': [1],
                'pow': 1,
                'var': 0
            },
        '-mu*d2u/dx2':
            {
                'coeff': -mu,
                'd2u/dx2': [0, 0],
                'pow': 1,
                'var': 0
            }
    }

    equation.add(heat_eq)

    grid = domain.build('autograd')

    return grid,domain,equation,boundaries




def experiment_data_amount_heat_1d_PINN(grid_res,exp_name='poisson1d_PINN',save_plot=True):
    solver_device('cuda')
    exp_dict_list = []

    grid,domain,equation,boundaries = heat1d_problem_formulation(grid_res)

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


    print('grid_res=', N)
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


def experiment_data_amount_poisson_1d_PSO(grid_res,exp_name='poisson1d_PSO',save_plot=True):
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
    
    cb_es = early_stopping.EarlyStopping(eps=1e-9, randomize_parameter=1e-6)

    optimizer = Optimizer('Adam', {'lr': 1e-4})

    start=time.time()

    model.train(optimizer, 5e5, save_model=False, callbacks=[cb_es])

    l2_pinn = l2_norm(net, grid)
    print('l2_norm_before_PSO = ', l2_pinn)
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

    print('grid_res=', N)
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





def experiment_data_amount_poisson_1d_mat(grid_res,exp_name='poisson1d_mat',save_plot=True):
    solver_device('cuda')

    exp_dict_list = []

    x0 = 0
    xmax = 1

    domain = Domain()

    domain.variable('x', [x0, xmax], grid_res, dtype='float32')

    boundaries = Conditions()

    boundaries.dirichlet({'x': x0}, value=u)
    boundaries.dirichlet({'x': xmax}, value=u)

    grid = domain.variable_dict['x']
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

    cb_es = early_stopping.EarlyStopping(eps=1e-9, randomize_parameter=1e-6)

    optimizer = Optimizer('Adam', {'lr': 1e-4})

    start=time.time()

    model.train(optimizer, 5e5, save_model=False, callbacks=[cb_es])

    grid = torch.linspace(x0, xmax, grid_res+1).reshape(-1,1)

    l2_pinn = l2_norm(net, grid)
    print('l2_norm_before_mat = ', l2_pinn)
    net = net.to('cuda')


    ########
    solver_device('cpu')

    net = net(grid).reshape(1, grid_res+1).detach().cpu()

    domain = Domain()

    domain.variable('x', [x0, xmax], grid_res, dtype='float32')

    boundaries = Conditions()

    boundaries.dirichlet({'x': x0}, value=u)
    boundaries.dirichlet({'x': xmax}, value=u)

    grid = domain.variable_dict['x']
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

    model = Model(net, domain, equation, boundaries)

    model.compile('mat', lambda_operator=1, lambda_bound=1, derivative_points=3)

    cb_es = early_stopping.EarlyStopping(eps=1e-9, randomize_parameter=1e-6)

    optimizer = Optimizer('LBFGS', {'lr': 1e-2})

    model.train(optimizer, 5e5, save_model=False, callbacks=[cb_es])

    end=time.time()

    run_time=end-start

    end_loss = model.solution_cls.evaluate()[0].detach().cpu().numpy()

    grid = domain.build('mat')

    exp_dict={'grid_res': grid_res,
                    'l2_error_train': l2_norm_mat(net, grid),
                    'l2_error_test': None,
                    'c2_error_train': None,
                    'c2_error_test': None,
                    'loss': end_loss,
                    "lu_f": None,
                    "lu": None,
                    'run_time': run_time,
                    'type':exp_name}

    exp_dict_list.append(exp_dict)

    print('grid_res=', grid_res)
    print('l2_norm = ', l2_norm_mat(net, grid))

    if save_plot:
        if not os.path.isdir('examples\\AAAI_expetiments\\figures_{}'.format(exp_name)):
            os.mkdir('examples\\AAAI_expetiments\\figures_{}'.format(exp_name))
        plt.figure()
        plt.plot(grid.detach().cpu().numpy().reshape(-1), u(grid).detach().cpu().numpy().reshape(-1), label='Exact')
        plt.plot(grid.detach().cpu().numpy().reshape(-1), net.detach().cpu().numpy().reshape(-1), '--', label='Predicted')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='upper right')
        plt.savefig('examples\\AAAI_expetiments\\figures_{}\\img_{}.png'.format(exp_name,N))

    return exp_dict_list


def experiment_data_amount_poisson_1d_lam(grid_res,exp_name='poisson1d_lam',save_plot=True):
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

    print('grid_res=', N)
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




def experiment_data_amount_poisson_1d_fourier(grid_res,exp_name='poisson1d_fourier',save_plot=True):
    solver_device('cuda')

    exp_dict_list = []

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

    FFL = Fourier_embedding(L=[2], M=[1])

    out = FFL.out_features

    net = torch.nn.Sequential(
            FFL,
            torch.nn.Linear(out, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 1)
        )

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=1, lambda_bound=1)
    
    cb_es = early_stopping.EarlyStopping(eps=1e-9, randomize_parameter=1e-6)

    optimizer = Optimizer('Adam', {'lr': 1e-3}, gamma=0.9, decay_every=1000)

    start=time.time()

    model.train(optimizer, 5e5, save_model=False, callbacks=[cb_es])

    end=time.time()

    run_time=end-start

    end_loss = model.solution_cls.evaluate()[0].detach().cpu().numpy()

    grid_test = torch.linspace(0, 1, 100).reshape(-1,1)

    exp_dict={'grid_res': grid_res,
                        'l2_error_train': l2_norm_fourier(net, grid),
                        'l2_error_test': l2_norm_fourier(net, grid_test),
                        'c2_error_train': None,
                        'c2_error_test': None,
                        'loss': end_loss,
                        "lu_f": None,
                        "lu": None,
                        'run_time': run_time,
                        'type':exp_name}

    exp_dict_list.append(exp_dict)

    print('grid_res=', grid_res)
    # print('c2_norm = ', c2_norm(net, grid_error))
    print('l2_norm = ', l2_norm_fourier(net, grid_test))

    if save_plot:
        if not os.path.isdir('examples\\AAAI_expetiments\\figures_{}'.format(exp_name)):
            os.mkdir('examples\\AAAI_expetiments\\figures_{}'.format(exp_name))
        plt.figure()
        plt.plot(grid.detach().cpu().numpy(), u(grid).detach().cpu().numpy(), label='Exact')
        plt.plot(grid.detach().cpu().numpy(), net(grid.to(torch.device('cuda:0'))).detach().cpu().numpy(), '--', label='Predicted')
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


def experiment_data_amount_poisson_1d_NGD(grid_res,NGD_info_string=True,exp_name='poisson1d_NGD',save_plot=True):
    solver_device('cuda')
    exp_dict_list = []
    start = time.time()
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
    
    net = torch.nn.Sequential(
        torch.nn.Linear(1, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 1)
    )

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=l_op, lambda_bound=l_bound)

    ls_update = grid_line_search_factory(model.solution_cls.evaluate, steps)

    loss, _ = model.solution_cls.evaluate()
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
    # Создайте список для каждой части
    part1 = np.arange(2, 10, 3)
    part2 = np.arange(10, 100, 30)
    part3 = np.arange(100, 1000, 300)
    part4 = np.arange(1000, 11000, 3000)
    part5 = np.array([100000])

    neurons = np.array([2, 8, 16, 32, 64, 128, 216, 512])

    grid_n = np.concatenate([part1, part2, part3, part4])

    if not os.path.isdir('examples\\AAAI_expetiments\\results'):
        os.mkdir('examples\\AAAI_expetiments\\results')
    

    exp_dict_list=[]

    nruns = 1

    for N in grid_n:
        for _ in range(nruns):
            exp_dict_list.append(experiment_data_amount_poisson_1d_PINN(N))

    

    exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
    df = pd.DataFrame(exp_dict_list_flatten)
    df.to_csv('examples\\AAAI_expetiments\\results\\poisson_PINN.csv')

    exp_dict_list=[]


    for N in grid_n:
        for _ in range(nruns):
            exp_dict_list.append(experiment_data_amount_poisson_1d_PSO(N))

    

    exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
    df = pd.DataFrame(exp_dict_list_flatten)
    df.to_csv('examples\\AAAI_expetiments\\results\\poisson_PSO.csv')

    exp_dict_list=[]

    for N in grid_n:
        for _ in range(nruns):
            exp_dict_list.append(experiment_data_amount_poisson_1d_mat(N))

    

    exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
    df = pd.DataFrame(exp_dict_list_flatten)
    df.to_csv('examples\\AAAI_expetiments\\results\\poisson_mat.csv')

    exp_dict_list=[]

    for N in grid_n:
        for _ in range(nruns):
            exp_dict_list.append(experiment_data_amount_poisson_1d_lam(N))


    exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
    df = pd.DataFrame(exp_dict_list_flatten)
    df.to_csv('examples\\AAAI_expetiments\\results\\poisson_lam.csv')

    exp_dict_list=[]

    for N in grid_n:
        for _ in range(nruns):
            exp_dict_list.append(experiment_data_amount_poisson_1d_fourier(N))

    

    exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
    df = pd.DataFrame(exp_dict_list_flatten)
    df.to_csv('examples\\AAAI_expetiments\\results\\poisson_fourier.csv')

    exp_dict_list=[]

    for N in grid_n:
        for _ in range(nruns):
            exp_dict_list.append(experiment_data_amount_poisson_1d_NGD(N,NGD_info_string=True))

  
    exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
    df = pd.DataFrame(exp_dict_list_flatten)
    df.to_csv('examples\\AAAI_expetiments\\results\\poisson_NGD.csv')
