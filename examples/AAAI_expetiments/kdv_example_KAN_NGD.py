import torch
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import time
from scipy.integrate import quad

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

mu = 0.01 / np.pi

def soliton(x,t):
    E=np.exp(1)
    s=((18*torch.exp((1/125)*(t + 25*x))*(16*torch.exp(2*t) +
       1000*torch.exp((126*t)/125 + (4*x)/5) + 9*torch.exp(2*x) + 576*torch.exp(t + x) +
       90*torch.exp((124*t)/125 + (6*x)/5)))/(5*(40*torch.exp((126*t)/125) +
        18*torch.exp(t + x/5) + 9*torch.exp((6*x)/5) + 45*torch.exp(t/125 + x))**2))
    return s

def u(grid):
    solution = []
    for point in grid:
        x=point[0]
        t=point[1]
        s=soliton(x,t)
        solution.append(s)
    return torch.tensor(solution)


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




def kdv_problem_formulation(grid_res):
    
    domain = Domain()
    domain.variable('x', [-10, 10], grid_res)
    domain.variable('t', [0, 1], grid_res)
    
    boundaries = Conditions()


    # u(0,t) = u(1,t)
    boundaries.periodic([{'x': -10, 't': [0, 1]}, {'x': 10, 't': [0, 1]}])


    """
    Initial conditions at t=0
    """

    x = domain.variable_dict['x']

    boundaries.dirichlet({'x': [-10, 10], 't': 0}, value=soliton(x,torch.tensor([0])))

    equation = Equation()
    
    # operator is du/dt+6u*(du/dx)+d3u/dx3-sin(x)*cos(t)=0
    kdv = {
        '1*du/dt**1':
            {
                'coeff': 1,
                'du/dt': [1],
                'pow': 1,
                'var': 0
            },
        '6*u**1*du/dx**1':
            {
                'coeff': 6,
                'u*du/dx': [[None], [0]],
                'pow': [1,1],
                'var':[0,0]
            },
        'd3u/dx3**1':
            {
                'coeff': 1,
                'd3u/dx3': [0, 0, 0],
                'pow': 1,
                'var':0
            }
    }
    
    equation.add(kdv)

    grid = domain.build('autograd')

    return grid,domain,equation,boundaries






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


from tedeous.utils import create_random_fn

def experiment_data_amount_kdv_NGD_KAN(grid_res,NGD_info_string=True,exp_name='kdv_NGD_KAN',save_plot=True,loss_window=20,randomize_parameter=1e-6):

    _r= create_random_fn(randomize_parameter)

    exp_dict_list = []

    
    l_op = 1
    l_bound = 100
    grid_steps = torch.linspace(0, 10, 11)
    steps = 0.5**grid_steps

    grid,domain,equation,boundaries = kdv_problem_formulation(grid_res)
    
    net = KAN(layers_hidden=[2,16,16,1])

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=l_op, lambda_bound=l_bound)


    ls_update = grid_line_search_factory(model.solution_cls.evaluate, steps)

    loss, _ = model.solution_cls.evaluate()


    start = time.time()

    iteration=0
    #for iteration in range(100):

    

    min_loss=loss.item()

    last_loss = np.zeros(loss_window) + float(min_loss)

    patience=0

    while True:
        
        loss, _ = model.solution_cls.evaluate()
        grads = torch.autograd.grad(loss, model.net.parameters(), retain_graph=True, allow_unused=True)
        grads = replace_none_with_zero(grads)
        f_grads = parameters_to_vector(grads)

        int_res = model.solution_cls.operator._pde_compute()
        bval, true_bval, _, _ = model.solution_cls.boundary.apply_bcs()
        bound_res = bval.reshape(-1)-true_bval.reshape(-1)

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

        #G=G.float()
        #f_grads=f_grads.float()

        #f_nat_grad=torch.linalg.lstsq(G,f_grads)[0]

        #f_nat_grad=f_nat_grad.double()

        #g = g.detach().cpu()
        #f_grads =f_grads.detach().cpu()
        #f_nat_grad=torch.linalg.lstsq(G, f_grads)[0] 
        
        G = G.detach().cpu().numpy()
        f_grads =f_grads.detach().cpu().numpy()

        f_nat_grad=np.linalg.lstsq(G, f_grads,rcond=-1)[0] 

        f_nat_grad=torch.from_numpy(f_nat_grad)

        f_nat_grad = f_nat_grad.to("cuda:0")

        

        # one step of NGD
        actual_step = ls_update(model.net, f_nat_grad)
        iteration+=1

        cur_loss=model.solution_cls.evaluate()[0].item()

        print(cur_loss)
        
        if abs(cur_loss)<1e-8:
            break

        last_loss[(iteration-3)%loss_window]=cur_loss

        if iteration>0 and iteration%loss_window==0:
            line = np.polyfit(range(loss_window), last_loss, 1)
            print(last_loss)
            print('crit={}'.format(abs(line[0] / cur_loss)))
            if abs(line[0] / cur_loss)<1e-6:
                print('increasing patience')
                patience+=1
                if patience < 5:
                    print('model before = {}'.format(parameters_to_vector(model.net.parameters())))
                    loss, _ = model.solution_cls.evaluate()
                    print('loss before = {}'.format(loss.item()))
                    model.net=model.net.apply(_r)
                    print('model after = {}'.format(parameters_to_vector(model.net.parameters())))
                    loss, _ = model.solution_cls.evaluate()
                    print('loss after = {}'.format(loss.item()))
                else:
                    break
                

        if iteration>1000:
            break

    if NGD_info_string:
        print('iteration= ', iteration)
        print('step= ', actual_step.item())
        print('loss=' , model.solution_cls.evaluate()[0].item())
    end = time.time()
    run_time = end-start
    
    grid = domain.build('autograd')

    grid_test = torch.cartesian_prod(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100))

    u_exact_train = u(grid).reshape(-1)

    u_exact_test = u(grid_test).reshape(-1)

    error_train = torch.sqrt(torch.mean((u_exact_train - net(grid).reshape(-1)) ** 2))

    error_test = torch.sqrt(torch.mean((u_exact_test - net(grid_test).reshape(-1)) ** 2))

    loss = model.solution_cls.evaluate()[0].detach().cpu().numpy()

    lu_f = model.solution_cls.operator.operator_compute()

    lu_f, gr = integration(lu_f, grid)

    lu_f, _ = integration(lu_f, gr)


    exp_dict={'grid_res': grid_res,
                          'error_train': error_train.item(),
                          'error_test': error_test.item(),
                          'loss': loss.item(),
                          "lu_f_adam": lu_f.item(),
                          'time_adam': run_time,
                          'type': exp_name}

    exp_dict_list.append(exp_dict)



    print('grid_res=', grid_res)
    print('l2_norm = ', error_train.item())
    print('lu_f = ', lu_f.item())


    return exp_dict_list




if __name__ == '__main__':

    if not os.path.isdir('examples\\AAAI_expetiments\\results'):
        os.mkdir('examples\\AAAI_expetiments\\results')
    

    exp_dict_list=[]

    nruns = 1


    for grid_res in range(10, 61, 10):
        for _ in range(nruns):
            exp_dict_list.append(experiment_data_amount_kdv_NGD_KAN(grid_res))
            exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
            df = pd.DataFrame(exp_dict_list_flatten)
            df.to_csv('examples\\AAAI_expetiments\\results\\kdv_NGD_KAN_{}.csv'.format(grid_res))
  
