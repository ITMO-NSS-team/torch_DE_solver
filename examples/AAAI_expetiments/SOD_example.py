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



import pandas as pd

solver_device('cuda')

p_l = 1
v_l = 0
Ro_l = 1
gam_l = 1.4

p_r = 0.1
v_r = 0
Ro_r = 0.125
gam_r = 1.4

x0 = 0.5
h = 0.05

def exact(point):
    N = 100
    Pl = 1
    Pr = 0.1
    Rg = 519.4
    Gl = 1.4
    Gr = 1.4
    Tl = 273
    Tr = 248
    Rol = 1
    Ror = 0.125

    Cr = (Gr * Pr / Ror) ** (1 / 2)
    Cl = (Gl * Pl / Rol) ** (1 / 2)
    vl = 0
    vr = 0
    t = float(point[-1])
    x = float(point[0])
    x0 = 0
    x1 = 1
    xk = 0.5

    eps = 1e-5
    Pc1 = Pl / 2
    vc1 = 0.2
    u = 1
    while u >= eps:
        Pc = Pc1
        vc = vc1
        f = vl + 2 / (Gl - 1) * Cl * (-(Pc / Pl) ** ((Gl - 1) / (2 * Gl)) + 1) - vc
        g = vr + (Pc - Pr) / (Ror * Cr * ((Gr + 1) / (2 * Gr) * Pc / Pr + (Gr - 1) / (2 * Gr)) ** (1 / 2)) - vc
        fp = -2 / (Gl - 1) * Cl * (1 / Pl) ** ((Gl - 1) / 2 / Gl) * (Gl - 1) / 2 / Gl * Pc ** ((Gl - 1) / (2 * Gl) - 1)
        gp = (1 - (Pc - Pr) * (Gr + 1) / (4 * Gr * Pr * ((Gr + 1) / (2 * Gr) * Pc / Pr + (Gr - 1) / (2 * Gr)))) / (
                Ror * Cr * ((Gr + 1) / (2 * Gr) * Pc / Pr + (Gr - 1) / (2 * Gr)) ** (1 / 2))
        fu = -1
        gu = -1
        Pc1 = Pc - (fu * g - gu * f) / (fu * gp - gu * fp)
        vc1 = vc - (f * gp - g * fp) / (fu - gp - gu * fp)
        u1 = abs((Pc - Pc1) / Pc)
        u2 = abs((vc - vc1) / vc)
        u = max(u1, u2)

    Pc = Pc1
    vc = vc1

    if x <= xk - Cl * t:
        p = Pl
        v = vl
        T = Tl
        Ro = Rol
    Roc = Rol / (Pl / Pc) ** (1 / Gl)
    if xk - Cl * t < x <= xk + (vc - (Gl * Pc / Roc) ** (1 / 2)) * t:
        Ca = (vl + 2 * Cl / (Gl - 1) + (xk - x) / t) / (1 + 2 / (Gl - 1))
        va = Ca - (xk - x) / t
        p = Pl * (Ca / Cl) ** (2 * Gl / (Gl - 1))
        v = va
        Ro = Rol / (Pl / p) ** (1 / Gl)
        T = p / Rg / Ro
    if xk + (vc - (Gl * Pc / Roc) ** (1 / 2)) * t < x <= xk + vc * t:
        p = Pc
        Ro = Roc
        v = vc
        T = p / Rg / Ro
    D = vr + Cr * ((Gr + 1) / (2 * Gr) * Pc / Pr + (Gr - 1) / (2 * Gr)) ** (1 / 2)
    if xk + vc * t < x <= xk + D * t:
        p = Pc
        v = vc
        Ro = Ror * ((Gr + 1) * Pc + (Gr - 1) * Pr) / ((Gr + 1) * Pr + (Gr - 1) * Pc)
        T = p / Rg / Ro
    if xk + D * t < x:
        p = Pr
        v = vr
        Ro = Ror
        T = p / Rg / Ro
    return p, v, Ro

def u(grid):
    solution = []
    for point in grid:
        solution.append(exact(point))
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


    l2_norm_pressure = torch.sqrt(sum((predict[:, 0]-exact[:, 0])**2))
    l2_norm_velocity = torch.sqrt(sum((predict[:, 1]-exact[:, 1])**2))
    l2_norm_density = torch.sqrt(sum((predict[:, 2]-exact[:, 2])**2))

    return l2_norm_pressure.detach().cpu().numpy(),l2_norm_velocity.detach().cpu().numpy(),l2_norm_density.detach().cpu().numpy()

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




def SOD_problem_formulation(grid_res):
    
    domain = Domain()
    domain.variable('x', [0, 1], grid_res)
    domain.variable('t', [0, 0.2], grid_res)

    ## BOUNDARY AND INITIAL CONDITIONS
    # p:0, v:1, Ro:2

    def u0(x, x0):
        if x > x0:
            return [p_r, v_r, Ro_r]
        else:
            return [p_l, v_l, Ro_l]

    boundaries = Conditions()

    # Initial conditions at t=0
    x = domain.variable_dict['x']

    u_init0 = np.zeros(x.shape[0])
    u_init1 = np.zeros(x.shape[0])
    u_init2 = np.zeros(x.shape[0])
    j=0
    for i in x:
      u_init0[j] = u0(i, x0)[0]
      u_init1[j] = u0(i, x0)[1]
      u_init2[j] = u0(i, x0)[2]
      j +=1

    bndval1_0 = torch.from_numpy(u_init0)
    bndval1_1 = torch.from_numpy(u_init1)
    bndval1_2 = torch.from_numpy(u_init2)

    boundaries.dirichlet({'x': [0, 1], 't': 0}, value=bndval1_0, var=0)
    boundaries.dirichlet({'x': [0, 1], 't': 0}, value=bndval1_1, var=1)
    boundaries.dirichlet({'x': [0, 1], 't': 0}, value=bndval1_2, var=2)

    #  Boundary conditions at x=0
    boundaries.dirichlet({'x': 0, 't': [0, 0.2]}, value=p_l, var=0)
    boundaries.dirichlet({'x': 0, 't': [0, 0.2]}, value=v_l, var=1)
    boundaries.dirichlet({'x': 0, 't': [0, 0.2]}, value=Ro_l, var=2)

    # Boundary conditions at x=1
    boundaries.dirichlet({'x': 1, 't': [0, 0.2]}, value=p_r, var=0)
    boundaries.dirichlet({'x': 1, 't': [0, 0.2]}, value=v_r, var=1)
    boundaries.dirichlet({'x': 1, 't': [0, 0.2]}, value=Ro_r, var=2)

    '''
    gas dynamic system equations:
    Eiler's equations system for Sod test in shock tube

    '''

    equation = Equation()

    gas_eq1={
            'dro/dt':
            {
                'coeff': 1,
                'term': [1],
                'pow': 1,
                'var': 2
            },
            'v*dro/dx':
            {
                'coeff': 1,
                'term': [[None], [0]],
                'pow': [1, 1],
                'var': [1, 2]
            },
            'ro*dv/dx':
            {
                'coeff': 1,
                'term': [[None], [0]],
                'pow': [1, 1],
                'var': [2, 1]
            }
         }
    gas_eq2 = {
            'ro*dv/dt':
            {
                'coeff': 1,
                'term': [[None], [1]],
                'pow': [1, 1],
                'var': [2, 1]
            },
            'ro*v*dv/dx':
            {
                'coeff': 1,
                'term': [[None],[None], [0]],
                'pow': [1, 1, 1],
                'var': [2, 1, 1]
            },
            'dp/dx':
            {
                'coeff': 1,
                'term': [0],
                'pow': 1,
                'var': 0
            }
         }
    gas_eq3 =  {
            'dp/dt':
            {
                'coeff': 1,
                'term': [1],
                'pow': 1,
                'var': 0
            },
            'gam*p*dv/dx':
            {
                'coeff': gam_l,
                'term': [[None], [0]],
                'pow': [1, 1],
                'var': [0, 1]
            },
            'v*dp/dx':
            {
                'coeff': 1,
                'term': [[None], [0]],
                'pow': [1, 1],
                'var': [1, 0]
            }

         }

    equation.add(gas_eq1)
    equation.add(gas_eq2)
    equation.add(gas_eq3)

    grid = domain.build('autograd')

    return grid,domain,equation,boundaries




def experiment_data_amount_SOD_PINN(grid_res,exp_name='SOD_PINN',save_plot=True):
    solver_device('cuda')
    exp_dict_list = []

    grid,domain,equation,boundaries = SOD_problem_formulation(grid_res)

    net = torch.nn.Sequential(
    torch.nn.Linear(2, 64),
    torch.nn.Tanh(),
    torch.nn.Linear(64, 256),
    torch.nn.Tanh(),
    torch.nn.Linear(256, 256),
    torch.nn.Tanh(),
    torch.nn.Linear(256, 3)
    )

    model = Model(net, domain, equation, boundaries)

    model.compile("autograd", lambda_operator=1, lambda_bound=100)

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                    loss_window=100,
                                    no_improvement_patience=500,
                                    patience=5,
                                    randomize_parameter=1e-6)

    optim = Optimizer('Adam', {'lr': 1e-3})

    start=time.time()
    model.train(optim, 1e5, callbacks=[cb_es])
    end = time.time()

    run_time = end - start

    grid = domain.build('autograd')

    grid_test = torch.cartesian_prod(torch.linspace(0, 1, 100), torch.linspace(0, 0.2, 100))

    u_exact_train = u(grid)

    u_exact_test = u(grid_test)

    error_train = torch.sqrt(torch.mean((u_exact_train - net(grid))** 2, dim=0))

    error_test = torch.sqrt(torch.mean((u_exact_test - net(grid_test)) ** 2 , dim=0))

    loss = model.solution_cls.evaluate()[0].detach().cpu().numpy()



    exp_dict={'grid_res': grid_res,
                          'error_train_pressure': error_train[0].item(),
                          'error_train_velocity': error_train[1].item(),
                          'error_train_density': error_train[2].item(),
                          'error_test_pressure': error_test[0].item(),
                          'error_test_velocity': error_test[1].item(),
                          'error_test_density': error_test[2].item(),
                          'loss': loss.item(),
                          'time_adam': run_time,
                          'type': exp_name}

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE p {}= {}'.format(grid_res, error_test[0]))
    print('RMSE v {}= {}'.format(grid_res, error_test[1]))
    print('RMSE ro {}= {}'.format(grid_res, error_test[2]))

    exp_dict_list.append(exp_dict)

    #if save_plot:
    #    if not os.path.isdir('examples\\AAAI_expetiments\\figures_{}'.format(exp_name)):
    #        os.mkdir('examples\\AAAI_expetiments\\figures_{}'.format(exp_name))
    #    plt.figure()
    #    plt.plot(grid.detach().cpu().numpy(), u(grid).detach().cpu().numpy(), label='Exact')
    #    plt.plot(grid.detach().cpu().numpy(), net(grid.cpu()).detach().cpu().numpy(), '--', label='Predicted')
    #    plt.xlabel('x')
    #    plt.ylabel('y')
    #    plt.legend(loc='upper right')
    #    plt.savefig('examples\\AAAI_expetiments\\figures_{}\\img_{}.png'.format(exp_name,N))

    return exp_dict_list


def experiment_data_amount_SOD_PSO(grid_res,exp_name='SOD_PSO',save_plot=True):
    solver_device('cuda')
    exp_dict_list = []

    grid,domain,equation,boundaries = SOD_problem_formulation(grid_res)

    net = torch.nn.Sequential(
        torch.nn.Linear(2, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 3)
    )

    model = Model(net, domain, equation, boundaries)

    model.compile("autograd", lambda_operator=1, lambda_bound=100)


    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                        loss_window=100,
                                        no_improvement_patience=500,
                                        patience=5,
                                        randomize_parameter=1e-6,
                                        info_string_every=1000,
                                        abs_loss=0.0001)

    optim = Optimizer('Adam', {'lr': 1e-3})

    start=time.time()
    model.train(optim, 1e5, callbacks=[cb_es])
    end = time.time()

    run_time_adam = end - start

    grid = domain.build('autograd')

    grid_test = torch.cartesian_prod(torch.linspace(0, 1, 100), torch.linspace(0, 0.2, 100))

    u_exact_train = u(grid)

    u_exact_test = u(grid_test)

    error_train_adam = torch.sqrt(torch.mean((u_exact_train - net(grid))** 2, dim=0))

    error_test_adam = torch.sqrt(torch.mean((u_exact_test - net(grid_test)) ** 2 , dim=0))

    loss_adam = model.solution_cls.evaluate()[0].detach().cpu().numpy()


    print('Time taken adam {}= {}'.format(grid_res, run_time_adam))
    print('RMSE p {}= {}'.format(grid_res, error_test_adam[0]))
    print('RMSE v {}= {}'.format(grid_res, error_test_adam[1]))
    print('RMSE ro {}= {}'.format(grid_res, error_test_adam[2]))


    ########

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                        loss_window=100,
                                        no_improvement_patience=100,
                                        patience=2,
                                        randomize_parameter=1e-5,
                                        verbose=False)

    optim = Optimizer('PSO', {'pop_size': 50, #30
                                  'b': 0.4, #0.5
                                  'c2': 0.5, #0.05
                                  'c1': 0.5, 
                                  'variance': 5e-2,
                                  'lr': 1e-3})
    start = time.time()
    model.train(optim, 2e4, save_model=False, callbacks=[cb_es])
    end = time.time()

    run_time_pso=end-start

    error_train_pso = torch.sqrt(torch.mean((u_exact_train - net(grid))** 2, dim=0))

    error_test_pso = torch.sqrt(torch.mean((u_exact_test - net(grid_test)) ** 2 , dim=0))

    loss_pso = model.solution_cls.evaluate()[0].detach().cpu().numpy()

    #########

    

    exp_dict={'grid_res': grid_res,
                        'error_train_pressure_adam': error_train_adam[0].item(),
                        'error_train_velocity_adam': error_train_adam[1].item(),
                        'error_train_density_adam': error_train_adam[2].item(),
                        'error_test_pressure_adam': error_train_adam[0].item(),
                        'error_test_velocity_adam': error_train_adam[1].item(),
                        'error_test_density_adam': error_train_adam[2].item(),
                        'error_train_pressure_pso': error_train_pso[0].item(),
                        'error_train_velocity_pso': error_train_pso[1].item(),
                        'error_train_density_pso': error_train_pso[2].item(),
                        'error_test_pressure_pso': error_train_pso[0].item(),
                        'error_test_velocity_pso': error_train_pso[1].item(),
                        'error_test_density_pso': error_train_pso[2].item(),
                        'loss_adam': loss_adam.item(),
                        'loss_pso': loss_pso.item(),
                        'time_adam': run_time_adam,
                        'time_pso': run_time_pso,
                        'type': exp_name}

    print('Time taken pso {}= {}'.format(grid_res, run_time_pso))
    print('RMSE p {}= {}'.format(grid_res, error_test_pso[0]))
    print('RMSE v {}= {}'.format(grid_res, error_test_pso[1]))
    print('RMSE ro {}= {}'.format(grid_res, error_test_pso[2]))

    exp_dict_list.append(exp_dict)

    return exp_dict_list




def experiment_data_amount_SOD_lam(grid_res,exp_name='SOD_lam',save_plot=True):

    solver_device('cuda')
    exp_dict_list = []

    grid,domain,equation,boundaries = SOD_problem_formulation(grid_res)

    net = torch.nn.Sequential(
        torch.nn.Linear(2, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 3)
    )

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=1, lambda_bound=100)

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                        loss_window=100,
                                        no_improvement_patience=1000,
                                        patience=5,
                                        randomize_parameter=1e-6)

    cb_lam = adaptive_lambda.AdaptiveLambda()

    optim = Optimizer('Adam', {'lr': 1e-3})

    start=time.time()
    model.train(optim, 2e5, callbacks=[cb_es,cb_lam])
    end = time.time()

    run_time = end - start

    grid = domain.build('autograd')

    grid_test = torch.cartesian_prod(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100))

    u_exact_train = u(grid)

    u_exact_test = u(grid_test)

    error_train = torch.sqrt(torch.mean((u_exact_train - net(grid))** 2, dim=0))

    error_test = torch.sqrt(torch.mean((u_exact_test - net(grid_test)) ** 2 , dim=0))

    loss = model.solution_cls.evaluate()[0].detach().cpu().numpy()

    lu_f = model.solution_cls.operator.operator_compute()

    lu_f, gr = integration(lu_f, grid)

    lu_f, _ = integration(lu_f, gr)


    exp_dict={'grid_res': grid_res,
                        'error_train_pressure': error_train[0].item(),
                        'error_train_velocity': error_train[1].item(),
                        'error_train_density': error_train[2].item(),
                        'error_test_pressure': error_train[0].item(),
                        'error_test_velocity': error_train[1].item(),
                        'error_test_density': error_train[2].item(),
                        'loss': loss.item(),
                        'time': run_time,
                        'type': exp_name}

    print('Time taken pso {}= {}'.format(grid_res, run_time))
    print('RMSE p {}= {}'.format(grid_res, error_test[0]))
    print('RMSE v {}= {}'.format(grid_res, error_test[1]))
    print('RMSE ro {}= {}'.format(grid_res, error_test[2]))

    exp_dict_list.append(exp_dict)

    return exp_dict_list




def experiment_data_amount_SOD_fourier(grid_res,exp_name='SOD_fourier',save_plot=True):
    solver_device('cuda')
    exp_dict_list = []

    grid,domain,equation,boundaries = SOD_problem_formulation(grid_res)

    FFL = Fourier_embedding(L=[2,2], M=[1,1])

    out = FFL.out_features


    net = torch.nn.Sequential(
        FFL,
        torch.nn.Linear(out, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 1)
    )

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=1, lambda_bound=100)

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                        loss_window=100,
                                        no_improvement_patience=1000,
                                        patience=5,
                                        randomize_parameter=1e-6)

    optim = Optimizer('Adam', {'lr': 1e-3})

    start=time.time()
    model.train(optim, 2e5, callbacks=[cb_es])
    end = time.time()

    run_time = end - start

    grid = domain.build('autograd')

    grid_test = torch.cartesian_prod(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100))

    u_exact_train = u(grid)

    u_exact_test = u(grid_test)

    error_train = torch.sqrt(torch.mean((u_exact_train - net(grid))** 2, dim=0))

    error_test = torch.sqrt(torch.mean((u_exact_test - net(grid_test)) ** 2 , dim=0))

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

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_test))


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


def experiment_data_amount_SOD_NGD(grid_res,NGD_info_string=True,exp_name='burgers1d_NGD',save_plot=True):

    exp_dict_list = []

    
    l_op = 1
    l_bound = 100
    grid_steps = torch.linspace(0, 30, 31)
    steps = 0.5**grid_steps

    grid,domain,equation,boundaries = SOD_problem_formulation(grid_res)
    
    net = torch.nn.Sequential(
        torch.nn.Linear(2, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 1)
    )

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=l_op, lambda_bound=l_bound)


    ls_update = grid_line_search_factory(model.solution_cls.evaluate, steps)

    loss, _ = model.solution_cls.evaluate()


    start = time.time()

    iteration=0
    #for iteration in range(100):
    while loss.item()>1e-6:
        
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

    grid_test = torch.cartesian_prod(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100))

    u_exact_train = u(grid)

    u_exact_test = u(grid_test)

    error_train = torch.sqrt(torch.mean((u_exact_train - net(grid))** 2, dim=0))

    error_test = torch.sqrt(torch.mean((u_exact_test - net(grid_test)) ** 2 , dim=0))

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
    
    nruns = 1


    exp_dict_list=[]

    



    #for grid_res in range(10, 101, 10):
    #    for _ in range(nruns):
    #        exp_dict_list.append(experiment_data_amount_SOD_PSO(grid_res))

    

    #exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
    #df = pd.DataFrame(exp_dict_list_flatten)
    #df.to_csv('examples\\AAAI_expetiments\\results\\SOD_PSO.csv')


    exp_dict_list=[]

    for grid_res in range(10, 61, 10):
        for _ in range(nruns):
            exp_dict_list.append(experiment_data_amount_SOD_lam(grid_res))
            exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
            df = pd.DataFrame(exp_dict_list_flatten)
            df.to_csv('examples\\AAAI_expetiments\\results\\SOD_lam_{}.csv'.format(grid_res))

    exp_dict_list=[]

    for grid_res in range(10, 61, 10):
        for _ in range(nruns):
            exp_dict_list.append(experiment_data_amount_SOD_NGD(grid_res,NGD_info_string=True))
            exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
            df = pd.DataFrame(exp_dict_list_flatten)
            df.to_csv('examples\\AAAI_expetiments\\results\\SOD_NGD_{}.csv'.format(grid_res))
