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


# Lotka-Volterra equations also known as predator-prey equations, describe the variation in populations
# of two species which interact via predation.
# For example, wolves (predators) and deer (prey). This is a classical model to represent the dynamic of two populations.

# Let αlpha > 0, beta > 0, delta > 0 and gamma > 0 . The system is given by

# dx/dt = x(alpha-beta*y)
# dy/dt = y(-delta+gamma*x)

# Where 'x' represents prey population and 'y' predators population. It’s a system of first-order ordinary differential equations.
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import time
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import  early_stopping
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device, check_device, device_type


from tedeous.models import KAN


alpha = 20.
beta = 20.
delta = 20.
gamma = 20.
x0 = 4.
y0 = 2.
t0 = 0.
tmax = 1.

def exact(grid):
    # scipy.integrate solution of Lotka_Volterra equations and comparison with NN results

    def deriv(X, t, alpha, beta, delta, gamma):
        x, y = X
        dotx = x * (alpha - beta * y)
        doty = y * (-delta + gamma * x)
        return np.array([dotx, doty])

    t = grid.cpu()

    X0 = [x0, y0]
    res = integrate.odeint(deriv, X0, t, args = (alpha, beta, delta, gamma))
    x, y = res.T
    return np.hstack((x.reshape(-1,1),y.reshape(-1,1)))

def u(grid):
    solution=exact(grid)
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




def LV_problem_formulation(grid_res):
    
    domain = Domain()
    domain.variable('t', [0, tmax], grid_res)

    boundaries = Conditions()
    #initial conditions
    boundaries.dirichlet({'t': 0}, value=x0, var=0)
    boundaries.dirichlet({'t': 0}, value=y0, var=1)

    #equation system
    # eq1: dx/dt = x(alpha-beta*y)
    # eq2: dy/dt = y(-delta+gamma*x)

    # x var: 0
    # y var:1
    
    equation = Equation()

    eq1 = {
        'dx/dt':{
            'coeff': 1,
            'term': [0],
            'pow': 1,
            'var': [0]
        },
        '-x*alpha':{
            'coeff': -alpha,
            'term': [None],
            'pow': 1,
            'var': [0]
        },
        '+beta*x*y':{
            'coeff': beta,
            'term': [[None], [None]],
            'pow': [1, 1],
            'var': [0, 1]
        }
    }

    eq2 = {
        'dy/dt':{
            'coeff': 1,
            'term': [0],
            'pow': 1,
            'var': [1]
        },
        '+y*delta':{
            'coeff': delta,
            'term': [None],
            'pow': 1,
            'var': [1]
        },
        '-gamma*x*y':{
            'coeff': -gamma,
            'term': [[None], [None]],
            'pow': [1, 1],
            'var': [0, 1]
        }
    }

    equation.add(eq1)
    equation.add(eq2)

    grid = domain.build('autograd')

    return grid,domain,equation,boundaries





def experiment_data_amount_LV_PSO(grid_res,exp_name='LV_PSO',save_plot=True):
    solver_device('cuda')
    exp_dict_list = []

    grid,domain,equation,boundaries = LV_problem_formulation(grid_res)

    net = KAN(layers_hidden=[1,32,32,2])

    model = Model(net, domain, equation, boundaries)

    model.compile("autograd", lambda_operator=1, lambda_bound=100)


    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                        loss_window=100,
                                        no_improvement_patience=500,
                                        patience=3,
                                        randomize_parameter=1e-5)

    optim = Optimizer('Adam', {'lr': 1e-3})

    start=time.time()
    model.train(optim, 1e6, callbacks=[cb_es])
    end = time.time()

    run_time_adam = end - start

    grid = domain.build('autograd')

    grid_test = torch.linspace(0, tmax, 100)

    u_exact_train = u(grid.cpu().reshape(-1))

    u_exact_test = u(grid_test.cpu().reshape(-1))

    error_train_adam = torch.sqrt(torch.mean((u_exact_train - net(grid))** 2, dim=0))

    error_test_adam = torch.sqrt(torch.mean((u_exact_test - net(grid_test.reshape(-1,1))) ** 2 , dim=0))

    loss_adam = model.solution_cls.evaluate()[0].detach().cpu().numpy()


    print('Time taken adam {}= {}'.format(grid_res, run_time_adam))
    print('RMSE u {}= {}'.format(grid_res, error_test_adam[0]))
    print('RMSE v {}= {}'.format(grid_res, error_test_adam[1]))


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

    error_test_pso = torch.sqrt(torch.mean((u_exact_test - net(grid_test.reshape(-1,1))) ** 2 , dim=0))

    loss_pso = model.solution_cls.evaluate()[0].detach().cpu().numpy()

    #########

    

    exp_dict={'grid_res': grid_res,
                        'error_train_u_adam': error_train_adam[0].item(),
                        'error_train_v_adam': error_train_adam[1].item(),
                        'error_test_u_adam': error_train_adam[0].item(),
                        'error_test_v_adam': error_train_adam[1].item(),
                        'error_train_u_pso': error_train_pso[0].item(),
                        'error_train_v_pso': error_train_pso[1].item(),
                        'error_test_u_pso': error_train_pso[0].item(),
                        'error_test_v_pso': error_train_pso[1].item(),
                        'loss_adam': loss_adam.item(),
                        'loss_pso': loss_pso.item(),
                        'time_adam': run_time_adam,
                        'time_pso': run_time_pso,
                        'type': exp_name}

    print('Time taken pso {}= {}'.format(grid_res, run_time_pso))
    print('RMSE u {}= {}'.format(grid_res, error_test_pso[0]))
    print('RMSE v {}= {}'.format(grid_res, error_test_pso[1]))

    exp_dict_list.append(exp_dict)

    return exp_dict_list







if __name__ == '__main__':

    results_dir=os.path.join(os.path.abspath(os.path.join(os.path.dirname( __file__ ))),'results')

    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    
    nruns = 1


    exp_dict_list=[]

   

    for grid_res in range(10, 101, 10):
        for _ in range(nruns):
            exp_dict_list.append(experiment_data_amount_LV_PSO(grid_res))
            exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
            df = pd.DataFrame(exp_dict_list_flatten)
            df_path=os.path.join(results_dir,'LV_PSO_KAN_{}.csv'.format(grid_res))
            df.to_csv(df_path)



