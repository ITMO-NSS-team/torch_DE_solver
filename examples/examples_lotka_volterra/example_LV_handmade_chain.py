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
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)


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





alpha = 20.
beta = 20.
delta = 20.
gamma = 20.
x0 = 4.
y0 = 2.
t0 = 0.
tmax = 1.

def exact(grid):
    """
    Calculates the solution of the Lotka-Volterra equations using `scipy.integrate.odeint`.
    
        This method provides a baseline for comparison, solving the equations
        numerically to generate a reference solution on the given time grid.
        This allows us to evaluate the accuracy and efficiency of neural network
        approximations developed within the project.
    
        Args:
            grid (torch.Tensor): The time grid on which to solve the equations.
    
        Returns:
            np.ndarray: A NumPy array containing the x and y values of the
            solution at each time point in the grid. The array has shape
            (len(grid), 2), where the first column represents x and the
            second column represents y.
    """
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
    """
    Solves a given differential equation by finding the exact solution and converting it into a tensor.
    
    This function is used to obtain a ground truth solution for comparison and validation of neural network-based solvers.
    
    Args:
        grid: The spatial or temporal grid on which the differential equation is defined.
    
    Returns:
        torch.Tensor: The exact solution of the differential equation, represented as a PyTorch tensor.
    """
    solution=exact(grid)
    return torch.tensor(solution)


def u_net(net, x):
    """
    Applies a neural network to an input and detaches the result for CPU-based differential equation solving.
    
        This method ensures that the neural network and input are processed on the CPU,
        applies the network to the input, and detaches the result from the computation graph.
        This is done to ensure compatibility and efficient computation when solving differential equations,
        especially when GPU resources are limited or unnecessary.
    
        Args:
            net: The neural network to apply.
            x: The input to the neural network.
    
        Returns:
            The output of the neural network, detached from the computation graph.
    """
    net = net.to('cpu')
    x = x.to('cpu')
    return net(x).detach()


def l2_norm(net, x):
    """
    Calculates the L2 norm to quantify the discrepancy between the neural network's predicted solution and the analytical solution for key physical quantities: pressure, velocity, and density. This provides a measure of the solution accuracy.
    
        Args:
            net (torch.nn.Module): The trained neural network model used to approximate the solution.
            x (torch.Tensor): The input data representing the spatial or temporal coordinates at which the solution is evaluated.
    
        Returns:
            tuple (np.ndarray, np.ndarray, np.ndarray): A tuple containing the L2 norm for pressure, velocity, and density, respectively. These norms indicate the accuracy of the neural network's approximation for each quantity.
    """
    x = x.to('cpu')
    net = net.to('cpu')
    predict = net(x).detach().cpu().reshape(-1)
    exact = u(x).detach().cpu().reshape(-1)


    l2_norm_pressure = torch.sqrt(sum((predict[:, 0]-exact[:, 0])**2))
    l2_norm_velocity = torch.sqrt(sum((predict[:, 1]-exact[:, 1])**2))
    l2_norm_density = torch.sqrt(sum((predict[:, 2]-exact[:, 2])**2))

    return l2_norm_pressure.detach().cpu().numpy(),l2_norm_velocity.detach().cpu().numpy(),l2_norm_density.detach().cpu().numpy()

def l2_norm_mat(net, x):
    """
    Calculates the L2 norm between the neural network's prediction and the exact solution of the differential equation. This metric quantifies the accuracy of the neural network's approximation.
    
        Args:
            net (torch.nn.Module): The neural network model used to approximate the solution.
            x (torch.Tensor): The input tensor representing the spatial or temporal coordinates at which to evaluate the solution.
    
        Returns:
            numpy.ndarray: The L2 norm between the network's prediction and the exact solution, as a NumPy array. It represents the overall error in the approximation.
    """
    x = x.to('cpu')
    net = net.to('cpu')
    predict = net.detach().cpu().reshape(-1)
    exact = u(x).detach().cpu().reshape(-1)
    l2_norm = torch.sqrt(sum((predict-exact)**2))
    return l2_norm.detach().cpu().numpy()

def l2_norm_fourier(net, x):
    """
    Computes the L2 norm between the network's prediction and the exact solution in Fourier space.
    
    This metric quantifies the accuracy of the neural network's solution
    by comparing it against the true solution in the frequency domain.
    A lower L2 norm indicates a better approximation of the solution.
    
    Args:
        net (torch.nn.Module): The neural network model.
        x (torch.Tensor): The input data tensor.
    
    Returns:
        np.ndarray: The L2 norm between the prediction and the exact solution as a NumPy array.
    """
    x = x.to(torch.device('cuda:0'))
    predict = net(x).detach().cpu().reshape(-1)
    exact = u(x).detach().cpu().reshape(-1)
    l2_norm = torch.sqrt(sum((predict-exact)**2))
    return l2_norm.detach().cpu().numpy()




def LV_problem_formulation(grid_res):
    """
    Sets up the Lotka-Volterra problem for neural network-based solution.
    
    This method defines the problem's domain, initial conditions, and governing
    equations, which are then used to construct the computational grid. This
    setup is crucial for training a neural network to approximate the solution
    of the Lotka-Volterra system. By formulating the problem in this way, we
    prepare the data and structure needed for the neural network to learn the
    underlying dynamics of the system.
    
    Args:
        grid_res (int): The resolution of the time grid.
    
    Returns:
        tuple: A tuple containing the grid, domain, equation, and boundaries.
            - grid (torch.Tensor): The computational grid.
            - domain (Domain): The domain of the problem.
            - equation (Equation): The equation system.
            - boundaries (Conditions): The boundary conditions.
    """
    
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





def experiment_data_amount_LV_adam_lbfgs_nncg(grid_res,exp_name='LV_adam_lbfgs_nncg'):
    """
    Performs an experiment to evaluate the performance of different optimization algorithms when solving the Lotka-Volterra equations with neural networks.
    
        This experiment trains a neural network model to approximate the solution of the Lotka-Volterra equations using Adam, LBFGS, and NNCG optimizers.
        It assesses each optimizer's effectiveness by measuring training time and accuracy (RMSE) on training and test datasets.
        This helps to understand how different optimization strategies affect the neural network's ability to learn the underlying dynamics of the differential equation.
    
        Args:
            grid_res: The resolution of the grid used for training.
            exp_name: The name of the experiment. Defaults to 'LV_adam_lbfgs_nncg'.
    
        Returns:
            list: A list containing a dictionary with the results of the experiment, including training and testing errors, losses, and training times for each optimizer.
    """
    solver_device('cuda')
    exp_dict_list = []

    grid,domain,equation,boundaries = LV_problem_formulation(grid_res)

    net = torch.nn.Sequential(
            torch.nn.Linear(1, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 2)
        )

    model = Model(net, domain, equation, boundaries)

    model.compile("autograd", lambda_operator=1, lambda_bound=100)


    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                        loss_window=100,
                                        no_improvement_patience=500,
                                        patience=3,
                                        randomize_parameter=1e-5,
                                        verbose = False,
                                        info_string_every=500)

    optim = Optimizer('Adam', {'lr': 1e-3})

    start=time.time()
    model.train(optim, 1000, callbacks=[cb_es])
    #model.train(optim, 10, callbacks=[cb_es])
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
                                        verbose = False,
                                        info_string_every=100)

    optim = Optimizer('LBFGS', {'history_size': 100,
                                "line_search_fn": 'strong_wolfe'})

    start = time.time()
    model.train(optim, 1000, save_model=False, callbacks=[cb_es])
    end = time.time()
    time_LBFGS = end - start


    error_train_LBFGS = torch.sqrt(torch.mean((u_exact_train - net(grid))** 2, dim=0))

    error_test_LBFGS = torch.sqrt(torch.mean((u_exact_test - net(grid_test.reshape(-1,1))) ** 2 , dim=0))

    loss_LBFGS = model.solution_cls.evaluate()[0].detach().cpu().numpy()

    print('Time taken LBFGS {}= {}'.format(grid_res, time_LBFGS))
    print('RMSE u {}= {}'.format(grid_res, error_test_LBFGS[0]))
    print('RMSE v {}= {}'.format(grid_res, error_test_LBFGS[1]))

    ########

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                        loss_window=100,
                                        no_improvement_patience=100,
                                        patience=2,
                                        randomize_parameter=1e-5,
                                        verbose = False,
                                        info_string_every=1)

    optim = Optimizer('NNCG', {'mu': 1e-1,
                               'lr': 1,
                               "rank": 10,
                               'line_search_fn': "armijo",
                               "precond_update_frequency": 20,
                               "eigencdecomp_shift_attepmt_count":10,
                               #'cg_max_iters':1000,
                               "chunk_size": 8,
                               'verbose': False})

    start = time.time()
    model.train(optim, 100, save_model=False, callbacks=[cb_es])
    end = time.time()

    time_NNCG = end - start

    error_train_NNCG = torch.sqrt(torch.mean((u_exact_train - net(grid))** 2, dim=0))

    error_test_NNCG = torch.sqrt(torch.mean((u_exact_test - net(grid_test.reshape(-1,1))) ** 2 , dim=0))

    loss_NNCG = model.solution_cls.evaluate()[0].detach().cpu().numpy()

    #########

    exp_dict={'grid_res': grid_res,
                        'error_train_u_adam': error_train_adam[0].item(),
                        'error_train_v_adam': error_train_adam[1].item(),
                        'error_test_u_adam': error_test_adam[0].item(),
                        'error_test_v_adam': error_test_adam[1].item(),
                        'error_train_u_LBFGS': error_train_LBFGS[0].item(),
                        'error_train_v_LBFGS': error_train_LBFGS[1].item(),
                        'error_test_u_LBFGS': error_test_LBFGS[0].item(),
                        'error_test_v_LBFGS': error_test_LBFGS[1].item(),
                        'error_train_u_NNCG': error_train_NNCG[0].item(),
                        'error_train_v_NNCG': error_train_NNCG[1].item(),
                        'error_test_u_NNCG': error_test_NNCG[0].item(),
                        'error_test_v_NNCG': error_test_NNCG[1].item(),
                        'loss_adam': loss_adam.item(),
                        'loss_LBFGS': loss_LBFGS.item(),
                        'loss_NNCG': loss_NNCG.item(),
                        'time_adam': run_time_adam,
                        'time_LBFGS': time_LBFGS,
                        'time_NNCG': time_NNCG,
                        'type': exp_name}

    print('Time taken NNCG {}= {}'.format(grid_res, time_NNCG))
    print('RMSE u {}= {}'.format(grid_res, error_test_NNCG[0]))
    print('RMSE v {}= {}'.format(grid_res, error_test_NNCG[1]))

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
            exp_dict_list.append(experiment_data_amount_LV_adam_lbfgs_nncg(grid_res))
            exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
            df = pd.DataFrame(exp_dict_list_flatten)
            df_path=os.path.join(results_dir,'LV_adam_lbfgs_nncg_{}.csv'.format(grid_res))
            df.to_csv(df_path)



