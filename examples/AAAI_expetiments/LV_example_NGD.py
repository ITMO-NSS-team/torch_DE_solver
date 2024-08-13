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
from tedeous.callbacks import cache, early_stopping, plot
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


def experiment_data_amount_LV_NGD(grid_res,NGD_info_string=True,exp_name='LV_NGD',save_plot=True,loss_window=20,randomize_parameter=1e-6):

    _r= create_random_fn(randomize_parameter)

    exp_dict_list = []

    
    l_op = 1
    l_bound = 100
    grid_steps = torch.linspace(0, 30, 31)
    steps = 0.5**grid_steps

    grid,domain,equation,boundaries = LV_problem_formulation(grid_res)
    
    net = torch.nn.Sequential(
        torch.nn.Linear(1, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 2)
    )

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=l_op, lambda_bound=l_bound)


    ls_update = grid_line_search_factory(model.solution_cls.evaluate, steps)

    loss, _ = model.solution_cls.evaluate()


    start = time.time()

    iteration=0

    min_loss=loss.item()

    last_loss = np.zeros(loss_window) + float(min_loss)

    patience=0

    while True:
        
        loss, _ = model.solution_cls.evaluate()
        grads = torch.autograd.grad(loss, model.net.parameters(), retain_graph=True, allow_unused=True)
        grads = replace_none_with_zero(grads)
        f_grads = parameters_to_vector(grads)

        int_res = model.solution_cls.operator._pde_compute()

        int_res = int_res.reshape(-1)

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

        
        G = G.detach().cpu().numpy()
        f_grads =f_grads.detach().cpu().numpy()

        f_nat_grad=np.linalg.lstsq(G, f_grads,rcond=None)[0] 

        f_nat_grad=torch.from_numpy(f_nat_grad)

        f_nat_grad = f_nat_grad.to("cuda:0")

        

        # one step of NGD
        actual_step = ls_update(model.net, f_nat_grad)
        iteration+=1

        cur_loss=model.solution_cls.evaluate()[0].item()
        
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

    end = time.time()

    run_time = end - start

    grid = domain.build('autograd')

    grid_test = torch.linspace(0, tmax, 100)

    u_exact_train = u(grid.cpu().reshape(-1))

    u_exact_test = u(grid_test.cpu().reshape(-1))

    error_train = torch.sqrt(torch.mean((u_exact_train - net(grid))** 2, dim=0))

    error_test = torch.sqrt(torch.mean((u_exact_test - net(grid_test.reshape(-1,1))) ** 2 , dim=0))

    loss = model.solution_cls.evaluate()[0].detach().cpu().numpy()


    exp_dict={'grid_res': grid_res,
                        'error_train_u': error_train[0].item(),
                        'error_train_v': error_train[1].item(),
                        'error_test_u': error_train[0].item(),
                        'error_test_v': error_train[1].item(),
                        'loss': loss.item(),
                        'time': run_time,
                        'type': exp_name}

    print('Time taken NGD {}= {}'.format(grid_res, run_time))
    print('RMSE u {}= {}'.format(grid_res, error_test[0]))
    print('RMSE v {}= {}'.format(grid_res, error_test[1]))

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
            exp_dict_list.append(experiment_data_amount_LV_NGD(grid_res,NGD_info_string=True))
            exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
            df = pd.DataFrame(exp_dict_list_flatten)
            df_path=os.path.join(results_dir,'LV_NGD_{}.csv'.format(grid_res))
            df.to_csv(df_path)
