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
from tedeous.callbacks import early_stopping, plot, cache
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device

solver_device('cpu')

alpha = 20.
beta = 20.
delta = 20.
gamma = 20.
x0 = 4.
y0 = 2.
t0 = 0.
tmax = 1.


def Lotka_experiment(grid_res, CACHE):
    exp_dict_list = []

    domain = Domain()
    domain.variable('t', [t0, tmax], grid_res)
    t = domain.variable_dict['t']
    h = (t[1]-t[0]).item()

    boundaries = Conditions()
    #initial conditions
    boundaries.dirichlet({'t': 0}, value=x0, var=0)
    boundaries.dirichlet({'t': 0}, value=y0, var=1)

    equation = Equation()

    #equation system
    # eq1: dx/dt = x(alpha-beta*y)
    # eq2: dy/dt = y(-delta+gamma*x)

    # x var: 0
    # y var:1

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

    net = torch.nn.Sequential(
            torch.nn.Linear(1, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 2)
        )

    def v(grid):
        return (0.5+0.5*torch.sin(grid[:,0]))*(2/h)**(0.5)/10
    weak_form = [v]

    model =  Model(net, domain, equation, boundaries)

    model.compile("NN", lambda_operator=1, lambda_bound=100, h=h, weak_form=weak_form)

    cb_es = early_stopping.EarlyStopping(eps=1e-6, no_improvement_patience=500, info_string_every=500)

    cb_cache = cache.Cache(cache_verbose=True, model_randomize_parameter=1e-5)

    img_dir=os.path.join(os.path.dirname( __file__ ), 'img_weak_Lotka_Volterra_paper')

    cb_plots = plot.Plots(save_every=500, print_every=None, img_dir=img_dir)

    optimizer = Optimizer('Adam', {'lr': 1e-4})

    if CACHE:
        callbacks = [cb_es, cb_cache, cb_plots]
    else:
        callbacks = [cb_es, cb_plots]
    start = time.time()
    model.train(optimizer, 5e6, save_model=CACHE, callbacks=callbacks)

    end = time.time()
    
    rmse_t_grid=np.linspace(0,1,grid_res+1)

    rmse_t = torch.from_numpy(rmse_t_grid)

    rmse_grid = rmse_t.reshape(-1, 1).float()
    
    def exact():
        # scipy.integrate solution of Lotka_Volterra equations and comparison with NN results

        def deriv(X, t, alpha, beta, delta, gamma):
            x, y = X
            dotx = x * (alpha - beta * y)
            doty = y * (-delta + gamma * x)
            return np.array([dotx, doty])

        t = np.linspace(0.,tmax, grid_res+1)

        X0 = [x0, y0]
        res = integrate.odeint(deriv, X0, t, args = (alpha, beta, delta, gamma))
        x, y = res.T
        return np.hstack((x.reshape(-1,1),y.reshape(-1,1)))

    u_exact = exact()

    u_exact=torch.from_numpy(u_exact)

    error_rmse=torch.sqrt(torch.mean((u_exact-net(rmse_grid))**2, 0).sum())
    
    exp_dict_list.append({'grid_res':grid_res,'time':end - start,'RMSE':error_rmse.detach().numpy(),'type':'Lotka_eqn','cache':CACHE})
    
    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))

    grid = domain.build('NN').cpu()
    net = net.cpu()

    plt.figure()
    plt.grid()
    plt.title("odeint and NN methods comparing")
    plt.plot(t, u_exact[:,0].detach().numpy().reshape(-1), '+', label = 'preys_odeint')
    plt.plot(t, u_exact[:,1].detach().numpy().reshape(-1), '*', label = "predators_odeint")
    plt.plot(grid, net(grid)[:,0].detach().numpy().reshape(-1), label='preys_NN')
    plt.plot(grid, net(grid)[:,1].detach().numpy().reshape(-1), label='predators_NN')
    plt.xlabel('Time t, [days]')
    plt.ylabel('Population')
    plt.legend(loc='upper right')
    plt.show()

    return exp_dict_list

nruns=10

exp_dict_list=[]

CACHE=False

for grid_res in range(70,111,10):
    for _ in range(nruns):
        exp_dict_list.append(Lotka_experiment(grid_res,CACHE))

import pandas as pd

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df=pd.DataFrame(exp_dict_list_flatten)
#df.boxplot(by='grid_res',column='time',fontsize=42,figsize=(20,10))
#df.boxplot(by='grid_res',column='RMSE',fontsize=42,figsize=(20,10),showfliers=False)
df.to_csv('benchmarking_data/weak_Lotka_experiment_70_111_cache={}.csv'.format(str(CACHE)))