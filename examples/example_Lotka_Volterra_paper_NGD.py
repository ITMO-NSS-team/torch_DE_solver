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
from tedeous.callbacks import cache, early_stopping, plot, adaptive_lambda
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device, check_device, device_type
from tedeous.models import mat_model, Fourier_embedding,FourierNN

alpha = 20.
beta = 20.
delta = 20.
gamma = 20.
x0 = 4.
y0 = 2.
t0 = 0.
tmax = 1.


from copy import deepcopy


def train_net(net,grid,exact):

    exact=torch.Tensor(exact).float()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    t0=torch.Tensor([0.])
    x0 = 4.
    y0 = 2.

    loss = torch.mean(torch.square(net(grid) - exact))+100*torch.mean(torch.square(net(t0)-torch.Tensor([x0,y0])))

    def closure():
        optimizer.zero_grad()
        loss = torch.mean(torch.square(net(grid) - exact))+100*torch.mean(torch.square(net(t0)-torch.Tensor([x0,y0])))
        loss.backward()
        return loss

    t = 0
    while loss > 1e-5 and t < 1e5:
        optimizer.step(closure)
        loss = torch.mean(torch.square(net(grid) - exact))+100*torch.mean(torch.square(net(t0)-torch.Tensor([x0,y0])))
        t += 1
        if t %1000 ==0:
            print('Interpolate from exact t={}, loss={}'.format(t, loss))

    return net


# Define the model
class MultiOutputModel(torch.nn.Module):
    def __init__(self):
        super(MultiOutputModel, self).__init__()
        
        self.width_out=[2]

        # Shared layers (base network)
        self.shared_fc1 = torch.nn.Linear(1, 64)  # Input size of 1 (for t)
        self.shared_fc2 = torch.nn.Linear(64, 32)
        # Output head for Process 1
        self.process1_fc = torch.nn.Linear(32, 1)
        
        # Output head for Process 2
        self.process2_fc = torch.nn.Linear(32, 1)
    
    def forward(self, t):
        # Shared layers forward pass
        x = torch.tanh(self.shared_fc1(t))
        x = torch.tanh(self.shared_fc2(x))
        # Process 1 output head
        process1_out = self.process1_fc(x)
        
        # Process 2 output head
        process2_out = self.process2_fc(x)
        
        out=torch.cat((process1_out, process2_out), dim=1)

        return out

# Initialize the model
#model = 


def Lotka_experiment(grid_res, CACHE):

    exp_dict_list = []
    solver_device('gpu')

    FFL = Fourier_embedding(L=[1/4], M=[4])

    out = FFL.out_features


    net = torch.nn.Sequential(
        FFL,
        torch.nn.Linear(out, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 2)
    )

    #net=MultiOutputModel()

    #net = FourierNN([512, 512, 512, 512, 2], [15], [7])

    #net = torch.nn.Sequential(
    #    torch.nn.Linear(1, 32),
    #    torch.nn.Tanh(),
    #    torch.nn.Linear(32, 32),
    #    torch.nn.Tanh(),
    #    torch.nn.Linear(32, 2)
    #)
    
    #def weights_init(m):
    #    if isinstance(m, torch.nn.Linear):
    #        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
    #        #torch.nn.init.zero_(m.bias)
    
    #net.apply(weights_init)


    def exact():
        # scipy.integrate solution of Lotka_Volterra equations and comparison with NN results

        def deriv(X, t, alpha, beta, delta, gamma):
            x, y = X
            dotx = x * (alpha - beta * y)
            doty = y * (-delta + gamma * x)
            return np.array([dotx, doty])

        t = np.linspace(0, tmax, grid_res+1)

        X0 = [x0, y0]
        res = integrate.odeint(deriv, X0, t, args = (alpha, beta, delta, gamma))
        x, y = res.T
        return np.hstack((x.reshape(-1,1),y.reshape(-1,1)))

    u_exact = exact()

    #net=train_net(net,torch.from_numpy(np.linspace(0, 1, grid_res+1)).reshape(-1,1).float(),u_exact)


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



    model =  Model(net, domain, equation, boundaries)

    model.compile("autograd", lambda_operator=1, lambda_bound=100)
    
    img_dir=os.path.join(os.path.dirname( __file__ ), 'img_Lotka_Volterra_paper_NGD')

    start = time.time()

    cb_es = early_stopping.EarlyStopping(eps=5e-6,
                                    loss_window=1000,
                                    no_improvement_patience=1000,
                                    patience=5,
                                    info_string_every=100,
                                    randomize_parameter=1e-5,
                                    save_best=True)
    
    cb_plots = plot.Plots(save_every=1000, print_every=None, img_dir=img_dir)

    cb_lambda = adaptive_lambda.AdaptiveLambda()

    #cb_cache = cache.Cache(cache_verbose=True, model_randomize_parameter=1e-5)

    optimizer = Optimizer('Adam', {'lr': 1e-4})

    model.train(optimizer, 5e5, callbacks=[cb_es, cb_plots])

    #model =  Model(net, domain, equation, boundaries)

    model.compile("autograd", lambda_operator=1, lambda_bound=100)


    optimizer = Optimizer('NGD', {'grid_steps_number': 20})

    cb_es = early_stopping.EarlyStopping(eps=5e-6,
                                loss_window=1000,
                                no_improvement_patience=1000,
                                patience=5,
                                info_string_every=100,
                                randomize_parameter=1e-5,
                                save_best=True)

    model.train(optimizer, 2e3, callbacks=[cb_es, cb_plots])

    end = time.time()
    
    grid = domain.build('NN')   


    u_exact = torch.from_numpy(u_exact)

    prediction=net(grid)

    prediction_np=prediction.cpu().detach().numpy()
    u_exact_np=u_exact.cpu().detach().numpy()


    error_rmse = np.sqrt(np.mean((prediction_np-u_exact_np)**2))

    exp_dict_list.append({'grid_res':grid_res,'time':end - start,'RMSE':error_rmse,'type':'Lotka_eqn','cache':CACHE})
    
    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))

    #t = domain.variable_dict['t']
    grid = domain.build('NN')

    t = np.linspace(0, 1, grid_res+1)

    plt.figure()
    plt.grid()
    plt.title("odeint and NN methods comparing")
    plt.plot(t, u_exact[:,0].detach().numpy().reshape(-1), '+', label = 'preys_odeint')
    plt.plot(t, u_exact[:,1].detach().numpy().reshape(-1), '*', label = "predators_odeint")
    plt.plot(grid.cpu(), net(check_device(grid))[:,0].cpu().detach().numpy().reshape(-1), label='preys_NN')
    plt.plot(grid.cpu(), net(check_device(grid))[:,1].cpu().detach().numpy().reshape(-1), label='predators_NN')
    plt.xlabel('Time t, [days]')
    plt.ylabel('Population')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(img_dir,'compare_{}.png'.format(grid_res)))


    return exp_dict_list

nruns=1

exp_dict_list=[]

CACHE=False

for grid_res in range(500,1001,100):
    for _ in range(nruns):
        exp_dict_list.append(Lotka_experiment(grid_res,CACHE))
   

        
#import pandas as pd

#exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
#df=pd.DataFrame(exp_dict_list_flatten)
#df.boxplot(by='grid_res',column='time',fontsize=42,figsize=(20,10))
#df.boxplot(by='grid_res',column='RMSE',fontsize=42,figsize=(20,10),showfliers=False)
#df.to_csv('benchmarking_data/Lotka_experiment_50_90_cache={}.csv'.format(str(CACHE)))