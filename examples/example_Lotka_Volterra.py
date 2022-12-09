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
import scipy
from scipy import integrate

import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from input_preprocessing import Equation
from solver import Solver

import time


alpha = 20.
beta = 20.
delta = 20.
gamma = 20.
x0 = 4.
y0 = 2.
t0 = 0.
tmax = 1.
Nt = 301

device = torch.device('cpu')

t = torch.from_numpy(np.linspace(t0, tmax, Nt))

grid = t.reshape(-1, 1).float()

grid.to(device)

h = 0.0001

#initial conditions

bnd1_0 = torch.from_numpy(np.array([[0]], dtype=np.float64)).float()
bndval1_0 = torch.from_numpy(np.array([[x0]], dtype=np.float64))
bnd1_1 = torch.from_numpy(np.array([[0]], dtype=np.float64)).float()
bndval1_1  = torch.from_numpy(np.array([[y0]], dtype=np.float64))

bconds = [[bnd1_0, bndval1_0, 0],
          [bnd1_1, bndval1_1, 1]]

#equation system
# eq1: dx/dt = x(alpha-beta*y)
# eq2: dy/dt = y(-delta+gamma*x)

# x var: 0
# y var:1

eq1 = {
    'dx/dt':{
        'coef': 1,
        'term': [0],
        'power': 1,
        'var': [0]
    },
    '-x*alpha':{
        'coef': -alpha,
        'term': [None],
        'power': 1,
        'var': [0]
    },
    '+beta*x*y':{
        'coef': beta,
        'term': [[None], [None]],
        'power': [1, 1],
        'var': [0, 1]
    }
}

eq2 = {
    'dy/dt':{
        'coef': 1,
        'term': [0],
        'power': 1,
        'var': [1]
    },
    '+y*delta':{
        'coef': delta,
        'term': [None],
        'power': 1,
        'var': [1]
    },
    '-gamma*x*y':{
        'coef': -gamma,
        'term': [[None], [None]],
        'power': [1, 1],
        'var': [0, 1]
    }
}

Lotka = [eq1, eq2]

model = torch.nn.Sequential(
        torch.nn.Linear(1, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 2)
    )

equation = Equation(grid, Lotka, bconds, h=h).set_strategy('NN')

start = time.time()

model = Solver(grid, equation, model, 'NN').solve(lambda_bound=100,
                                         verbose=True, learning_rate=1e-4, eps=1e-6, tmin=1000, tmax=5e6,
                                         use_cache=False,cache_dir='../cache/',cache_verbose=True,
                                         save_always=True,print_every=None,
                                         patience=5,loss_oscillation_window=100,no_improvement_patience=1000,
                                         model_randomize_parameter=1e-5,optimizer_mode='Adam',cache_model=None)

end = time.time()
    
print('Time taken = {}'.format(end - start))


# scipy.integrate solution of Lotka_Volterra equations and comparison with NN results

def deriv(X, t, alpha, beta, delta, gamma):
    x, y = X
    dotx = x * (alpha - beta * y)
    doty = y * (-delta + gamma * x)
    return np.array([dotx, doty])

t = np.linspace(0.,tmax, Nt)

X0 = [x0, y0]
res = integrate.odeint(deriv, X0, t, args = (alpha, beta, delta, gamma))
x, y = res.T

plt.figure()
plt.grid()
plt.title("odeint and NN methods comparing")
plt.plot(t, x, '+', label = 'preys_odeint')
plt.plot(t, y, '*', label = "predators_odeint")
plt.plot(grid, model(grid)[:,0].detach().numpy().reshape(-1), label='preys_NN')
plt.plot(grid, model(grid)[:,1].detach().numpy().reshape(-1), label='predators_NN')
plt.xlabel('Time t, [days]')
plt.ylabel('Population')
plt.legend(loc='upper right')
plt.show()

plt.figure()
plt.grid()
plt.title('Phase plane: prey vs predators')
plt.plot(model(grid)[:,0].detach().numpy().reshape(-1), model(grid)[:,1].detach().numpy().reshape(-1), '-*', label='NN')
plt.plot(x,y, label='odeint')
plt.xlabel('preys')
plt.ylabel('predators')
plt.legend()
plt.show()
