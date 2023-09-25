# Lotka-Volterra equations also known as predator-prey equations, describe the variation in populations
# of two species which interact via predation.
# For example, wolves (predators) and deer (prey). This is a classical model to represent the dynamic of two populations.

# Let αlpha > 0, beta > 0, delta > 0 and gamma > 0 . The system is given by

# dx/dt = x(alpha-beta*y)
# dy/dt = y(-delta+gamma*x)

# Where 'x' represents prey population and 'y' predators population. It’s a system of first-order ordinary differential equations.
import torch
import torchtext
import SALib
import numpy as np
import matplotlib.pyplot as plt
import fontTools
from scipy import integrate
import time
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.input_preprocessing import Equation
from tedeous.solver import Solver, grid_format_prepare
from tedeous.solution import Solution
from tedeous.device import solver_device
from tedeous.models import mat_model


alpha = 20.
beta = 20.
delta = 20.
gamma = 20.
x0 = 4.
y0 = 2.
t0 = 0.
tmax = 2.

Nt = 401

solver_device('cpu')

t = torch.from_numpy(np.linspace(t0, tmax, Nt))

grid = t.reshape(-1, 1).float()

h = abs((grid[1]-grid[0]).item())
#initial conditions

coord_list = [t]

grid = grid_format_prepare(coord_list,mode='mat')

bnd1_0 = torch.from_numpy(np.array([[0]], dtype=np.float64)).float()
bndval1_0 = torch.from_numpy(np.array([[x0]], dtype=np.float64))
bnd1_1 = torch.from_numpy(np.array([[0]], dtype=np.float64)).float()
bndval1_1  = torch.from_numpy(np.array([[y0]], dtype=np.float64))

bconds = [[bnd1_0, bndval1_0, 0, 'dirichlet'],
            [bnd1_1, bndval1_1, 1, 'dirichlet']]

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

Lotka = [eq1, eq2]

equation = Equation(grid, Lotka, bconds, h=h).set_strategy('mat')

model = mat_model(grid, Lotka)

img_dir = os.path.join(os.path.dirname( __file__ ), 'img_Lotka_Volterra_mat')

start = time.time()

model = Solver(grid, equation, model, 'mat').solve(lambda_bound=100, derivative_points=3, gamma=0.9, lr_decay=400,
                                        verbose=True, learning_rate=1,
                                        print_every=100, patience=3,
                                        optimizer_mode='LBFGS',
                                        step_plot_save=True, image_save_dir=img_dir)

end = time.time()

def exact():
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
    return np.array([x.reshape(-1), y.reshape(-1)])

u_exact = exact()

u_exact=torch.from_numpy(u_exact)

error_rmse=torch.sqrt(torch.mean((u_exact-model)**2))

plt.figure()
plt.grid()
plt.plot(t, u_exact[0].detach().numpy().reshape(-1), '+', label = 'x_odeint')
plt.plot(t, u_exact[1].detach().numpy().reshape(-1), '*', label = "y_odeint")
plt.plot(t, model[0].detach().numpy().reshape(-1), label='x_tedeous')
plt.plot(t, model[1].detach().numpy().reshape(-1), label='y_tedeous')
plt.xlabel('Time, t')
plt.ylabel('Population')
plt.legend(loc='upper right')
plt.show()

