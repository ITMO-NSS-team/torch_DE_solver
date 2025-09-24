import torch
import numpy as np
import os
import sys


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device

solver_device('gpu')

m = 0.2
L = 1
Q = -0.1
Sq = 1
mu_water = 0.89e-3
mu_o = 4.62e-3
Swi0 = 0.
Sk = 1.
t_end = 1.

domain = Domain()

domain.variable('x', [0, 1], 21, dtype='float32')
domain.variable('t', [0, 1], 21, dtype='float32')

boundaries = Conditions()

##initial cond
boundaries.dirichlet({'x': [0, 1], 't': 0}, value=Swi0)

##boundary cond
boundaries.dirichlet({'x': 0, 't': [0, 1]}, value=Sk)

net = torch.nn.Sequential(
    torch.nn.Linear(2, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 1)
)

def k_oil(x):
    """
    Calculates the squared difference between the neural network's output and the ideal value of 1.
    
        This function penalizes deviations of the network's output from 1, encouraging the network to learn solutions that approach this target value. This is useful in the context of solving differential equations, where the network's output might represent an approximation of the solution, and we want to guide it towards a specific behavior.
    
        Args:
            x (torch.Tensor): The input tensor to the neural network.
    
        Returns:
            torch.Tensor: The squared difference between 1 and the neural network's output.
    """
    return (1-net(x))**2

def k_water(x):
    """
    Computes the squared output of the neural network for a given input.
    
    This function is used to calculate a component of the loss function, specifically focusing on the squared difference between the neural network's prediction and the desired behavior derived from the differential equation. Squaring the network's output can emphasize larger deviations from zero, potentially guiding the training process to find solutions that more closely satisfy the equation.
    
    Args:
        x (torch.Tensor): The input value, representing the independent variable of the differential equation, to be passed through the neural network.
    
    Returns:
        torch.Tensor: The square of the neural network's output for the given input 'x'. This represents a component of the loss, penalizing deviations from zero.
    """
    return (net(x))**2

def dk_water(x):
    """
    Calculates a scaled representation of the differential equation's solution.
    
    This function scales the output of the neural network, which approximates the solution
    to a differential equation. The scaling factor of 2 is applied to refine the solution
    obtained from the network.
    
    Args:
        x: The input value, representing the independent variable of the differential equation.
    
    Returns:
        The scaled output of the neural network (2 * net(x)), representing a refined
        approximation of the differential equation's solution at the given input.
    """
    return 2*net(x)

def dk_oil(x):
    """
    Calculates a value based on the input 'x' and a neural network.
    
        This method computes a value by applying a neural network 'net' to the input 'x',
        subtracting the result from 1, multiplying by -2, and returning the final value.
        This transformation is applied to scale and shift the output of the neural network,
        preparing it for use in solving differential equations. The scaling ensures that the
        network's output contributes appropriately to the overall solution.
    
        Args:
            x (torch.Tensor): The input value, typically representing a point in the domain of the differential equation, to be processed by the neural network.
    
        Returns:
            torch.Tensor: The calculated value, representing a modified output of the neural network, ready for use in the differential equation solver.
    """
    return -2*(1-net(x))

def df(x):
    """
    Calculates the derivative of the fractional flow function with respect to the saturation.
    
        This derivative is a key component in determining the stability and behavior
        of solutions when solving differential equations that model two-phase flow
        using neural networks. It leverages derivatives and values of permeability
        functions for water and oil, along with the viscosity ratio of water to oil,
        to provide insights into the flow characteristics.
    
        Args:
            x (float): The water saturation at which to evaluate the derivative.
    
        Returns:
            float: The value of the derivative of the fractional flow function at the given saturation x.
    """
    return (dk_water(x)*(k_water(x)+mu_water/mu_o*k_oil(x))-
            k_water(x)*(dk_water(x)+mu_water/mu_o*dk_oil(x)))/(k_water(x)+mu_water/mu_o*k_oil(x))**2

def coef_model(x):
    """
    Calculates the coefficient based on the derivative of the function `f` at a given point.
    
    This coefficient is used to adjust the optimization process, guiding the neural network towards a solution that satisfies the differential equation.
    
    Args:
        x (float): The input value at which the derivative is evaluated.
    
    Returns:
        float: The calculated coefficient value. It represents the scaled derivative of `f` at `x`.
    """
    return -Q/Sq*df(x)

equation = Equation()

buckley_eq = {
    'm*ds/dt**1':
        {
            'coeff': m,
            'ds/dt': [1],
            'pow': 1
        },
    '-Q/Sq*df*ds/dx**1':
        {
            'coeff': coef_model,
            'ds/dx': [0],
            'pow':1
        }
}

equation.add(buckley_eq)

model = Model(net, domain, equation, boundaries)

model.compile('autograd', lambda_operator=1, lambda_bound=1)

img_dir=os.path.join(os.path.dirname( __file__ ), 'Buckley_NN_img')


cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                     loss_window=100,
                                     no_improvement_patience=500,
                                     patience=5,
                                     abs_loss=1e-5,
                                     randomize_parameter=1e-5,
                                     info_string_every=500)

cb_plots = plot.Plots(save_every=500, print_every=None, img_dir=img_dir)

optimizer = Optimizer('Adam', {'lr': 1e-3})

model.train(optimizer, 1e6, save_model=False, callbacks=[cb_es, cb_plots])
                                    
                                     
                                    