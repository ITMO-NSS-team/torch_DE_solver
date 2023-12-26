import torch
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

#sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model

domain = Domain()

domain.variable('x', [0,1], 3)
domain.variable('t', [1,2], 5)

# print(grid)
boundaries = Conditions()

bop= {
        'du/dx':
            {
                'coeff': 1,
                'du/dx': [0],
                'pow': 1,
                'var': 0
            }
}

boundaries.operator({'x': 0, 't': [1, 2]}, operator=bop, value=5)
boundaries.periodic([{'x':0, 't':[0,2]}, {'x':1, 't':[1,2]}], bop)

equation = Equation()

wave_eq = {
    '-C*d2u/dx2**1':
        {
            'coeff': -4,
            'd2u/dx2': [1, 1],
            'pow': 1
        },
    'd2u/dt2**1':
        {
            'coeff': 1,
            'd2u/dt2': [0, 0],
            'pow':1
        }
}

equation.add(wave_eq)

net = torch.nn.Sequential(
    torch.nn.Linear(2, 256),
    torch.nn.Tanh(),
    torch.nn.Linear(256, 256),
    torch.nn.Tanh(),
    torch.nn.Linear(256, 256),
    torch.nn.Tanh(),
    torch.nn.Linear(256, 256),
    torch.nn.Tanh(),
    torch.nn.Linear(256, 1))

model = Model(net, domain, equation, boundaries)

equal_cls = model.compile(mode='autograd')

print(equal_cls.bnd_prepare())