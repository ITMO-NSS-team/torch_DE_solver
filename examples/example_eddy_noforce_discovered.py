# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:33:44 2021

@author: user
"""
import torch
import numpy as np
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


solver_device('gpu')

exp_dict_list=[]



data=np.load('C:\\Users\\user\\Documents\\GitHub\\epde_test_playground\\data_eddy_noforce\\data_eddy_noforce_q.npy')
#time=np.load('C:\\Users\\user\\Documents\\GitHub\\epde_test_playground\\data_eddy_noforce\\t_eddy_noforce_q.npy')
x=np.load('C:\\Users\\user\\Documents\\GitHub\\epde_test_playground\\data_eddy_noforce\\x_eddy_noforce_q.npy')
y=np.load('C:\\Users\\user\\Documents\\GitHub\\epde_test_playground\\data_eddy_noforce\\y_eddy_noforce_q.npy')
data=data[30,0,:,:]
#print(data.shape)

x=(x-np.min(x))/np.max(x-np.min(x))

y=(y-np.min(y))/np.max(y-np.min(y))

#data=(data-np.min(data))/np.max(data-np.min(data))

data=(data-np.mean(data))/np.sqrt(np.var(data-np.mean(data)))

data=(data-np.min(data))/np.max(data-np.min(data))

x=torch.from_numpy(x)
y=torch.from_numpy(y)
data=torch.from_numpy(data)


    
"""
Preparing grid

Grid is an essentially torch.Tensor of a n-D points where n is the problem
dimensionality
"""
domain = Domain()
domain.variable('x', x, None)
domain.variable('y', y, None)


    


boundaries = Conditions()
    


boundaries.dirichlet({'x': torch.Tensor([x[0]]), 'y': y}, value=data[0,:])
boundaries.dirichlet({'x': torch.Tensor([x[-1]]), 'y': y}, value=data[-1,:])
boundaries.dirichlet({'x': x, 'y': torch.Tensor([y[0]])}, value=data[:,0])
boundaries.dirichlet({'x': x, 'y': torch.Tensor([y[-1]])}, value=data[:,-1])


equation = Equation()

# operator is  1.5324278029487333e-05 * d^2u/dy^2{power: 1.0} 
# +  0.00403585166336794 * du/dx{power: 1.0} 
# + 0.61295452113519 * du/dy{power: 1.0} 
# + 1.5547072242712984e-07 * d^3u/dy^3{power: 1.0} 
# + -0.0017373670045932412 
# - du/dy{power: 1.0} * u{power: 1.0} 
# =0
eq1 = {
    'd^2u/dy^2':
        {
            'coeff': 1.5324278029487333e-05,
            'd^2u/dy^2': [1,1],
            'pow': 1,
            'var': 0
        },
    'du/dx':
        {
            'coeff': 0.00403585166336794,
            'du/dx': [0],
            'pow': 1,
            'var': 0
        },
    'du/dy':
        {
            'coeff': 0.61295452113519,
            'du/dy': [1],
            'pow': 1,
            'var': 0
        },
    'd3u/dy3**1':
        {
            'coeff': 1.5547072242712984e-07,
            'd3u/dx3': [1, 1, 1],
            'pow': 1,
            'var':0
        },
    'du/dy*u**1':
        {
            'coeff': -1,
            'u*du/dy': [[None], [1]],
            'pow': [1,1],
            'var':[0,0]
        },
    '-C':
        {
            'coeff': -0.0017373670045932412,
            'u': [None],
            'pow': 0,
            'var':0
        }
}

equation.add(eq1)

"""
Solving equation
"""


        
        
net = torch.nn.Sequential(
        torch.nn.Linear(2, 500),
        torch.nn.Tanh(),
        torch.nn.Linear(500, 500),
        torch.nn.Tanh(),
        torch.nn.Linear(500, 500),
        torch.nn.Tanh(),
        torch.nn.Linear(500, 1)
    )
    
start = time.time()
        
model = Model(net, domain, equation, boundaries)
        
model.compile('NN', lambda_operator=1, lambda_bound=100, h=0.01)

img_dir = os.path.join(os.path.dirname( __file__ ), 'eddy_noforce')

cb_lambda = adaptive_lambda.AdaptiveLambda()

cb_cache = cache.Cache(cache_verbose=True, model_randomize_parameter=1e-6)

cb_es = early_stopping.EarlyStopping(eps=1e-5,
                                    loss_window=100,
                                    no_improvement_patience=1000,
                                    patience=5,
                                    randomize_parameter=1e-6)
        
cb_plots = plot.Plots(save_every=100, print_every=None, img_dir=img_dir)

optimizer = Optimizer('Adam', {'lr': 1e-4})

model.train(optimizer, 1e5, save_model=True, callbacks=[cb_es, cb_cache, cb_plots,cb_lambda])

end = time.time()



var_list=[torch.tensor(x),torch.tensor(y)]

grid=torch.cartesian_prod(*var_list)

device = torch.device('cuda')

grid.to(device)

import matplotlib.pyplot as plt
from matplotlib import cm


fig = plt.figure(figsize=(15, 8))

X,Y = np.meshgrid(x,y)

Z=model(grid).reshape(256,256).detach().cpu().numpy()

plt.pcolormesh(X,Y,Z)
plt.colorbar()
plt.show()

fig = plt.figure(figsize=(15, 8))

X,Y = np.meshgrid(x,y)

Z=np.abs(data.reshape(256,256).detach().cpu().numpy())

plt.pcolormesh(X,Y,Z)
plt.colorbar()
plt.show()


fig = plt.figure(figsize=(15, 8))

X,Y = np.meshgrid(x,y)

Z=np.abs(model(grid).reshape(256,256).detach().cpu().numpy()-data.reshape(256,256).detach().cpu().numpy())

plt.pcolormesh(X,Y,Z)
plt.colorbar()
plt.show()