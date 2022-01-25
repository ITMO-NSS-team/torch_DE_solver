# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 11:38:13 2022

@author: user
"""
import pickle

with open('MM_eqn.pickle', 'rb') as f:
        eqn_data= pickle.load(f)
        
        
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:33:44 2021

@author: user
"""
import torch
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys

sys.path.append('../')

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from solver import *
import time

device = torch.device('cpu')



grid = eqn_data['grid']

grid.to(device)

t = torch.from_numpy(np.linspace(0, 4*np.pi, 100))

grid = t.reshape(-1, 1).float()

grid.to(device)


# Putting all bconds together
bconds = eqn_data['bc']

# # point t=0
# bnd1 = torch.from_numpy(np.array([[4]], dtype=np.float64))

# bop1 = None

# #  So u(0)=-1/2
# bndval1 = torch.from_numpy(np.array([[-1.6065392024306238]], dtype=np.float64))


# bconds.append([bnd1,bndval1])

op=eqn_data['operator']


# for _ in range(1):
    
#     model = torch.nn.Sequential(
#         torch.nn.Linear(1, 100),
#         torch.nn.Tanh(),
#         torch.nn.Linear(100, 100),
#         torch.nn.Tanh(),
#         torch.nn.Linear(100, 1)
#     )

#     lp_par={'operator_p':2,
#     'operator_weighted':True,
#     'operator_normalized':True,
#     'boundary_p':2,
#     'boundary_weighted':True,
#     'boundary_normalized':True}

#     start = time.time()
#     model_epde = point_sort_shift_solver(grid, model, op, bconds, lambda_bound=10, verbose=True, learning_rate=1e-5,h=0.001,
#                                     eps=1e-7, tmin=1000, tmax=3e5,use_cache=True,cache_dir='../cache/',cache_verbose=True
#                                     ,batch_size=None, save_always=False,lp_par=lp_par)
#     end = time.time()

#     print('Time taken 10= ', end - start)



op[0][0]=-torch.cos(grid)
op[1][0]=torch.from_numpy(np.ones(len(grid)))
op[2][0]=-torch.sin(grid)

for _ in range(1):
    
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 1)
    )

    # lp_par={'operator_p':2,
    #     'operator_weighted':True,
    #     'operator_normalized':True,
    #     'boundary_p':1,
    #     'boundary_weighted':True,
    #     'boundary_normalized':True}
    
    test=[]

    start = time.time()
    model_true = point_sort_shift_solver(grid, model, op, bconds, lambda_bound=10, verbose=True, learning_rate=1e-5,h=float(torch.min(grid[1]-grid[0])),
                                    eps=1e-7, tmin=1000, tmax=3e5,use_cache=True,cache_dir='../cache/',cache_verbose=True
                                    ,batch_size=None, save_always=False,grid_point_subset=None,print_every=1000)
    end = time.time()

    print('Time taken 10= ', end - start)
    
    fig = plt.figure()
    # plt.scatter(grid.reshape(-1), model_epde(grid).detach().numpy().reshape(-1))
    # analytical sln is 1/2*(-1 + 3*t**2)
    plt.scatter(grid.reshape(-1), model_true(grid).detach().numpy().reshape(-1))
    plt.scatter(grid.reshape(-1), ((1.3)*torch.cos(grid)+torch.sin(grid)).reshape(-1))
    plt.show()
    
    print('RMSE=',torch.mean((model_true(grid)-((1.3)*torch.cos(grid)+torch.sin(grid)))**2))

    # lp_par={'operator_p':2,
    #     'operator_weighted':False,
    #     'operator_normalized':False,
    #     'boundary_p':1,
    #     'boundary_weighted':False,
    #     'boundary_normalized':False}
    
    # start = time.time()
    
    # model = point_sort_shift_solver(grid, model, op, bconds, lambda_bound=10, verbose=2, learning_rate=1e-3,
    #                                 eps=1e-7, tmin=1000, tmax=1e5,use_cache=False,cache_dir='../cache/',cache_verbose=True
    #                                 ,batch_size=None, save_always=True,lp_par=lp_par)
    # end = time.time()
    
    # print('Time taken = ', end - start)