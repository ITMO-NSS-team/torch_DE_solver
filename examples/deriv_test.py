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
from solver import lbfgs_solution,matrix_optimizer,grid_format_prepare
import time
from scipy.special import legendre
from solver import matrix_cache_lookup
device = torch.device('cpu')
from cache import save_model
from solver import optimization_solver
from config import read_config
from metrics import derivative
import warnings
"""
Preparing grid

Grid is an essentially torch.Tensor of a n-D points where n is the problem
dimensionality
"""



t = np.linspace(10, 89, 80)
coord_list = [t]



grid=grid_format_prepare(coord_list,mode='mat')

exp_dict_list=[]

model=torch.sin(grid)

plt.plot(model.detach().numpy().reshape(-1))

# derivative(u_tensor, h_tensor, axis, scheme_order=1, boundary_order=1)

d1=derivative(model,grid,0)

# d1=np.gradient(model.reshape(-1),grid.reshape(-1),edge_order=2)

d1_anal=torch.cos(grid)

plt.figure()
plt.plot(t,d1.detach().numpy().reshape(-1))
plt.plot(t,d1_anal.detach().numpy().reshape(-1))

d2=derivative(d1,grid,0)

# d2=PolyDiff(d1,grid,window_width=3)


d2_anal=-torch.sin(grid)

plt.figure()
plt.plot(t,d2.detach().numpy().reshape(-1))
plt.plot(t,d2_anal.detach().numpy().reshape(-1))