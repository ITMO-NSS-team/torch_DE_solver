# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 11:11:30 2022

@author: user
"""

from metrics import derivative
from solver import grid_format_prepare
import numpy as np
import matplotlib.pyplot as plt
import torch


for npts in range(10,110,10):
    t = np.linspace(0, 2*np.pi, npts)
    coord_list = [t]
    
    
    
    grid=grid_format_prepare(coord_list,mode='mat')
    
    u=np.sin(grid)
        
    d1=derivative(u, grid, 0,scheme_order=2,boundary_order=2)
    plt.figure()
    plt.scatter(grid,d1)
    
    
    # plt.scatter(grid,(n)*grid**(n-1))
    
    err=torch.mean((d1-np.cos(grid))**2)
    print('npts={} err={}'.format(npts,err))