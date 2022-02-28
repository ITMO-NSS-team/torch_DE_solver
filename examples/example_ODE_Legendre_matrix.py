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
from solver_matrix import *
import time
from scipy.special import legendre
device = torch.device('cpu')

"""
Preparing grid

Grid is an essentially torch.Tensor of a n-D points where n is the problem
dimensionality
"""

t = np.linspace(0, 1, 100)

grid = [t]
grid = np.meshgrid(*grid)
grid = torch.tensor(grid, device=device)


exp_dict_list=[]

CACHE=True

for n in range(3,11):  
    """
    Preparing boundary conditions (BC)
    
    For every boundary we define three items
    
    bnd=torch.Tensor of a boundary n-D points where n is the problem
    dimensionality
    
    bop=dict in form {'term1':term1,'term2':term2}-> term1+term2+...=0
    
    NB! dictionary keys at the current time serve only for user-frienly 
    description/comments and are not used in model directly thus order of
    items must be preserved as (coeff,op,pow)
    
    term is a dict term={coefficient:c1,[sterm1,sterm2],'pow': power}
    
    Meaning c1*u*d2u/dx2 has the form
    
    {'coefficient':c1,
     'u*d2u/dx2': [[None],[0,0]],
     'pow':[1,1]}
    
    None is for function without derivatives
    
    
    bval=torch.Tensor prescribed values at every point in the boundary
    """
    
    # point t=0
    bnd1 = torch.from_numpy(np.array([[0]], dtype=np.float64))
    
    bop1 = None
    
    #  So u(0)=-1/2
    bndval1 = legendre(n)(bnd1)
    
    # point t=1
    bnd2 = torch.from_numpy(np.array([[1]], dtype=np.float64))
    
    # d/dt
    bop2 = {
        '1*du/dt**1':
            {
                'coefficient': 1,
                'du/dt': [0],
                'pow': 1
            }
    }
    
    # So, du/dt |_{x=1}=3
    bndval2 = torch.from_numpy(legendre(n).deriv(1)(bnd2))
    
    # Putting all bconds together
    bconds = [[bnd1, bop1, bndval1], [bnd2, bop2, bndval2]]
    
    """
    Defining Legendre polynomials generating equations
    
    Operator has the form
    
    op=dict in form {'term1':term1,'term2':term2}-> term1+term2+...=0
    
    NB! dictionary keys at the current time serve only for user-frienly 
    description/comments and are not used in model directly thus order of
    items must be preserved as (coeff,op,pow)
    
    
    
    term is a dict term={coefficient:c1,[sterm1,sterm2],'pow': power}
    
    c1 may be integer, function of grid or tensor of dimension of grid
    
    Meaning c1*u*d2u/dx2 has the form
    
    {'coefficient':c1,
     'u*d2u/dx2': [[None],[0,0]],
     'pow':[1,1]}
    
    None is for function without derivatives
    
    
    """
    
    
    # 1-t^2
    def c1(grid):
        return 1 - grid ** 2
    
    
    # -2t
    def c2(grid):
        return -2 * grid
    
    
    
    
    
    # operator is  (1-t^2)*d2u/dt2-2t*du/dt+n*(n-1)*u=0 (n=3)
    legendre_poly= {
        '(1-t^2)*d2u/dt2**1':
            {
                'coeff': c1(grid), #coefficient is a torch.Tensor
                'du/dt': [0, 0],
                'pow': 1
            },
        '-2t*du/dt**1':
            {
                'coeff': c2(grid),
                'u*du/dx': [0],
                'pow':1
            },
        'n*(n-1)*u**1':
            {
                'coeff': n*(n+1),
                'u':  [None],
                'pow': 1
            }
    }
    
    # this one is to show that coefficients may be a function of grid as well
    legendre_poly= {
        '(1-t^2)*d2u/dt2**1':
            {
                'coeff': c1, #coefficient is a function
                'du/dt': [0, 0],
                'pow': 1
            },
        '-2t*du/dt**1':
            {
                'coeff': c2,
                'u*du/dx': [0],
                'pow':1
            },
        'n*(n-1)*u**1':
            {
                'coeff': n*(n+1),
                'u':  [None],
                'pow': 1
            }
    }
    
    
    
    for _ in range(10):
        model = torch.rand(grid.shape)
    
        start = time.time()
        model = lbfgs_solution(model, grid, legendre_poly, 100, bconds, tol=1e-6,nsteps=10000)
        end = time.time()
    
        print('Time taken {} = {}'.format(n,  end - start))
    
        fig = plt.figure()
        plt.scatter(grid.reshape(-1), model.detach().numpy().reshape(-1))
        # analytical sln is 1/2*(-1 + 3*t**2)
        plt.scatter(grid.reshape(-1), legendre(n)(grid).reshape(-1))
        plt.show()
        
        error_rmse=torch.sqrt(torch.mean((legendre(n)(grid)-model)**2))
        print('RMSE {}= {}'.format(n, error_rmse))
        
        exp_dict_list.append({'grid_res':100,'time':end - start,'RMSE':error_rmse.detach().numpy(),'type':'L'+str(n),'cache':str(CACHE)})

import pandas as pd
df=pd.DataFrame(exp_dict_list)
df.boxplot(by='type',column='RMSE',figsize=(20,10),fontsize=42,showfliers=False)
df.boxplot(by='type',column='time',figsize=(20,10),fontsize=42,showfliers=False)
df.to_csv('benchmarking_data/legendre_poly_exp_martix.csv')

#full paper plot

# import seaborn as sns

# sns.set(rc={'figure.figsize':(11.7,8.27)},font_scale=2)


# df1=pd.read_csv('benchmarking_data/legendre_poly_exp_cache=False.csv',index_col=0)
# df2=pd.read_csv('benchmarking_data/legendre_poly_exp_cache=True.csv',index_col=0)
# df=pd.concat((df1,df2))

# sns.boxplot(x='type', y='RMSE', data=df, showfliers=False, hue='cache')

# plt.figure()

# sns.boxplot(x='type', y='time', data=df, showfliers=False, hue='cache')
