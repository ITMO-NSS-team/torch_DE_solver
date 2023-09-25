# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:33:44 2021

@author: user
"""
import torch
import torchtext
import SALib
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
import sys
import time
from scipy.special import legendre


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.input_preprocessing import Equation
from tedeous.solver import Solver

from tedeous.device import solver_device


solver_device('cpu')

"""
Preparing grid

Grid is an essentially torch.Tensor of a n-D points where n is the problem
dimensionality
"""

t = np.linspace(0, 1, 100)

coord_list = [t]

coord_list=torch.tensor(coord_list)
grid=coord_list.reshape(-1,1).float()


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
    bnd1 = torch.from_numpy(np.array([[0]], dtype=np.float64)).float()
    
    
    #  So u(0)=-1/2
    bndval1 = legendre(n)(bnd1)
    
    # point t=1
    bnd2 = torch.from_numpy(np.array([[1]], dtype=np.float64)).float()
    
    # d/dt
    bop2 = {
        '1*du/dt**1':
            {
                'coeff': 1,
                'du/dt': [0],
                'pow': 1,
                'var':0
            }
    }
    
    # So, du/dt |_{x=1}=3
    bndval2 = torch.from_numpy(legendre(n).deriv(1)(bnd2))
    
    # Putting all bconds together
    bconds = [[bnd1, bndval1, 'dirichlet'],
              [bnd2, bop2, bndval2, 'operator']]
    
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
    
    
      # this one is to show that coefficients may be a function of grid as well
    legendre_poly= {
        '(1-t^2)*d2u/dt2**1':
            {
                'coeff': c1, #coefficient is a function
                'du/dt': [0, 0],
                'pow': 1,
                'var':0
            },
        '-2t*du/dt**1':
            {
                'coeff': c2,
                'u*du/dx': [0],
                'pow':1,
                'var':0
            },
        'n*(n-1)*u**1':
            {
                'coeff': n*(n+1),
                'u':  [None],
                'pow': 1,
                'var':0
            }
    }
      
    
    
    # operator is  (1-t^2)*d2u/dt2-2t*du/dt+n*(n-1)*u=0 (n=3)
    legendre_poly= {
        '(1-t^2)*d2u/dt2**1':
            {
                'coeff': c1(grid), #coefficient is a torch.Tensor
                'du/dt': [0, 0],
                'pow': 1,
                'var':0
            },
        '-2t*du/dt**1':
            {
                'coeff': c2(grid),
                'u*du/dx': [0],
                'pow':1,
                'var':0
            },
        'n*(n-1)*u**1':
            {
                'coeff': n*(n+1),
                'u':  [None],
                'pow': 1,
                'var':0
            }
    }
    

    
    
    for _ in range(10):
        model = torch.nn.Sequential(
            torch.nn.Linear(1, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 1),
            #torch.nn.Tanh()
        )
    
        start = time.time()

        equation = Equation(grid, legendre_poly, bconds).set_strategy('NN')

        img_dir=os.path.join(os.path.dirname( __file__ ), 'leg_img')


        model = Solver(grid, equation, model, 'NN').solve(lambda_bound=10, verbose=True, learning_rate=1e-3,
                                        eps=1e-5, tmin=1000, tmax=1e5,use_cache=True,cache_verbose=True
                                        ,save_always=False,print_every=None,model_randomize_parameter=1e-6,step_plot_print=False,step_plot_save=True,image_save_dir=img_dir)
        end = time.time()
    
        print('Time taken {} = {}'.format(n,  end - start))
    
        #fig = plt.figure()
        #plt.scatter(grid.reshape(-1), model(grid).detach().numpy().reshape(-1))
        # analytical sln is 1/2*(-1 + 3*t**2)
        #plt.scatter(grid.reshape(-1), legendre(n)(grid).reshape(-1))
        #plt.show()
        
        error_rmse=torch.sqrt(torch.mean((legendre(n)(grid)-model(grid))**2))
        print('RMSE {}= {}'.format(n, error_rmse))
        
        exp_dict_list.append({'grid_res':100,'time':end - start,'RMSE':error_rmse.detach().numpy(),'type':'L'+str(n),'cache':str(CACHE)})

# import pandas as pd
# df=pd.DataFrame(exp_dict_list)
# df.boxplot(by='type',column='RMSE',figsize=(20,10),fontsize=42,showfliers=False)
# df.boxplot(by='type',column='time',figsize=(20,10),fontsize=42,showfliers=False)
# df.to_csv('benchmarking_data/legendre_poly_exp_cache='+str(CACHE)+'.csv')

#full paper plot

# import seaborn as sns

# sns.set(rc={'figure.figsize':(11.7,8.27)},font_scale=2)


# df1=pd.read_csv('benchmarking_data/legendre_poly_exp_cache=False.csv',index_col=0)
# df2=pd.read_csv('benchmarking_data/legendre_poly_exp_cache=True.csv',index_col=0)
# df=pd.concat((df1,df2))

# sns.boxplot(x='type', y='RMSE', data=df, showfliers=False, hue='cache')

# plt.figure()

# sns.boxplot(x='type', y='time', data=df, showfliers=False, hue='cache')
