# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:33:44 2021

@author: user
"""
import os
import numpy as np
import torch
import torchtext
import SALib
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
sys.path.append('../')

from tedeous.input_preprocessing import Equation
from tedeous.solver import Solver, grid_format_prepare, Plots
from tedeous.solution import Solution
from tedeous.device import solver_device
from tedeous.models import mat_model



solver_device('cpu')

exp_dict_list=[]


for grid_res in [20,30]:
    
    """
    Preparing grid

    Grid is an essentially torch.Tensor of a n-D points where n is the problem
    dimensionality
    """

    x = torch.from_numpy(np.linspace(0, 1, grid_res + 1))
    t = torch.from_numpy(np.linspace(0, 1, grid_res + 1))

    coord_list = []
    coord_list.append(x)
    coord_list.append(t)

    grid=grid_format_prepare(coord_list,mode='mat')


    #grid = np.meshgrid(*grid)
    #grid = torch.tensor(grid, device=device)
    
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
    
    # coefficients for BC
    
    a1, a2, a3 = [1, 2, 1]
    
    b1, b2, b3 = [2, 1, 3]
    
    r1, r2 = [5, 5]
    
    """
    Boundary x=0
    """
    
    # # points
    bnd1 = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), t).float()
    
    # operator a1*d2u/dx2+a2*du/dx+a3*u
    bop1 = {
        'a1*d2u/dx2':
            {
                'coeff': a1,
                'd2u/dx2': [0, 0],
                'pow': 1
            },
        'a2*du/dx':
            {
                'coeff': a2,
                'du/dx': [0],
                'pow': 1
            },
        'a3*u':
            {
                'coeff': a3,
                'u': [None],
                'pow': 1
            }
    }
    
    #bop1=[[a1,[0,0],1],[a2,[0],1],[a3,[None],1]]
    
    # equal to zero
    bndval1 = torch.zeros(len(bnd1))
    
    """
    Boundary x=1
    """
    
    # points
    bnd2 = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)), t).float()
    
    # operator b1*d2u/dx2+b2*du/dx+b3*u
    bop2 = {
        'b1*d2u/dx2':
            {
                'coeff': b1,
                'd2u/dx2': [0, 0],
                'pow': 1
            },
        'b2*du/dx':
            {
                'coeff': b2,
                'du/dx': [0],
                'pow': 1
            },
        'b3*u':
            {
                'coeff': b3,
                'u': [None],
                'pow': 1
            }
    }
    
    #bop2=[[b1,[0,0],1],[b2,[0],1],[b3,[None],1]]
    # equal to zero
    bndval2 = torch.zeros(len(bnd2))
    
    """
    Another boundary x=1
    """
    # points
    bnd3 = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)), t).float()
    
    # operator r1*du/dx+r2*u
    bop3 = {
        'r1*du/dx':
            {
                'coeff': r1,
                'du/dx': [0],
                'pow': 1
            },
        'r2*u':
            {
                'coeff': r2,
                'u': [None],
                'pow': 1
            }
    }
    
    #bop3=[[r1,[0],1],[r2,[None],1]]
    
    # equal to zero
    bndval3 = torch.zeros(len(bnd3))
    
    """
    Initial conditions at t=0
    """
    
    bnd4 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()
    
    # No operator applied,i.e. u(x,0)=0
    
    
    # equal to zero
    bndval4 = torch.zeros(len(bnd4))
    
    
    # Putting all bconds together
    bconds = [[bnd1, bop1, bndval1, 'operator'],
              [bnd2, bop2, bndval2, 'operator'],
              [bnd3, bop3, bndval3, 'operator'],
              [bnd4, bndval4, 'dirichlet']]
    
    """
    Defining kdv equation
    
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
    
    # -sin(x)cos(t)
    def c1(grid):
        return (-1) * torch.sin(grid[0]) * torch.cos(grid[1])
    
    # operator is du/dt+6u*(du/dx)+d3u/dx3-sin(x)*cos(t)=0
    kdv = {
        '1*du/dt**1':
            {
                'coeff': 1,
                'du/dt': [1],
                'pow': 1
            },
        '6*u**1*du/dx**1':
            {
                'coeff': 6,
                'u*du/dx': [[None], [0]],
                'pow': [1,1]
            },
        'd3u/dx3**1':
            {
                'coeff': 1,
                'd3u/dx3': [0, 0, 0],
                'pow': 1
            },
        '-sin(x)cos(t)':
            {
                'coeff': c1(grid),
                'u': [None],
                'pow': 0
            }
    }
    
    
    
    """
    Solving equation
    """
    for _ in range(10):

        sln=np.genfromtxt(os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'wolfram_sln/KdV_sln_'+str(grid_res)+'.csv')),delimiter=',')
        sln_torch=torch.from_numpy(sln)
    
        start = time.time()

        equation = Equation(grid, kdv, bconds).set_strategy('mat')

        model = mat_model(grid, kdv)*0

        img_dir=os.path.join(os.path.dirname( __file__ ), 'kdv_img_mat')

        if not(os.path.isdir(img_dir)):
            os.mkdir(img_dir)

        model = Solver(grid, equation, model, 'mat').solve(lambda_bound=10, derivative_points=2,
                                         verbose=True, learning_rate=0.5, eps=1e-8, tmin=1000, tmax=5e6,
                                         use_cache=True,cache_dir='../cache/',cache_verbose=True,
                                         save_always=True,print_every=100,
                                         patience=5,loss_oscillation_window=100,no_improvement_patience=1000,
                                         model_randomize_parameter=1e-5,optimizer_mode='LBFGS',cache_model=None,step_plot_print=False,step_plot_save=True,image_save_dir=img_dir)


        end = time.time()

        model1 = torch.transpose(model, 0, 1)
        error_rmse=np.sqrt(np.mean((sln_torch.cpu().numpy().reshape(-1)-model1.detach().cpu().numpy().reshape(-1))**2))

        Plots(grid, model, 'mat').solution_print()


        _, end_loss = Solution(grid=grid, equal_cls=equation, model=model,
             mode='mat', weak_form=None, lambda_bound=100,lambda_operator=1).evaluate()
    
        exp_dict_list.append({'grid_res':grid_res,'time':end - start,'RMSE':error_rmse,'loss':end_loss.detach().numpy(),'type':'kdv_eqn'})
        
        print('Time taken {}= {}'.format(grid_res, end - start))
        print('RMSE {}= {}'.format(grid_res, error_rmse))
        print('loss {}= {}'.format(grid_res, end_loss))

#result_assessment=pd.DataFrame(exp_dict_list)
#result_assessment.to_csv('results_kdv_matrix.csv')

#result_assessment.boxplot(by='grid_res',column='time',showfliers=False,figsize=(20,10),fontsize=42)

#result_assessment.boxplot(by='grid_res',column='RMSE',figsize=(20,10),fontsize=42)
