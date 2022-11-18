# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:33:44 2021

@author: user
"""

import os
import sys


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
sys.path.append('../')


from tedeous.solver import *
import time
from tedeous.config import Config



device = torch.device('cpu')

exp_dict_list=[]


for grid_res in [10]:
    
    """
    Preparing grid

    Grid is an essentially torch.Tensor of a n-D points where n is the problem
    dimensionality
    """

    
    x = torch.from_numpy(np.linspace(0, 1, grid_res+1))
    t = torch.from_numpy(np.linspace(0, 1, grid_res+1))
    
    coord_list=[x,t]
    

    
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
                'a1': a1,
                'd2u/dx2': [0, 0],
                'pow': 1,
                'var': 0
            },
        'a2*du/dx':
            {
                'a2': a2,
                'du/dx': [0],
                'pow': 1,
                'var': 0
            },
        'a3*u':
            {
                'a3': a3,
                'u': [None],
                'pow': 1,
                'var': 0
            }
    }
    
    # bop1=[[a1,[0,0],1],[a2,[0],1],[a3,[None],1]]
    
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
                'a1': b1,
                'd2u/dx2': [0, 0],
                'pow': 1,
                'var': 0 
            },
        'b2*du/dx':
            {
                'a2': b2,
                'du/dx': [0],
                'pow': 1,
                'var': 0
            },
        'b3*u':
            {
                'a3': b3,
                'u': [None],
                'pow': 1,
                'var':0
            }
    }
    
    # bop2=[[b1,[0,0],1],[b2,[0],1],[b3,[None],1]]
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
                'r1': r1,
                'du/dx': [0],
                'pow': 1,
                'var':0
            },
        'r2*u':
            {
                'r2': r2,
                'u': [None],
                'pow': 1,
                'var': 0
            }
    }
    
    # bop3=[[r1,[0],1],[r2,[None],1]]
    
    # equal to zero
    bndval3 = torch.zeros(len(bnd3))
    
    """
    Initial conditions at t=0
    """
    
    bnd4 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()
    
    # No operator applied,i.e. u(x,0)=0
    bop4 = None
    
    # equal to zero
    bndval4 = torch.zeros(len(bnd4))
    
    
    # Putting all bconds together
    bconds = [[bnd1, bop1, bndval1], [bnd2, bop2, bndval2], [bnd3, bop3, bndval3], [bnd4, bop4, bndval4]]
    
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
    grid=grid_format_prepare(coord_list, mode='NN')
    # -sin(x)cos(t) ONLY for NN grid!
    def c1(grid):
        return (-1) * torch.sin(grid[:, 0]) * torch.cos(grid[:, 1])
    
    # operator is du/dt+6u*(du/dx)+d3u/dx3-sin(x)*cos(t)=0
    kdv = {
        '1*du/dt**1':
            {
                'coeff': 1,
                'du/dt': [1],
                'pow': 1,
                'var': 0
            },
        '6*u**1*du/dx**1':
            {
                'coeff': 6,
                'u*du/dx': [[None], [0]],
                'pow': [1,1],
                'var':[0,0]
            },
        'd3u/dx3**1':
            {
                'coeff': 1,
                'd3u/dx3': [0, 0, 0],
                'pow': 1,
                'var':0
            },
        '-sin(x)cos(t)':
            {
                '-sin(x)cos(t)': c1(grid).reshape(-1,1),
                'u': [None],
                'pow': 0,
                'var':0
            }
    }
    
    
    
    """
    Solving equation
    """
    for _ in range(1):
        sln=np.genfromtxt(os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'wolfram_sln/KdV_sln_'+str(grid_res)+'.csv')),delimiter=',')
        sln_torch=torch.from_numpy(sln)
        sln_torch1=sln_torch.reshape(-1,1)
        
        
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 1)
        )
        
        config=Config()

        config.set_parameter('Optimizer.optimizer','Adam')
        config.set_parameter('Optimizer.learning_rate',1e-4)
        config.set_parameter('Optimizer.lambda_bound',100)
        config.set_parameter('StopCriterion.eps',1e-5)
        config.set_parameter('StopCriterion.no_improvement_patience',None)
        config.set_parameter('Cache.save_always',True)
        config.set_parameter('Cache.use_cache',True)
        config.set_parameter('Cache.model_randomize_parameter',1e-5)
        config.set_parameter('Verbose.verbose',True)
        config.set_parameter('NN.h',0.01)


        start = time.time()
        model=optimization_solver(coord_list, model, kdv, bconds, config,mode='mat')
        # model = point_sort_shift_solver(grid, model, kdv, bconds, lambda_bound=1000,verbose=True, learning_rate=1e-4,
        #                                 eps=1e-6, tmin=1000, tmax=1e5, h=0.01,use_cache=True,cache_verbose=True,
        #                             batch_size=64, save_always=True)
        
        end = time.time()
        
        if callable(model):
            error_rmse=torch.sqrt(torch.mean((sln_torch1-model(grid))**2))
        else:
            error_rmse=torch.sqrt(torch.mean((sln_torch1.reshape(model.shape)-model)**2))
        
        exp_dict_list.append({'grid_res':grid_res,'time':end - start,'RMSE':error_rmse.detach().numpy(),'type':'kdv_eqn','cache':True})
        
        print('Time taken {}= {}'.format(grid_res, end - start))
        print('RMSE {}= {}'.format(grid_res, error_rmse))

