# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:33:44 2021

@author: user
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys

# sys.path.append('../')
sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.solver import *
import time




device = torch.device('cpu')


def p_IV_exp(grid_res,CACHE):
    
    exp_dict_list=[]
    
    t = torch.from_numpy(np.linspace(1/4, 7/4, grid_res+1))
    
    grid = t.reshape(-1, 1).float()
    
    grid.to(device)
    
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
    bnd1 = torch.from_numpy(np.array([[1]], dtype=np.float64))
    
    bop1 = None
    
    #  So u(0)=-1/2
    bndval1 = torch.from_numpy(np.array([[1]], dtype=np.float64))
    
    # point t=1
    bnd2 = torch.from_numpy(np.array([[1]], dtype=np.float64))
    
    # d/dt
    bop2 ={
            '1*du/dt**1':
                {
                    'coeff': 1,
                    'du/dt': [0],
                    'pow': 1
                    }
                
        }
        
        
    
    # So, du/dt |_{x=1}=3
    bndval2 = torch.from_numpy(np.array([[0]], dtype=np.float64))
    
    
    # # point t=0
    # bnd1 = torch.from_numpy(np.array([[t[0]]], dtype=np.float64))
    
    # bop1 = None
    
    # #  So u(0)=-1/2
    # bndval1 = torch.from_numpy(np.array([[4.63678]], dtype=np.float64))
    
    # # point t=1
    # bnd2 = torch.from_numpy(np.array([[t[-1]]], dtype=np.float64))
    
    # # d/dt
    # bop2 =None
        
    
    # # So, du/dt |_{x=1}=3
    # bndval2 = torch.from_numpy(np.array([[167.894]], dtype=np.float64))
    
    
    
    # Putting all bconds together
    bconds = [[bnd1, bop1, bndval1], [bnd2, bop2, bndval2]]
    
    
    a=1
    b=1
    g=1
    d=1
    
    # t
    def p4_c1(grid):
        return grid
    
    
    # t^2-a
    def p4_c2(grid):
        return grid**2-a
    
    
    # P_IV operator is  u*d2u/dt2-1/2 (du/dt)^2-b-2*(t^2-a)*u^2-4t*u^3-3/2*u^4
    p_4= {
        'u*d2u/dt2**1':
            {
                'coeff': 1,
                'du/dt': [[None],[0, 0]],
                'pow': [1,1]
            },
        '-1/2 (du/dt)^2':
            {
                'coeff': -1/2,
                'u':  [0],
                'pow': 2
            },
        '-b':
            {
                'coeff': -b,
                'u':  [None],
                'pow': 0
            },
        '-2*(t^2-a)*u^2':
            {
                'coeff': -2*p4_c2(grid),
                'u':  [None],
                'pow': 2
            },
        '-4t*u^3':
            {
                'coeff': -4*p4_c1(grid),
                'u':  [None],
                'pow': 3
            },
        '-3/2*u^4':
            {
                'coeff': -3/2,
                'u':  [None],
                'pow': 4
            }
            
    }

  
        
    sln=np.genfromtxt(os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'wolfram_sln/P_IV_sln_'+str(grid_res)+'.csv')),delimiter=',')
    sln_torch=torch.from_numpy(sln)
    sln_torch1=sln_torch.reshape(-1,1)
    
            
    
    model = torch.nn.Sequential(
    torch.nn.Linear(1, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 1)
    # torch.nn.Tanh()
    )

    lp_par={'operator_p':2,
        'operator_weighted':False,
        'operator_normalized':False,
        'boundary_p':1,
        'boundary_weighted':False,
        'boundary_normalized':False}
    
    start = time.time()
    
    model = point_sort_shift_solver(grid, model, p_4, bconds, lambda_bound=100, verbose=2,h=abs((t[1]-t[0]).item()), learning_rate=1e-4,
                                    eps=1e-7, tmin=1000, tmax=1e6,use_cache=CACHE,cache_dir='../cache/',cache_verbose=True
                                    ,batch_size=None, save_always=True,lp_par=lp_par,no_improvement_patience=10000,print_every=None,
                                    model_randomize_parameter=1e-6)
    end = time.time()
        
    error_rmse=torch.sqrt(torch.mean((sln_torch1-model(grid))**2))
    
  
    
    prepared_grid,grid_dict,point_type = grid_prepare(grid)
    
    prepared_bconds = bnd_prepare(bconds, prepared_grid,grid_dict, h=(7/4-1/4)/grid_res)
    prepared_operator = operator_prepare(p_4, grid_dict, subset=['central'], true_grid=grid, h=(7/4-1/4)/grid_res)
    end_loss = point_sort_shift_loss(model, prepared_grid, prepared_operator, prepared_bconds, lambda_bound=100)
    exp_dict_list.append({'grid_res':grid_res,'time':end - start,'RMSE':error_rmse.detach().numpy(),'loss':end_loss.detach().numpy(),'type':'PIV','cache':CACHE})
    
    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))
    print('loss {}= {}'.format(grid_res, end_loss))
    return exp_dict_list




nruns=1

exp_dict_list=[]

CACHE=True

for grid_res in range(100,501,100):
    for _ in range(nruns):
        exp_dict_list.append(p_IV_exp(grid_res,CACHE))
        
        
import pandas as pd

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df=pd.DataFrame(exp_dict_list_flatten)
df.boxplot(by='grid_res',column='time',fontsize=42,figsize=(20,10))
df.boxplot(by='grid_res',column='RMSE',fontsize=42,figsize=(20,10),showfliers=False)
df.to_csv('PIV_experiment_100_500_cache={}.csv'.format(str(CACHE)))





