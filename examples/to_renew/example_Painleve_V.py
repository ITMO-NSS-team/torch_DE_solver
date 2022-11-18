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


def chebgrid(a,b,n):
    k=np.arange(n)+1
    k1=2*k-1
    n1=2*n
    cos_vec=np.cos(k1/n1*np.pi)
    grid=(a+b)/2+(b-a)/2*cos_vec
    grid=np.flip(grid)
    return grid

# t = torch.from_numpy(chebgrid(1/4, 7/4,500).copy())



CACHE=True

device = torch.device('cuda')


def p_V_exp(grid_res):
    
    exp_dict_list=[]
    
    t = torch.from_numpy(np.linspace(0.9, 1.2, grid_res+1))



    grid = t.reshape(-1, 1).float()

    grid.to(device)


    # point t=0
    bnd1 = torch.from_numpy(np.array([[0.9]], dtype=np.float64))

    bop1 = None

    #  So u(0)=-1/2
    bndval1 = torch.from_numpy(np.array([[2.979325765542575]], dtype=np.float64))

    # point t=0
    bnd2 = torch.from_numpy(np.array([[1.2]], dtype=np.float64))

    bop2 = None

    #  So u(0)=-1/2
    bndval2 = torch.from_numpy(np.array([[4.117945909429169]], dtype=np.float64))




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



    # t^2
    def p5_c1(grid):
        return grid**2

    # 2*(a+3b+t(g-td))
    def p5_c2(grid):
        return 2*(a+3*b+grid*(g-d*grid))

    # 2*(3a+3+t(g-td))
    def p5_c3(grid):
        return -2*(3*a+b+grid*(g+d*grid))

    # t
    def p5_c4(grid):
        return grid

    # P_V operator is  u*d2u/dt2-1/2 (du/dt)^2-b-2*(t^2-a)*u^2-4t*u^3-3/2*u^4
    p_5= {
        '-2*t^2*u*d2u/dt2**1':
            {
                'coeff': -2*p5_c1(grid),
                'du/dt': [[None],[0, 0]],
                'pow': [1,1]
            },
        '2*t^2*u**2*d2u/dt2**1':
            {
                'coeff': 2*p5_c1(grid),
                'du/dt': [[None],[0, 0]],
                'pow': [2,1]
            },
        '2*b':
            {
                'coeff': 2*b,
                'u':  [None],
                'pow': 0
            },
        '-6*b*u':
            {
                'coeff': -6*b,
                'u':  [None],
                'pow': 1
            },
        '2*(a+3b+t(g-td))*u**2':
            {
                'coeff': p5_c2(grid),
                'u':  [None],
                'pow': 2
            },
        '-2*(3a+b+t(g+td))*u**3':
            {
                'coeff': p5_c3(grid),
                'u':  [None],
                'pow': 3
            },
        '6*a*u**4':
            {
                'coeff': 6*a,
                'u':  [None],
                'pow': 4
            },
        '-2*a*u**5':
            {
                'coeff': -2*a,
                'u':  [None],
                'pow': 5
            },
        '-2*t*u*du/dt':
            {
                'coeff': -2*p5_c4(grid),
                'du/dt': [[None],[0]],
                'pow': [1,1]
            }, 
        '2*t*u**2*du/dt':
            {
                'coeff': 2*p5_c4(grid),
                'du/dt': [[None],[0]],
                'pow': [2,1]
            }, 
        't**2*du/dt**2':
            {
                'coeff': p5_c1(grid),
                'du/dt': [0],
                'pow': 2
            }, 
        '-3*t**2*u*du/dt**2':
            {
                'coeff': -3*p5_c1(grid),
                'du/dt': [[None],[0]],
                'pow': [1,2]
            }
    }



        
    sln=np.genfromtxt(os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'wolfram_sln/P_V_sln_'+str(grid_res)+'.csv')),delimiter=',')
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
    
    model = point_sort_shift_solver(grid, model, p_5, bconds, lambda_bound=100, verbose=2,h=abs((t[1]-t[0]).item()), learning_rate=1e-4,
                                    eps=1e-7, tmin=1000, tmax=1e6,use_cache=True,cache_dir='../cache/',cache_verbose=True
                                    ,batch_size=None, save_always=True,lp_par=lp_par,no_improvement_patience=10000,print_every=None)
    end = time.time()
        
    error_rmse=torch.sqrt(torch.mean((sln_torch1-model(grid))**2))
    
  
    
    prepared_grid,grid_dict,point_type = grid_prepare(grid)
    
    prepared_bconds = bnd_prepare(bconds, prepared_grid,grid_dict, h=abs((t[1]-t[0]).item()))
    prepared_operator = operator_prepare(p_5, grid_dict, subset=['central'], true_grid=grid, h=abs((t[1]-t[0]).item()))
    end_loss = point_sort_shift_loss(model, prepared_grid, prepared_operator, prepared_bconds, lambda_bound=100)
    exp_dict_list.append({'grid_res':grid_res,'time':end - start,'RMSE':error_rmse.detach().numpy(),'loss':end_loss.detach().numpy(),'type':'PV'})
    
    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))
    print('loss {}= {}'.format(grid_res, end_loss))
    return exp_dict_list




nruns=1

exp_dict_list=[]



for grid_res in range(100,501,100):
    for _ in range(nruns):
        exp_dict_list.append(p_V_exp(grid_res))
        
        
import pandas as pd

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df=pd.DataFrame(exp_dict_list_flatten)
df.boxplot(by='grid_res',column='time',fontsize=42,figsize=(20,10))
df.boxplot(by='grid_res',column='RMSE',fontsize=42,figsize=(20,10),showfliers=False)
df.to_csv('PV_experiment_100_500_cache={}.csv'.format(str(CACHE)))

