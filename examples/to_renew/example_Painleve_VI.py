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


def p_VI_exp(grid_res):
    
    exp_dict_list=[]
    

    # t = torch.from_numpy(chebgrid(1/4, 7/4,500).copy())
    
    t = torch.from_numpy(np.linspace(1.2, 1.4, grid_res+1))
    
    
    
    grid = t.reshape(-1, 1).float()
    
    grid.to(device)


    # point t=0
    bnd1 = torch.from_numpy(np.array([[1.2]], dtype=np.float64))
    
    bop1 = None
    
    #  So u(0)=-1/2
    bndval1 = torch.from_numpy(np.array([[2]], dtype=np.float64))
    
    # point t=0
    bnd2 = torch.from_numpy(np.array([[1.4]], dtype=np.float64))
    
    bop2 = None
    
    #  So u(0)=-1/2
    bndval2 = torch.from_numpy(np.array([[2]], dtype=np.float64))
    
    
    
    
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
    
    
    
    p6coeff={0:-grid**3*b,
             1:2*grid**2*(1 + grid)*b,
             2:-grid*(b-d+grid*(a+(4+grid)*b+(-1+grid)*g+d)),
             3:2*grid*((1+grid)*a+(1+grid)*b+(-1+grid)*(g+d)),
             4:-(a+grid*(4+grid)*a-g+grid*(b+g+(-1+grid)*d)),
             5:2*(1+grid)*a,
             6:(-1+grid)*grid**3,
             7:-grid*(1+grid*(-3+grid+grid**2)),
             8:(-1+grid)*grid*(-1+2*grid),
             9:-(1/2)*(-1+grid)**2*grid**3,
             10:(-1+grid)**2*grid**2*(1+grid),
             11:-(3/2)*(-1+grid)**2*grid**2,
             12:(-1+grid)**2*grid**3,
             13:-(-1+grid)**2*grid**2*(1+grid),
             14:(-1+grid)**2*grid**2}
    

    # P_VI operator is  
    p_6= {
        '-t**3*b':
            {
                'coeff': p6coeff[0],
                'u':  [None],
                'pow': 0
            },
        'u':
            {
                'coeff': p6coeff[1],
                'u':  [None],
                'pow': 1
            },
        'u**2':
            {
                'coeff': p6coeff[2],
                'u':  [None],
                'pow': 2
            },
        'u**3':
            {
                'coeff': p6coeff[3],
                'u':  [None],
                'pow': 3
            },
        'u**4':
            {
                'coeff': p6coeff[4],
                'u':  [None],
                'pow': 4
            },
        'u**5':
            {
                'coeff': p6coeff[5],
                'u':  [None],
                'pow': 5
            },
        'u**6':
            {
                'coeff': -a,
                'u':  [None],
                'pow': 6
            },
        'u*du/dt':
            {
                'coeff': p6coeff[6],
                'du/dt': [[None],[0]],
                'pow': [1,1]
            }, 
        'u**2*du/dt':
            {
                'coeff': p6coeff[7],
                'du/dt': [[None],[0]],
                'pow': [2,1]
            }, 
        'u**3*du/dt':
            {
                'coeff': p6coeff[8],
                'du/dt': [[None],[0]],
                'pow': [3,1]
            }, 
        'du/dt**2':
            {
                'coeff': p6coeff[9],
                'du/dt': [0],
                'pow': 2
            },
        'u*du/dt**2':
            {
                'coeff': p6coeff[10],
                'du/dt': [[None],[0]],
                'pow': [1,2]
            }, 
        'u**2*du/dt**2':
            {
                'coeff': p6coeff[11],
                'du/dt': [[None],[0]],
                'pow': [2,2]
            }, 
        'u*d2u/dt2':
            {
                'coeff': p6coeff[12],
                'du/dt': [[None],[0,0]],
                'pow': [1,1]
            },
        'u**2*d2u/dt2':
            {
                'coeff': p6coeff[13],
                'du/dt': [[None],[0,0]],
                'pow': [2,1]
            },
        'u**3*d2u/dt2':
            {
                'coeff': p6coeff[14],
                'du/dt': [[None],[0,0]],
                'pow': [3,1]
            }
            
    }

        
    sln=np.genfromtxt(os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'wolfram_sln/P_VI_sln_'+str(grid_res)+'.csv')),delimiter=',')
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
    
    model = point_sort_shift_solver(grid, model, p_6, bconds, lambda_bound=100, verbose=2,h=abs((t[1]-t[0]).item()), learning_rate=1e-4,
                                    eps=1e-7, tmin=1000, tmax=1e6,use_cache=True,cache_dir='../cache/',cache_verbose=True
                                    ,batch_size=None, save_always=True,lp_par=lp_par,no_improvement_patience=10000,print_every=None)
    end = time.time()
        
    error_rmse=torch.sqrt(torch.mean((sln_torch1-model(grid))**2))
    
  
    
    prepared_grid,grid_dict,point_type = grid_prepare(grid)
    
    prepared_bconds = bnd_prepare(bconds, prepared_grid,grid_dict, h=abs((t[1]-t[0]).item()))
    prepared_operator = operator_prepare(p_6, grid_dict, subset=['central'], true_grid=grid, h=abs((t[1]-t[0]).item()))
    end_loss = point_sort_shift_loss(model, prepared_grid, prepared_operator, prepared_bconds, lambda_bound=100)
    exp_dict_list.append({'grid_res':grid_res,'time':end - start,'RMSE':error_rmse.detach().numpy(),'loss':end_loss.detach().numpy(),'type':'PVI'})
    
    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))
    print('loss {}= {}'.format(grid_res, end_loss))
    return exp_dict_list




nruns=1

exp_dict_list=[]



for grid_res in range(100,501,100):
    for _ in range(nruns):
        exp_dict_list.append(p_VI_exp(grid_res))
        
        
import pandas as pd

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df=pd.DataFrame(exp_dict_list_flatten)
df.boxplot(by='grid_res',column='time',fontsize=42,figsize=(20,10))
df.boxplot(by='grid_res',column='RMSE',fontsize=42,figsize=(20,10),showfliers=False)
df.to_csv('PVI_experiment_100_500_cache={}.csv'.format(str(CACHE)))


