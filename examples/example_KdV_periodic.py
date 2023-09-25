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
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import scipy
import time
import os
import sys



os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
sys.path.append('../')


from tedeous.input_preprocessing import Equation
from tedeous.solver import Solver
from tedeous.solution import Solution
from tedeous.device import solver_device,check_device


solver_device('gpu')

exp_dict_list=[]


def soliton(x,t):
    E=np.exp(1)
    s=((18*np.exp((1/125)*(t + 25*x))*(16*np.exp(2*t) + 
       1000*np.exp((126*t)/125 + (4*x)/5) + 9*np.exp(2*x) + 576*np.exp(t + x) + 
       90*np.exp((124*t)/125 + (6*x)/5)))/(5*(40*np.exp((126*t)/125) + 
        18*np.exp(t + x/5) + 9*np.exp((6*x)/5) + 45*np.exp(t/125 + x))**2))
    return s

def soliton_x(x,t):
    E=np.exp(1)
    s1=-(18*E**((1/125)*(t + 25*x))*(-640*E**((376*t)/125) + 
     288*E**(3*t + x/5) - 200000*E**((252*t)/125 + (4*x)/5) + 
     81*E**((16*x)/5) - 185760*E**((251*t)/125 + x) - 
     65088*E**(2*t + (6*x)/5) - 8100*E**((249*t)/125 + (7*x)/5) + 
     225000*E**((127*t)/125 + (9*x)/5) + 
         162720*E**((126*t)/125 + 2*x) + 41796*E**(t + (11*x)/5) - 
     405*E**(t/125 + 3*x) + 
     4050*E**((4/125)*(31*t + 75*x))))/(25*(40*E**((126*t)/125) + 
      18*E**(t + x/5) + 9*E**((6*x)/5) + 45*E**(t/125 + x))**3)
    return s1

for grid_res in [30,50,100]:
    
    """
    Preparing grid

    Grid is an essentially torch.Tensor of a n-D points where n is the problem
    dimensionality
    """

    
    x = torch.from_numpy(np.linspace(-10, 10, grid_res+1))
    t = torch.from_numpy(np.linspace(0, 1, grid_res+1))
    
    grid = torch.cartesian_prod(x, t).float()
    
    
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
       
    """
    Periodic conditions at x=x+20
    """
    
    # u(0,t) = u(1,t)
    bnd3_left = torch.cartesian_prod(torch.from_numpy(np.array([-10], dtype=np.float64)),t).float()
    bnd3_right = torch.cartesian_prod(torch.from_numpy(np.array([10], dtype=np.float64)),t).float()
    bnd3 = [bnd3_left,bnd3_right]
    
    """
    Initial conditions at t=0
    """
    
    bnd4 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()
    
    # No operator applied,i.e. u(x,0)=0
    
    # equal to zero
    bndval4 = soliton(x,0)
    
    
    # Putting all bconds together
    bconds = [[bnd3, 'periodic'],
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
            }
    }
    
    
    
    """
    Solving equation
    """
    for _ in range(10):
        #sln=np.genfromtxt(os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'wolfram_sln/KdV_sln_'+str(grid_res)+'.csv')),delimiter=',')
        sln_torch=torch.Tensor([soliton(point[0],point[1]) for point in grid]).detach().cpu()
        sln_torch1=sln_torch.reshape(-1,1)
        
        #fig = plt.figure()
        #ax = fig.add_subplot(projection='3d')

        #ax.plot_trisurf(grid[:, 0],grid[:, 1],
        #                     sln_torch,
        #                    cmap=cm.jet, linewidth=0.2, alpha=1)
        #plt.show()

        model = torch.nn.Sequential(
            torch.nn.Linear(2, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 1)
        )
    
        start = time.time()
        
        equation = Equation(grid, kdv, bconds, h=0.01).set_strategy('NN')

        img_dir=os.path.join(os.path.dirname( __file__ ), 'kdv_periodic_img')

        if not(os.path.isdir(img_dir)):
            os.mkdir(img_dir)



        model = Solver(grid, equation, model, 'NN').solve(lambda_bound=100,verbose=1, learning_rate=1e-4,
                                                    eps=1e-6, tmin=1000, tmax=1e5,use_cache=True,cache_verbose=True,
                                                    save_always=True,print_every=None,model_randomize_parameter=1e-6,
                                                    optimizer_mode='Adam',no_improvement_patience=1000,step_plot_print=False,step_plot_save=True,image_save_dir=img_dir)

        
        end = time.time()
        

        error_rmse=torch.sqrt(torch.mean((sln_torch1-model(grid))**2))
    
        
        _, end_loss = Solution(grid=grid, equal_cls=equation, model=model,
             mode='NN', weak_form=None, lambda_bound=100,lambda_operator=1).evaluate()
    
        exp_dict_list.append({'grid_res':grid_res,'time':end - start,'RMSE':error_rmse.detach().cpu().numpy(),'loss':end_loss.detach().cpu().numpy(),'type':'kdv_eqn_solitary','cache':True})
        
        print('Time taken {}= {}'.format(grid_res, end - start))
        print('RMSE {}= {}'.format(grid_res, error_rmse))
        print('loss {}= {}'.format(grid_res, end_loss))


CACHE=True

import pandas as pd

result_assessment=pd.DataFrame(exp_dict_list)

result_assessment.boxplot(by='grid_res',column='time',showfliers=False,figsize=(20,10),fontsize=42)

result_assessment.boxplot(by='grid_res',column='RMSE',figsize=(20,10),fontsize=42)

result_assessment.to_csv('examples/benchmarking_data/kdv_solitary_experiment_30_100_cache={}.csv'.format(str(CACHE)))

