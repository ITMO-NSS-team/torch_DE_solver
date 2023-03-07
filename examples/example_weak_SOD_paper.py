"""
                               U_t + AU_x = 0,         (x,t) in (0,1)x(0,0.2]
                              (rho,u,p)_t=0 = (1.0,0.0,1.0) 0 <= x < 0.5
                              (rho,u,p)_t=0 = (0.125,0.0,0.1) 0.5 <= x <=1
with Dirichlet boundary conditions which take the values of the initial condition at the boundaries
                               U = [ rho ]       and       A =  [    u, rho, 0    ]
                                   [  u  ]                      [   0,  u, 1/rho  ]
                                   [  p  ]                      [   0, gamma*p, u ]
rho -- Density of the fluid
u   -- Velocity of the fluid - x direction
p   -- Pressure of the fluid
E   --  Total energy of fluid
We relate the pressure and energy by the equation of state of the form
                                             p = (gamma - 1) ( rho*E - 0.5*rho||u||^2)
For this problem we use gamma = 1.4

"""

import torch
import numpy as np
import os
import time
import pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.append('../')

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.input_preprocessing import Equation
from tedeous.solver import Solver
from tedeous.metrics import Solution
from tedeous.device import solver_device

solver_device('cpu')
# constannts for PDE system
p_l = 1
v_l = 0
Ro_l = 1
gam_l = 1.4

p_r = 0.1
v_r = 0
Ro_r = 0.125
gam_r = 1.4

x0 = 0.5


def SOD_experiment(grid_res, CACHE):
    exp_dict_list=[]
    
    x_grid=np.linspace(0,1,grid_res+1)
    t_grid=np.linspace(0,0.2,grid_res+1)

    h = x_grid[1]-x_grid[0]

    x = torch.from_numpy(x_grid)
    t = torch.from_numpy(t_grid)

    grid = torch.cartesian_prod(x, t).float()


    ## BOUNDARY AND INITIAL CONDITIONS
    # p:0, v:1, Ro:2

    def u0(x,x0):
        if x>x0:
            return [p_r, v_r, Ro_r]
        else:
            return [p_l, v_l, Ro_l]

    # Initial conditions at t=0
    bnd1 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()

    u_init0 = np.zeros(bnd1.shape[0], dtype=np.float64)
    u_init1 = np.zeros(bnd1.shape[0], dtype=np.float64)
    u_init2 = np.zeros(bnd1.shape[0], dtype=np.float64)
    j=0
    for i in bnd1:
        u_init0[j] = u0(i[0], x0)[0]
        u_init1[j] = u0(i[0], x0)[1]
        u_init2[j] = u0(i[0], x0)[2]
        j +=1

    bndval1_0 = torch.from_numpy(u_init0)
    bndval1_1 = torch.from_numpy(u_init1)
    bndval1_2 = torch.from_numpy(u_init2)

    #  Boundary conditions at x=0
    bnd2 = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), t).float()

    bndval2_0 = torch.from_numpy(np.asarray([p_l for i in bnd2[:, 0]], dtype=np.float64))
    bndval2_1 = torch.from_numpy(np.asarray([v_l for i in bnd2[:, 0]], dtype=np.float64))
    bndval2_2 = torch.from_numpy(np.asarray([Ro_l for i in bnd2[:, 0]], dtype=np.float64))



    # Boundary conditions at x=1
    bnd3 = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)), t).float()

    # u(1,t)=0
    bndval3_0 = torch.from_numpy(np.asarray([p_r for i in bnd3[:, 0]], dtype=np.float64))
    bndval3_1 = torch.from_numpy(np.asarray([v_r for i in bnd3[:, 0]], dtype=np.float64))
    bndval3_2 = torch.from_numpy(np.asarray([Ro_r for i in bnd3[:, 0]], dtype=np.float64))


    # Putting all bconds together
    bconds = [[bnd1, bndval1_0, 0, 'dirichlet'],
            [bnd1, bndval1_1, 1, 'dirichlet'],
            [bnd1, bndval1_2, 2, 'dirichlet'],
            [bnd2, bndval2_0, 0, 'dirichlet'],
            [bnd2, bndval2_1, 1, 'dirichlet'],
            [bnd2, bndval2_2, 2, 'dirichlet'],
            [bnd3, bndval3_0, 0, 'dirichlet'],
            [bnd3, bndval3_1, 1, 'dirichlet'],
            [bnd3, bndval3_2, 2, 'dirichlet']]


    '''
    gas dynamic system equations:
    Eiler's equations system for Sod test in shock tube

    '''
    gas_eq1={
            'dro/dt':
            {
                'coeff': 1,
                'term': [1],
                'pow': 1,
                'var': 2
            },
            'v*dro/dx':
            {
                'coeff': 1,
                'term': [[None], [0]],
                'pow': [1, 1],
                'var': [1, 2]
            },
            'ro*dv/dx':
            {
                'coeff': 1,
                'term': [[None], [0]],
                'pow': [1, 1],
                'var': [2, 1]
            }
        }
    gas_eq2 = {
            'ro*dv/dt':
            {
                'coeff': 1,
                'term': [[None], [1]],
                'pow': [1, 1],
                'var': [2, 1]
            },
            'ro*v*dv/dx':
            {
                'coeff': 1,
                'term': [[None],[None], [0]],
                'pow': [1, 1, 1],
                'var': [2, 1, 1]
            },
            'dp/dx':
            {
                'coeff': 1,
                'term': [0],
                'pow': 1,
                'var': 0
            }
        }
    gas_eq3 =  {
            'dp/dt':
            {
                'coeff': 1,
                'term': [1],
                'pow': 1,
                'var': 0
            },
            'gam*p*dv/dx':
            {
                'coeff': gam_l,
                'term': [[None], [0]],
                'pow': [1, 1],
                'var': [0, 1]
            },
            'v*dp/dx':
            {
                'coeff': 1,
                'term': [[None], [0]],
                'pow': [1, 1],
                'var': [1, 0]
            }

        }

    gas_eq = [gas_eq1, gas_eq2, gas_eq3]

    model = torch.nn.Sequential(
            torch.nn.Linear(2, 150),
            torch.nn.Tanh(),
            torch.nn.Linear(150, 150),
            torch.nn.Tanh(),
            torch.nn.Linear(150, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 3)
            )

    def v(grid):
        return torch.sin(grid[:,0])+torch.sin(2*grid[:,0])+grid[:,1]
    weak_form = [v]

    start = time.time()

    equation = Equation(grid, gas_eq, bconds, h=h).set_strategy('NN')

    img_dir=os.path.join(os.path.dirname( __file__ ), 'SOD_NN_img')

    if not(os.path.isdir(img_dir)):
        os.mkdir(img_dir)


    model = Solver(grid, equation, model, 'NN', weak_form=weak_form).solve(
                                lambda_bound=100, verbose=True, learning_rate=1e-2,
                                eps=1e-6, tmin=1000, tmax=1e5, use_cache=CACHE, cache_dir='../cache/', cache_verbose=False,
                                save_always=True, patience=2, abs_loss=0.0035, no_improvement_patience=500,print_every=None,
                                model_randomize_parameter=1e-5,step_plot_save=True, image_save_dir=img_dir)

    end = time.time()

    rmse_x_grid=np.linspace(0,1,grid_res+1)
    rmse_t_grid=np.linspace(0,0.2,grid_res+1)

    rmse_x = torch.from_numpy(rmse_x_grid)
    rmse_t = torch.from_numpy(rmse_t_grid)

    rmse_grid = torch.cartesian_prod(rmse_x, rmse_t).float()
    def exact(point):
        N = 100
        Pl = 1
        Pr = 0.1
        Rg = 519.4
        Gl = 1.4
        Gr = 1.4
        Tl = 273
        Tr = 248
        Rol = 1
        Ror = 0.125
        
        Cr = (Gr*Pr/Ror)**(1/2)
        Cl = (Gl*Pl/Rol)**(1/2)
        vl = 0
        vr = 0
        t = float(point[-1])
        x = float(point[0])
        x0 = 0
        x1 = 1
        xk = 0.5
            

        eps = 1e-5
        Pc1 = Pl/2
        vc1 = 0.2
        u = 1
        while u >= eps:
            Pc = Pc1
            vc = vc1
            f = vl + 2/(Gl-1)*Cl*(-(Pc/Pl)**((Gl-1)/(2*Gl))+1)-vc
            g = vr + (Pc-Pr)/(Ror*Cr*((Gr+1)/(2*Gr)*Pc/Pr+(Gr-1)/(2*Gr))**(1/2))-vc
            fp = -2/(Gl-1)*Cl*(1/Pl)**((Gl-1)/2/Gl)*(Gl-1)/2/Gl*Pc**((Gl-1)/(2*Gl)-1)
            gp = (1-(Pc-Pr)*(Gr+1)/(4*Gr*Pr*((Gr+1)/(2*Gr)*Pc/Pr+(Gr-1)/(2*Gr))))/(Ror*Cr*((Gr+1)/(2*Gr)*Pc/Pr+(Gr-1)/(2*Gr))**(1/2))
            fu = -1
            gu = -1
            Pc1 = Pc - (fu*g-gu*f)/(fu*gp-gu*fp)
            vc1 = vc - (f*gp-g*fp)/(fu-gp-gu*fp)
            u1 = abs((Pc-Pc1)/Pc)
            u2 = abs((vc-vc1)/vc)
            u = max(u1, u2)

        Pc = Pc1
        vc = vc1

        if x <= xk - Cl*t:
            p = Pl
            v = vl
            T = Tl
            Ro = Rol
        Roc = Rol/(Pl/Pc)**(1/Gl)
        if xk - Cl*t < x <= xk + (vc-(Gl*Pc/Roc)**(1/2))*t:
            Ca = (vl + 2 * Cl / (Gl - 1) + (xk - x) / t) / (1 + 2 / (Gl - 1))
            va = Ca - (xk - x) / t
            p = Pl*(Ca/Cl)**(2*Gl/(Gl-1))
            v = va
            Ro = Rol/(Pl/p)**(1/Gl)
            T = p/Rg/Ro
        if xk + (vc - (Gl * Pc / Roc) ** (1 / 2)) * t < x <= xk + vc * t:
            p = Pc
            Ro = Roc
            v = vc
            T = p / Rg / Ro
        D = vr + Cr*((Gr+1)/(2*Gr)*Pc/Pr+(Gr-1)/(2*Gr))**(1/2)
        if xk + vc * t < x <= xk+D*t:
            p = Pc
            v = vc
            Ro = Ror*((Gr+1)*Pc+(Gr-1)*Pr)/((Gr+1)*Pr+(Gr-1)*Pc)
            T = p/ Rg / Ro
        if xk+D*t < x:
            p = Pr
            v = vr
            Ro = Ror
            T = p / Rg / Ro
        return p, v, Ro

    u_exact = np.zeros((rmse_grid.shape[0],3))
    j=0
    for i in rmse_grid:
        u_exact[j] = exact(i)
        j +=1

    u_exact = torch.from_numpy(u_exact)

    error_rmse=torch.sqrt(torch.mean((u_exact-model(rmse_grid))**2))
    
  
    end_loss = Solution(grid, equation, model, 'NN').loss_evaluation(lambda_bound=100, weak_form=weak_form)
    exp_dict_list.append({'grid_res':grid_res,'time':end - start,'RMSE':error_rmse.detach().numpy(),'loss':end_loss.detach().numpy(),'type':'SOD_eqn','cache':CACHE})
    
    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))
    print('loss {}= {}'.format(grid_res, end_loss))

    return exp_dict_list

nruns=10

exp_dict_list=[]

CACHE=False

for grid_res in range(10,41,5):
    for _ in range(nruns):
        exp_dict_list.append(SOD_experiment(grid_res,CACHE))
   


exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df=pd.DataFrame(exp_dict_list_flatten)
#df.boxplot(by='grid_res',column='time',fontsize=42,figsize=(20,10))
#df.boxplot(by='grid_res',column='RMSE',fontsize=42,figsize=(20,10),showfliers=False)
df.to_csv('examples/benchmarking_data/weak_SOD_experiment_10_40_cache={}.csv'.format(str(CACHE)))