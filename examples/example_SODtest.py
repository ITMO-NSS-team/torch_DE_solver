import torch
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.append('../')

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from input_preprocessing import Equation
from solver import Solver


device = torch.device('cpu')

p_l = 1
v_l = 0
Ro_l = 1
gam_l = 1.4

p_r = 0.1
v_r = 0
Ro_r = 0.125
gam_r = 1.4

x0 = 0.5
h = 0.05
x_grid=np.linspace(0,1,21)
t_grid=np.linspace(0,0.2,21)

x = torch.from_numpy(x_grid)
t = torch.from_numpy(t_grid)

grid = torch.cartesian_prod(x, t).float()

grid.to(device)

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
bconds = [[bnd1, bndval1_0, 0],
          [bnd1, bndval1_1, 1],
          [bnd1, bndval1_2, 2],
          [bnd2, bndval2_0, 0],
          [bnd2, bndval2_1, 1],
          [bnd2, bndval2_2, 2],
          [bnd3, bndval3_0, 0],
          [bnd3, bndval3_1, 1],
          [bnd3, bndval3_2, 2]]


'''
gas dynamic system equations:
Eiler's equations system for Sod test in shock tube

'''
gas_eq1={
        'dro/dt':
        {
            'const': 1,
            'term': [1],
            'power': 1,
            'var': 2
        },
        'v*dro/dx':
        {
            'const': 1,
            'term': [[None], [0]],
            'power': [1, 1],
            'var': [1, 2]
        },
        'ro*dv/dx':
        {
            'const': 1,
            'term': [[None], [0]],
            'power': [1, 1],
            'var': [2, 1]
        }
     }
gas_eq2 = {
        'ro*dv/dt':
        {
            'const': 1,
            'term': [[None], [1]],
            'power': [1, 1],
            'var': [2, 1]
        },
        'ro*v*dv/dx':
        {
            'const': 1,
            'term': [[None],[None], [0]],
            'power': [1, 1, 1],
            'var': [2, 1, 1]
        },
        'dp/dx':
        {
            'const': 1,
            'term': [0],
            'power': 1,
            'var': 0
        }
     }
gas_eq3 =  {
        'dp/dt':
        {
            'const': 1,
            'term': [1],
            'power': 1,
            'var': 0
        },
        'gam*p*dv/dx':
        {
            'const': gam_l,
            'term': [[None], [0]],
            'power': [1, 1],
            'var': [0, 1]
        },
        'v*dp/dx':
        {
            'const': 1,
            'term': [[None], [0]],
            'power': [1, 1],
            'var': [1, 0]
        }

     }

gas_eq = [gas_eq1, gas_eq2, gas_eq3]

model = torch.nn.Sequential(
        torch.nn.Linear(2, 200),
        torch.nn.Tanh(),
        torch.nn.Linear(200, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 200),
        torch.nn.Tanh(),
        torch.nn.Linear(200, 3)
    )
start = time.time()

equation = Equation(grid, gas_eq, bconds, h=h).set_strategy('NN')


model = Solver(grid, equation, model, 'NN').solve(
                                lambda_bound=1000, verbose=True, learning_rate=1e-2,
                                eps=1e-6, tmin=1000, tmax=1e5,use_cache=False,cache_dir='../cache/',cache_verbose=False,
                                save_always=False,no_improvement_patience=500,print_every=100)

end = time.time()

print('Time taken = {}'.format(end - start))

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

u_exact = np.zeros((grid.shape[0],3))
j=0
for i in grid:
  u_exact[j] = exact(i)
  j +=1

torch.from_numpy(u_exact)
def exact_solution_print(grid, u_exact):
  fig1 = plt.figure()
  ax1 = fig1.add_subplot(projection='3d')
  fig2 = plt.figure()
  ax2 = fig2.add_subplot(projection='3d')
  fig3 = plt.figure()
  ax3 = fig3.add_subplot(projection='3d')
  ax1.plot_trisurf(grid[:,0].reshape(-1), grid[:,1].reshape(-1), u_exact[:,0].reshape(-1), cmap='Blues', linewidth=0.2, alpha=1)
  ax1.plot_trisurf(grid[:,0].reshape(-1), grid[:,1].reshape(-1), model(grid)[:,0].detach().numpy().reshape(-1), cmap=cm.jet, linewidth=0.2, alpha=1)
  ax2.plot_trisurf(grid[:,0].reshape(-1), grid[:,1].reshape(-1), u_exact[:,1].reshape(-1), cmap='Blues', linewidth=0.2, alpha=1)
  ax2.plot_trisurf(grid[:,0].reshape(-1), grid[:,1].reshape(-1), model(grid)[:,1].detach().numpy().reshape(-1), cmap=cm.jet, linewidth=0.2, alpha=1)
  ax3.plot_trisurf(grid[:,0].reshape(-1), grid[:,1].reshape(-1), u_exact[:,2].reshape(-1), cmap='Blues', linewidth=0.2, alpha=1)
  ax3.plot_trisurf(grid[:,0].reshape(-1), grid[:,1].reshape(-1), model(grid)[:,2].detach().numpy().reshape(-1), cmap=cm.jet, linewidth=0.2, alpha=1)
  ax1.set_xlabel("x")
  ax1.set_ylabel("t")
  ax2.set_xlabel("x")
  ax2.set_ylabel("t")
  ax3.set_xlabel("x")
  ax3.set_ylabel("t")

exact_solution_print(grid, u_exact)
# 1d at one moment of time (t_var) plotting
t_var = 0.2
grid1 = torch.cartesian_prod(x, torch.from_numpy(np.array([t_var], dtype=np.float64))).float()
u_exact = np.zeros((grid1.shape[0],3))
j=0
for i in grid1:
  u_exact[j] = exact(i)
  j +=1
torch.from_numpy(u_exact)
plt.figure()
plt.title('pressure')
plt.plot(grid1[:,0], u_exact[:,0])
plt.plot(grid1[:,0], model(grid1)[:,0].detach().numpy().reshape(-1), '*')
plt.show()
plt.title('velocity')
plt.plot(grid1[:,0], u_exact[:,1])
plt.plot(grid1[:,0], model(grid1)[:,1].detach().numpy().reshape(-1), '*')
plt.show()
plt.title('density')
plt.plot(grid1[:,0], u_exact[:,2])
plt.plot(grid1[:,0], model(grid1)[:,2].detach().numpy().reshape(-1), '*')