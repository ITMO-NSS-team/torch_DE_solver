import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from SALib import ProblemSpec


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

x_grid=np.linspace(0,1,10+1)
t_grid=np.linspace(0,1,10+1)
    
x = torch.from_numpy(x_grid)
t = torch.from_numpy(t_grid)
    
#h=abs((t[1]-t[0]).item())

grid = torch.cartesian_prod(x, t).float()
    

# Initial conditions at t=0
bnd1 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()

# u(0,x)=sin(pi*x)
bndval1 = torch.sin(np.pi * bnd1[:, 0])

bnd2 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()

bop2 = {
'du/dt':
    {
        'coeff': 1,
        'du/dx': [1],
        'pow': 1,
        'var':0
    }
    }

bndval2=torch.from_numpy(np.zeros(len(bnd2), dtype=np.float64))

# Boundary conditions at x=0
bnd3 = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), t).float()

# u(0,t)=0
bndval3 = torch.from_numpy(np.zeros(len(bnd3), dtype=np.float64))

# Boundary conditions at x=1
bnd4 = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)), t).float()

# u(1,t)=0
bndval4 = torch.from_numpy(np.zeros(len(bnd4), dtype=np.float64))

# Putting all bconds together
bconds = [[bnd1, bndval1], [bnd2, bndval2], [bnd3, bndval3], [bnd4, bndval4]]


model = torch.nn.Sequential(
        torch.nn.Linear(2, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 1)
    )



def nn_autograd_simple(model, points, order,axis=0):
    points.requires_grad=True
    f = model(points).sum()
    for i in range(order):
        grads, = torch.autograd.grad(f, points, create_graph=True)
        f = grads[:,axis].sum()
    return grads[:,axis]


def nn_autograd_mixed(model, points,axis=[0]):
    points.requires_grad=True
    f = model(points).sum()
    for ax in axis:
        grads, = torch.autograd.grad(f, points, create_graph=True)
        f = grads[:,ax].sum()
    return grads[:,axis[-1]]



def nn_autograd(*args,axis=0):
    model=args[0]
    points=args[1]
    if len(args)==3:
        order=args[2]
        grads=nn_autograd_simple(model, points, order,axis=axis)
    else:
        grads=nn_autograd_mixed(model, points,axis=axis)
    return grads.reshape(-1,1)



optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)






def take_wave_op(model,grid):
    return nn_autograd_simple(model, grid, 2,axis=0)-1/4*nn_autograd_simple(model, grid, 2,axis=1)


bc_bnd=torch.cat((bnd3,bnd4))
bc_true_val=torch.cat((bndval3,bndval4)).reshape(-1,1)

def take_boundary_op(model,boundary):
    return model(boundary)-bc_true_val


ic_true_val=torch.cat((bndval1,bndval2)).reshape(-1,1)

def take_initial_op(model,bnd1,bnd2):
    ic_val1=model(bnd1)
    ic_val2=nn_autograd_simple(model, bnd2, 1,axis=1).reshape(-1,1)
    ic_val=torch.cat((ic_val1,ic_val2))
    return ic_val-ic_true_val

lam_ic=100
lam_bc=100

loss = torch.mean(take_wave_op(model,grid)**2)+lam_bc*torch.mean(torch.abs(take_boundary_op(model,bc_bnd)))+lam_ic*torch.mean(torch.abs(take_initial_op(model,bnd1,bnd2)))

print(loss.item())

t=0

loss_mean=1000
min_loss=np.inf

#while loss.item()>1e-5 and t<1e5:

op_list=[]
bc_list=[]
ic_list=[]
loss_list=[]

PLOT_SAVE=True
sampling_N=2
sampling_D=len(grid)+len(bc_bnd)+len(bnd1)+len(bnd2)

second_order_interactions=False

if second_order_interactions:
    sampling_amount=sampling_N*(2*sampling_D+2)
else:
    sampling_amount=sampling_N*(sampling_D+2)

while t<1e5:
        optimizer.zero_grad()

        # in case you wanted a semi-full example
        # outputs = model.forward(batch_x)

        op=take_wave_op(model,grid)
        bc=take_boundary_op(model,bc_bnd)
        ic=take_initial_op(model,bnd1,bnd2)

        op_list.append(op.detach().numpy())
        bc_list.append(bc.reshape(-1).detach().numpy())
        ic_list.append(ic.reshape(-1).detach().numpy())

        loss = torch.mean(op**2)+lam_bc*torch.mean(torch.abs(bc))+lam_ic*torch.mean(torch.abs(ic))

        loss_list.append(float(loss.item()))

        loss.backward()
        optimizer.step()

        if (len(op_list))==sampling_amount:
        #if (len(op_list))==400:
            op_array=np.array(op_list)
            bc_array=np.array(bc_list)
            ic_array=np.array(ic_list)

            X_array=np.hstack((op_array,bc_array,ic_array))

            loss_array=np.array(loss_list)


            bounds=[[-100,100] for i in range(sampling_D)]
            names=['x{}'.format(i) for i in range(sampling_D)]
            sp=ProblemSpec({'names':names,'bounds':bounds})
            sp.set_samples(X_array)
            sp.set_results(loss_array)
            sp.analyze_sobol(calc_second_order=second_order_interactions)

            ST=sp.analysis['ST']

            total_disp=sum(ST)
            bc_disp=sum(ST[len(grid):len(grid)+len(bc_bnd)])
            ic_disp=sum(ST[len(grid)+len(bc_bnd):])

            lam_bc=total_disp/bc_disp
            lam_ic=total_disp/ic_disp

            print('Lambda update t={}, loss={}'.format(t,loss.item()))
            print('New lam_bc={} lam_ic={}'.format(lam_bc,lam_ic))

            op_list=[]
            bc_list=[]
            ic_list=[]
            loss_list=[]
            


        if t%1000==0:
            print('Surface trainig t={}, loss={}'.format(t,loss.item()))
            
            if PLOT_SAVE:
                fig = plt.figure(figsize=(15,8))
                ax1 = fig.add_subplot( projection='3d')
                ax1.plot_trisurf(grid[:, 0].detach().cpu().numpy().reshape(-1), 
                            grid[:, 1].detach().cpu().numpy().reshape(-1),
                            model(grid).detach().cpu().numpy().reshape(-1),
                            cmap=plt.cm.jet, linewidth=0.2, alpha=1)
                ax1.set_xlabel("x1")
                ax1.set_ylabel("x2")
                plt.savefig('disp_exp/plot_disp_{}.png'.format(t))
        t+=1

