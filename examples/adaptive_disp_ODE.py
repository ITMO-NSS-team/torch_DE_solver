import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from SALib import ProblemSpec
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

t_grid = np.linspace(0, 0.25, 100)

coord_list = [t_grid]

coord_list=torch.tensor(coord_list)
grid=coord_list.reshape(-1,1).float()
    

def func(grid):
    t = grid
    sln=1/324*(200 - 567*torch.exp(t) + 43*torch.exp(9*t) + 180*t)
    return sln    

# Initial conditions at t=0
bnd1 = torch.from_numpy(np.array([[0]], dtype=np.float64)).float()

# u(0,x)=sin(pi*x)
bndval1 = torch.tensor([-1]).float()

bnd2 = torch.from_numpy(np.array([[0]], dtype=np.float64)).float()

bop2 = {
'du/dt':
    {
        'coeff': 1,
        'du/dx': [1],
        'pow': 1,
        'var':0
    }
    }

bndval2=torch.tensor([0]).float()


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


model = torch.nn.Sequential(
        torch.nn.Linear(1, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 1)
    )


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)






def take_ODE_op(model,grid):
    return nn_autograd_simple(model, grid, 2,axis=0).reshape(-1,1)-10*nn_autograd_simple(model, grid, 1,axis=0).reshape(-1,1)+9*model(grid)-5*grid


bc_bnd=bnd1
bc_true_val=bndval1

def take_boundary_op(model,boundary):
    return model(boundary)-bc_true_val


ic_true_val=bndval2
ic_bnd=bnd2

def take_initial_op(model,ic_bnd):
    ic_val=nn_autograd_simple(model, ic_bnd,1,axis=0)
    return ic_val-ic_true_val

lam_ic=50#100
lam_bc=50#100
#lam_op=1

#loss = lam_op*torch.mean(take_ODE_op(model,grid)**2)+lam_bc*torch.mean(torch.abs(take_boundary_op(model,bc_bnd)))+lam_ic*torch.mean(torch.abs(take_initial_op(model,ic_bnd)))
loss = torch.mean(take_ODE_op(model,grid)**2)+lam_bc*torch.mean(torch.abs(take_boundary_op(model,bc_bnd)))+lam_ic*torch.mean(torch.abs(take_initial_op(model,ic_bnd)))


print(loss.item())

t=0

loss_mean=1000
min_loss=np.inf


op_list=[]
bc_list=[]
ic_list=[]
loss_list=[]

PLOT_SAVE=True
sampling_N=1
sampling_D=len(grid)+len(bc_bnd)+len(ic_bnd)

second_order_interactions=True

if second_order_interactions:
    sampling_amount=sampling_N*(2*sampling_D+2)
else:
    sampling_amount=sampling_N*(sampling_D+2)

errors_list=[]

adaptive_lambdas=False
#adaptive_lambdas=False


while t<1e4:
#while t<sampling_amount:
        optimizer.zero_grad()


        op=take_ODE_op(model,grid)
        bc=take_boundary_op(model,bc_bnd)
        ic=take_initial_op(model,ic_bnd)

        op_list.append(op.reshape(-1).detach().numpy())
        bc_list.append(bc.reshape(-1).detach().numpy())
        ic_list.append(ic.reshape(-1).detach().numpy())

        #loss = lam_op*torch.mean(op**2)+lam_bc*torch.mean(torch.abs(bc))+lam_ic*torch.mean(torch.abs(ic))
        loss = torch.mean(op**2)+lam_bc*torch.mean(torch.abs(bc))+lam_ic*torch.mean(torch.abs(ic))

        loss_list.append(float(loss.item()))

        loss.backward()
        optimizer.step()

        if (adaptive_lambdas) and ((len(op_list))==sampling_amount):
        #if (len(op_list))==sampling_amount:
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
            op_disp=sum(ST[:len(grid)])
            bc_disp=sum(ST[len(grid):len(grid)+len(bc_bnd)])
            ic_disp=sum(ST[len(grid)+len(bc_bnd):])

            lam_bc=total_disp/bc_disp
            lam_ic=total_disp/ic_disp
            #lam_op=total_disp/op_disp

            print('Lambda update t={}, loss={}, op_loss={}'.format(t,loss.item(),torch.mean(op**2)))
            #print('New lambdas lam_op={} lam_bc={} lam_ic={}'.format(lam_op,lam_bc,lam_ic))
            print('New lambdas lam_bc={} lam_ic={}'.format(lam_bc,lam_ic))

            op_list=[]
            bc_list=[]
            ic_list=[]
            loss_list=[]
            


        if t%1000==0:
            error_rmse=torch.sqrt(torch.mean((func(grid).reshape(-1,1)-model(grid))**2))
            print('Surface trainig t={}, loss={}, op_loss={}, error={}'.format(t,loss.item(),torch.mean(op**2),error_rmse))
            errors_list.append(error_rmse.detach().cpu().numpy())


            if PLOT_SAVE:
                fig = plt.figure(figsize=(15,8))
                plt.plot(t_grid,model(grid).reshape(-1).detach().numpy())
                #plt.set_xlabel("t")
                #plt.set_ylabel("y")
                plt.savefig('disp_exp_ODE/plot_disp_{}.png'.format(t))

                #fig = plt.figure(figsize=(15,8))
                #plt.plot(t_grid,nn_autograd_simple(model, grid, 1,axis=0).reshape(-1).detach().numpy())
                #plt.set_xlabel("t")
                #plt.set_ylabel("y")
                #plt.savefig('disp_exp_ODE/plot_disp_d1_{}.png'.format(t))
        t+=1


plt.figure()
plt.plot(np.arange(len(errors_list))*1000,np.array(errors_list))

if adaptive_lambdas:
    plt.savefig('disp_exp_ODE/plot_error_N_{}.png'.format(sampling_N))
else:
    plt.savefig('disp_exp_ODE/plot_error_const_lambda_{}.png'.format(lam_bc))