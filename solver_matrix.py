import torch
from torch import tensor
from input_preprocessing import *
import sys
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.append('../')


def derivative(u_tensor, h_tensor, axis, scheme_order=1, boundary_order=1):
    u_tensor = torch.transpose(u_tensor, 0, axis)
    h_tensor = torch.transpose(h_tensor, 0, axis)

    du_forward = (-torch.roll(u_tensor, -1) + u_tensor) / \
                 (-torch.roll(h_tensor, -1) + h_tensor)

    du_backward = (torch.roll(u_tensor, 1) - u_tensor) / \
                  (torch.roll(h_tensor, 1) - h_tensor)
    du = (1 / 2) * (du_forward + du_backward)

    du[:, 0] = du_forward[:, 0]
    du[:, -1] = du_backward[:, -1]

    du = torch.transpose(du, 0, axis)

    return du


def take_matrix_diff_term(model, grid, term):
    """
    Axiluary function serves for single differential operator resulting field
    derivation

    Parameters
    ----------
    model : torch.Sequential
        Neural network.
    term : TYPE
        differential operator in conventional form.

    Returns
    -------
    der_term : torch.Tensor
        resulting field, computed on a grid.

    """
    # it is may be int, function of grid or torch.Tensor
    coeff = term[0]
    # this one contains product of differential operator
    operator_product = term[1]
    # float that represents power of the differential term
    power = term[2]
    # initially it is an ones field
    der_term = torch.zeros_like(model) + 1
    for j, scheme in enumerate(operator_product):
        prod=model
        if scheme!=[None]:
            for axis in scheme:
                if axis is None:
                    continue
                h = grid[axis]
                prod=derivative(prod, h, axis, scheme_order=1, boundary_order=1)
        der_term = der_term * prod ** power[j]

    if callable(coeff) is True:
        der_term = coeff(grid) * der_term
    else:
        der_term = coeff * der_term
    return der_term

def apply_matrix_diff_operator(model,grid, operator):
    """
    Deciphers equation in a single grid subset to a field.

    Parameters
    ----------
    model : torch.Sequential
        Neural network.
    operator : list
        Single (len(subset)==1) operator in input form. See
        input_preprocessing.operator_prepare()

    Returns
    -------
    total : torch.Tensor

    """
    for term in operator:
        dif = take_matrix_diff_term(model, grid, term)
        try:
            total += dif
        except NameError:
            total = dif
    return total

def bnd_prepare_matrix(bconds,grid):
    prepared_bconds=[]
    for bnd in bconds:
        if len(bnd)==2:
            bpts=bnd[0]
            bop=None
            bval=bnd[1]
        else:
            bpts=bnd[0]
            bop=bnd[1]
            bval=bnd[2]
        bpos=[]
        for pt in bpts:
            prod=(torch.zeros_like(grid[0])+1).bool()
            for axis in range(grid.shape[0]):
                axis_intersect=torch.isclose(pt[axis].float(),grid[axis].float())
                prod*=axis_intersect
            point_pos=torch.where(prod==True)
            bpos.append(point_pos)
        if type(bop)==dict:
            bop=op_dict_to_list(bop)
        if bop!=None:
            bop = operator_unify(bop)
        prepared_bconds.append([bpos,bop,bval])
    return prepared_bconds

def apply_matrix_bcond_operator(model,grid,b_prepared):
    true_b_val_list = []
    b_val_list = []

    # we apply no  boundary conditions operators if they are all None

    simpleform = False
    for bcond in b_prepared:
        if bcond[1] == None:
            simpleform = True
        if bcond[1] != None:
            simpleform = False
            break
    if simpleform:
        for bcond in b_prepared:
            b_pos = bcond[0]
            true_boundary_val = bcond[2]
            for position in b_pos:
                b_val_list.append(model[position])
            true_b_val_list.append(true_boundary_val)
        b_val=torch.cat(b_val_list)
        true_b_val=torch.cat(true_b_val_list)
    # or apply differential operatorn first to compute corresponding field and
    else:
        for bcond in b_prepared:
            b_pos = bcond[0]
            b_cond_operator = bcond[1]
            true_boundary_val = bcond[2]
            if b_cond_operator == None or b_cond_operator == [[1, [None], 1]]:
                b_op_val = model
            else:
                b_op_val = apply_matrix_diff_operator(model,grid,b_cond_operator)
            # take boundary values
            for position in b_pos:
                if grid.dim() == 1 or min(grid.shape) == 1:
                    b_val_list.append(b_op_val[:, position])
                else:
                    b_val_list.append(b_op_val[position])
            true_b_val_list.append(true_boundary_val)
        b_val=torch.cat(b_val_list)
        true_b_val=torch.cat(true_b_val_list)
    return b_val,true_b_val


def matrix_loss(model, grid, operator, bconds, lambda_bound=10):
    if bconds == None:
        print('No bconds is not possible, returning ifinite loss')
        return np.inf

    op = apply_matrix_diff_operator(model, grid, operator)

    # we apply no  boundary conditions operators if they are all None

    b_val, true_b_val = apply_matrix_bcond_operator(model, grid, bconds)
    """
    actually, we can use L2 norm for the operator and L1 for boundary
    since L2>L1 and thus, boundary values become not so signifnicant, 
    so the NN converges faster. On the other hand - boundary conditions is the
    crucial thing for all that stuff, so we should increase significance of the
    coundary conditions
    """
    # l1_lambda = 0.001
    # l1_norm =sum(p.abs().sum() for p in model.parameters())
    # loss = torch.mean((op) ** 2) + lambda_bound * torch.mean((b_val - true_b_val) ** 2)+ l1_lambda * l1_norm

    loss = torch.mean((op) ** 2) + lambda_bound * torch.mean((b_val - true_b_val) ** 2)

    return loss

def lbfgs_solution(model, grid, operator, norm_lambda, bcond, rtol=1e-6,atol=0.01,nsteps=10000):

    if type(operator) == dict:
        operator = op_dict_to_list(operator)
    unified_operator = operator_unify(operator)

    b_prepared = bnd_prepare_matrix(bcond, grid)

    # optimizer = torch.optim.Adam([model.requires_grad_()], lr=1e-4)
    optimizer = torch.optim.LBFGS([model.requires_grad_()], lr=1e-3)
   
    
    def closure():
        nonlocal cur_loss
        optimizer.zero_grad()

        loss = matrix_loss(model, grid, unified_operator, b_prepared, lambda_bound=norm_lambda)
        loss.backward()
        cur_loss = loss.item()

        return loss

    cur_loss = float('inf')
    # tol = 1e-20

    for i in range(nsteps):
        optimizer.zero_grad()
        past_loss = cur_loss
        
        # loss = matrix_loss(model, grid, unified_operator, b_prepared, lambda_bound=norm_lambda)
        optimizer.step(closure)
        
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # cur_loss = loss.item()
        
        
        if i % 1000 == 0:
            print('i={} loss={}'.format(i, cur_loss))

        if abs(cur_loss - past_loss) / abs(cur_loss) < rtol:
        #     # print("sosholsya")
            break
        if abs(cur_loss) < atol:
        #     # print("sosholsya")
            break

    return model.detach()

def solution_print(prepared_grid,model,title=None):
    if model.dim() == 1:
        fig = plt.figure()
        plt.scatter(prepared_grid.reshape(-1), model.detach().numpy().reshape(-1))
        plt.show()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if title!=None:
            ax.set_title(title)
        ax.plot_trisurf(prepared_grid[0].reshape(-1), prepared_grid[1].reshape(-1),
                        model.reshape(-1).detach().numpy(), cmap=cm.jet, linewidth=0.2, alpha=1)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        plt.show()
