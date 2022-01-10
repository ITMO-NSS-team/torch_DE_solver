import torch
import numpy as np
from TEDEouS.input_preprocessing import grid_prepare, bnd_prepare, operator_prepare,batch_bconds_transform
from TEDEouS.points_type import grid_sort

def take_derivative_shift_op(model, term):
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
    # this one contains shifted grids (see input_preprocessing module)
    shift_grid_list = term[1]
    # signs corresponding to a grid
    s_order_norm_list = term[2]
    # float that represents power of the differential term
    power = term[3]
    # number of variables in equation
    variables = term[4]
    # initially it is an ones field
    der_term = (torch.zeros_like(model(shift_grid_list[0][0])[0:,0]) + 1).reshape(-1,1)
    
    for j, scheme in enumerate(shift_grid_list):
        # every shift in grid we should add with correspoiding sign, so we start
        # from zeros
        grid_sum = torch.zeros_like(model(scheme[0]))[0:,0].reshape(-1,1) #почему схема от 0?
        for k, grid in enumerate(scheme):
            # and add grid sequentially
            grid_sum += (model(grid)[0:,variables[j]]).reshape(-1,1) * s_order_norm_list[j][k]
            # Here we want to apply differential operators for every term in the product
        der_term = der_term * grid_sum ** power[j]
    der_term = coeff * der_term

         
    return der_term


def apply_const_shift_operator(model, operator):
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
        dif = take_derivative_shift_op(model, term)
        try:
            total += dif
        except NameError:
            total = dif
    return total


def apply_operator_set(model, operator_set):
    """
    Deciphers equation in a whole grid to a field.

    Parameters
    ----------
    model : torch.Sequential
        Neural network.
    operator : list
        Multiple (len(subset)>=1) operators in input form. See 
        input_preprocessing.operator_prepare()

    Returns
    -------
    total : torch.Tensor

    """
    field_part = []
    for operator in operator_set:
        field_part.append(apply_const_shift_operator(model, operator))
    field_part = torch.cat(field_part)
    return field_part


flatten_list = lambda t: [item for sublist in t for item in sublist]

def lp_norm(*arg,p=2,normalized=False,weighted=False):
    if weighted==True and len(arg)==1:
        print('No grid is passed, using non-weighted norm')
        weighted=False
    if len(arg)==2:
        grid=arg[0]
        mat=arg[1]
    elif len(arg)==1:
        mat=arg[0]
        grid=None
    else:
        print('Something went wrong, passed more than two arguments')
        return
    grid_prod=1
    if weighted:
        for i in range(grid.shape[-1]):
            grid_prod*=grid[:,i]
    if p>1: 
        if not weighted and not normalized:
             norm=torch.mean((mat) ** p)
        elif not weighted and normalized:
            norm=torch.pow(torch.mean((mat) ** p),1/p)
        elif weighted and not normalized:
            norm=torch.mean(grid_prod*(mat) ** p)
        elif weighted and normalized:
            norm=torch.pow(torch.mean(grid_prod*(mat) ** p),1/p)
    elif p==1:
        if not weighted:
             norm=torch.mean(torch.abs(mat))
        elif weighted:
            norm=torch.mean(grid_prod*torch.abs(mat))  
    return norm


def point_sort_shift_loss(model, grid, operator_set, bconds, lambda_bound=10,norm=None):

    op = apply_operator_set(model, operator_set)
    if bconds==None:
        loss = torch.mean((op) ** 2)
        return loss
    
    true_b_val_list = []
    b_val_list = []
    b_pos_list = []

    # we apply no  boundary conditions operators if they are all None

    simpleform = False
    for bcond in bconds:
        if bcond[1] == None:
            simpleform = True
        if bcond[1] != None:
            simpleform = False
            break
    if simpleform:
        for bcond in bconds:
            b_pos_list.append(bcond[0])

            if len(bcond[2]) == bcond[2].shape[-1]:
                true_boundary_val = bcond[2].reshape(-1,1)
            else: 
                true_boundary_val = bcond[2]

            true_b_val_list.append(true_boundary_val)
        true_b_val = torch.cat(true_b_val_list)
        b_op_val = model(grid)
        b_val = b_op_val[flatten_list(b_pos_list)]
    # or apply differential operator first to compute corresponding field and
    else:
        for bcond in bconds:
            b_pos = bcond[0]
            b_pos_list.append(bcond[0])
            b_cond_operator = bcond[1]
            
            if len(bcond[2]) == bcond[2].shape[-1]:
                true_boundary_val = bcond[2].reshape(-1,1)
            else: 
                true_boundary_val = bcond[2]
            true_b_val_list.append(true_boundary_val)
            if b_cond_operator == None or b_cond_operator == [[1, [None], 1]]:
                b_op_val = model(grid)
            else:
                b_op_val = apply_operator_set(model, b_cond_operator)
            # take boundary values
            b_val_list.append(b_op_val[b_pos])
        true_b_val = torch.cat(true_b_val_list)
        b_val = torch.cat(b_val_list)

    """
    actually, we can use L2 norm for the operator and L1 for boundary
    since L2>L1 and thus, boundary values become not so signifnicant, 
    so the NN converges faster. On the other hand - boundary conditions is the
    crucial thing for all that stuff, so we should increase significance of the
    coundary conditions
    """
    if norm==None:
        op_weigthed=False
        op_normalized=False
        op_p=2
        b_weigthed=False
        b_normalized=False
        b_p=2
    else:
        op_weigthed=norm['operator_weighted']
        op_normalized=norm['operator_normalized']
        op_p=norm['operator_p']
        b_weigthed=norm['boundary_weighted']
        b_normalized=norm['boundary_weighted']
        b_p=norm['boundary_p']
    
    loss = lp_norm(grid[:len(op)],op,weighted=op_weigthed,normalized=op_normalized,p=op_p) + \
    lambda_bound * lp_norm(grid[flatten_list(b_pos_list)],b_val - true_b_val,p=b_p,weighted=b_weigthed,normalized=b_normalized)
    
    return loss

def point_sort_shift_loss_batch(model, prepared_grid, point_type, operator, bconds,subset=['central'], lambda_bound=10,batch_size=32,h=0.001,norm=None):
    permutation = torch.randperm(prepared_grid.size()[0])
    loss=0
    batch_num=0
    for i in range(0,prepared_grid.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        if len(indices)<5:
            continue
        # batch= grid[indices]
        
        # batch_grid = grid_prepare(batch)
        batch_grid=prepared_grid[indices]


        batch_types=np.array(list(point_type.values()))[indices.tolist()]
        
        batch_type=dict(zip(batch_grid, batch_types))
        
        batch_dict=grid_sort(batch_type)
        batch_bconds=batch_bconds_transform(batch_grid,bconds)
        batch_bconds = bnd_prepare(batch_bconds, batch_grid,batch_dict, h=h)

        batch_operator = operator_prepare(operator, batch_dict, subset=subset, true_grid=prepared_grid[indices], h=h)
        
        
        loss+= point_sort_shift_loss(model, batch_grid, batch_operator, batch_bconds, lambda_bound=lambda_bound,norm=norm)
        batch_num+=1
    loss=1/batch_num*loss
    return loss


def compute_operator_loss(grid, model, operator, bconds, grid_point_subset=['central'], lambda_bound=10,h=0.001):
    prepared_grid = grid_prepare(grid)
    bconds = bnd_prepare(bconds, prepared_grid, h=h)
    operator = operator_prepare(operator, prepared_grid, subset=grid_point_subset, true_grid=grid, h=h)
    loss = point_sort_shift_loss(model, prepared_grid, operator, bconds, lambda_bound=lambda_bound)
    loss=float(loss.float())
    return loss


def derivative_1d(model,grid):
    # print('1d>2d')
    u=model.reshape(-1)
    x=grid.reshape(-1)
    
    # du_forward = (u-torch.roll(u, -1)) / (x-torch.roll(x, -1))
    
    # du_backward = (torch.roll(u, 1) - u) / (torch.roll(x, 1) - x)
    du =  (torch.roll(u, 1) - torch.roll(u, -1))/(torch.roll(x, 1)-torch.roll(x, -1))
    du[0] = (u[0]-u[1])/(x[0]-x[1])
    du[-1] = (u[-1]-u[-2])/(x[-1]-x[-2])
    
    du=du.reshape(model.shape)
    
    return du


def derivative(u_tensor, h_tensor, axis, scheme_order=1, boundary_order=1):
    #print('shape=',u_tensor.shape)
    if (u_tensor.shape[0]==1):
        du=derivative_1d(u_tensor,h_tensor)
        return du

    u_tensor = torch.transpose(u_tensor, 0, axis)
    h_tensor = torch.transpose(h_tensor, 0, axis)
    
    
    if scheme_order==1:
        du_forward = (-torch.roll(u_tensor, -1) + u_tensor) / \
                     (-torch.roll(h_tensor, -1) + h_tensor)
    
        du_backward = (torch.roll(u_tensor, 1) - u_tensor) / \
                      (torch.roll(h_tensor, 1) - h_tensor)
        du = (1 / 2) * (du_forward + du_backward)
    
    # dh=h_tensor[0,1]-h_tensor[0,0]
    
    if scheme_order==2:
        u_shift_down_1 = torch.roll(u_tensor, 1)
        u_shift_down_2 = torch.roll(u_tensor, 2)
        u_shift_up_1 = torch.roll(u_tensor, -1)
        u_shift_up_2 = torch.roll(u_tensor, -2)
        
        h_shift_down_1 = torch.roll(h_tensor, 1)
        h_shift_down_2 = torch.roll(h_tensor, 2)
        h_shift_up_1 = torch.roll(h_tensor, -1)
        h_shift_up_2 = torch.roll(h_tensor, -2)
        
        h1_up=h_shift_up_1-h_tensor
        h2_up=h_shift_up_2-h_shift_up_1
        
        h1_down=h_tensor-h_shift_down_1
        h2_down=h_shift_down_1-h_shift_down_2
       
        a_up=-(2*h1_up+h2_up)/(h1_up*(h1_up+h2_up))
        b_up=(h2_up+h1_up)/(h1_up*h2_up)
        c_up=-h1_up/(h2_up*(h1_up+h2_up))
        
        a_down=(2*h1_down+h2_down)/(h1_down*(h1_down+h2_down))
        b_down=-(h2_down+h1_down)/(h1_down*h2_down)
        c_down=h1_down/(h2_down*(h1_down+h2_down))
        
        du_forward=a_up*u_tensor+b_up*u_shift_up_1+c_up*u_shift_up_2
        du_backward=a_down*u_tensor+b_down*u_shift_down_1+c_down*u_shift_down_2
        du = (1 / 2) * (du_forward + du_backward)
        
        
    if boundary_order==1:
        if scheme_order==1:
            du[:, 0] = du_forward[:, 0]
            du[:, -1] = du_backward[:, -1]
        elif scheme_order==2:
            du_forward = (-torch.roll(u_tensor, -1) + u_tensor) / \
                         (-torch.roll(h_tensor, -1) + h_tensor)
        
            du_backward = (torch.roll(u_tensor, 1) - u_tensor) / \
                          (torch.roll(h_tensor, 1) - h_tensor)
            du[:, 0] = du_forward[:, 0]
            du[:, 1] = du_forward[:, 1]
            du[:, -1] = du_backward[:, -1]
            du[:, -2] = du_backward[:, -2]
    elif boundary_order==2:
        if scheme_order==2:
             du[:, 0] = du_forward[:, 0]
             du[:, 1] = du_forward[:, 1]
             du[:, -1] = du_backward[:, -1]
             du[:, -2] = du_backward[:, -2]
        elif scheme_order==1:
            u_shift_down_1 = torch.roll(u_tensor, 1)
            u_shift_down_2 = torch.roll(u_tensor, 2)
            u_shift_up_1 = torch.roll(u_tensor, -1)
            u_shift_up_2 = torch.roll(u_tensor, -2)
            
            h_shift_down_1 = torch.roll(h_tensor, 1)
            h_shift_down_2 = torch.roll(h_tensor, 2)
            h_shift_up_1 = torch.roll(h_tensor, -1)
            h_shift_up_2 = torch.roll(h_tensor, -2)
            
            h1_up=h_shift_up_1-h_tensor
            h2_up=h_shift_up_2-h_shift_up_1
            
            h1_down=h_tensor-h_shift_down_1
            h2_down=h_shift_down_1-h_shift_down_2
           
       
            a_up=-(2*h1_up+h2_up)/(h1_up*(h1_up+h2_up))
            b_up=(h2_up+h1_up)/(h1_up*h2_up)
            c_up=-h1_up/(h2_up*(h1_up+h2_up))
            
            a_down=(2*h1_up+h2_up)/(h1_down*(h1_down+h2_down))
            b_down=-(h2_down+h1_down)/(h1_down*h2_down)
            c_down=h1_down/(h2_down*(h1_down+h2_down))
        
            
            du_forward=a_up*u_tensor+b_up*u_shift_up_1+c_up*u_shift_up_2
            du_backward=a_down*u_tensor+b_down*u_shift_down_1+c_down*u_shift_down_2
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
                if grid.dim() == 1 or min(grid.shape) == 1:
                    b_val_list.append(model[:,position])
                else:
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
        print('No bconds is not possible, returning infinite loss')
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


def take_derivative_autograd (model, grid, term):
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
    # this one contains shifted grids (see input_preprocessing module)
    product = term[1]
    # float that represents power of the differential term
    power = term[2]
    # initially it is an ones field
    der_term = torch.zeros_like(model(grid)) + 1
    
    for j,derivative in enumerate(product):
        if derivative==[None]:
            der=model(grid)
        else:
            der=nn_autograd(model,grid,axis=derivative)
        der_term = der_term * der ** power[j]
    
    der_term = coeff * der_term
    
    
    return der_term


def apply_autograd_operator(model,grid, operator):
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
        dif = take_derivative_autograd(model, grid, term)
        try:
            total += dif
        except NameError:
            total = dif
    return total


def apply_autograd_bcond_operator(model,grid,bconds):
    true_b_val_list = []
    b_val_list = []
    b_pos_list = []

    # we apply no  boundary conditions operators if they are all None

    simpleform = False
    for bcond in bconds:
        if bcond[1] == None:
            simpleform = True
        if bcond[1] != None:
            simpleform = False
            break
    if simpleform:
        for bcond in bconds:
            b_pos_list.append(bcond[0])
            true_boundary_val = bcond[2].reshape(-1, 1)
            true_b_val_list.append(true_boundary_val)
        # print(flatten_list(b_pos_list))
        # b_pos=torch.cat(b_pos_list)
        true_b_val = torch.cat(true_b_val_list)
        b_op_val = model(grid)
        b_val = b_op_val[flatten_list(b_pos_list)]
    # or apply differential operatorn first to compute corresponding field and
    else:
        for bcond in bconds:
            b_pos = bcond[0]
            b_pos_list.append(bcond[0])
            b_cond_operator = bcond[1]
            true_boundary_val = bcond[2].reshape(-1, 1)
            true_b_val_list.append(true_boundary_val)
            if b_cond_operator == None or b_cond_operator == [[1, [None], 1]]:
                b_op_val = model(grid)
            else:
                b_op_val =apply_autograd_operator(model,grid,b_cond_operator)
            # take boundary values
            b_val_list.append(b_op_val[b_pos])
        true_b_val = torch.cat(true_b_val_list)
        b_val = torch.cat(b_val_list)
    return b_val,true_b_val



def autograd_loss(model, grid, operator, bconds, lambda_bound=10):
    if bconds == None:
        print('No bconds is not possible, returning infinite loss')
        return np.inf

    op = apply_autograd_operator(model,grid, operator)

    # we apply no  boundary conditions operators if they are all None

    b_val, true_b_val = apply_autograd_bcond_operator(model,grid,bconds)
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
    bcond_part=b_val - true_b_val
    loss = torch.mean((op) ** 2) + lambda_bound * torch.mean((bcond_part) ** 2)

    return loss



