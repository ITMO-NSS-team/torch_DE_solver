import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys

sys.path.append('../')

from solver_matrix import *
import time

device = torch.device('cpu')

"""
Preparing grid

Grid is an essentially torch.Tensor of a n-D points where n is the problem
dimensionality
"""

t = torch.from_numpy(np.linspace(0, 1, 10))
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
bnd1 = torch.from_numpy(np.array([[0]], dtype=np.float64))

bop1 = None

#  So u(0)=-1/2
bndval1 = torch.from_numpy(np.array([[-1 / 2]], dtype=np.float64))

# point t=1
bnd2 = torch.from_numpy(np.array([[1]], dtype=np.float64))

# d/dt
bop2 = {
    '1*du/dt**1':
        {
            'coefficient': 1,
            'du/dt': [0],
            'pow': 1
        }
}
    
    
# So, du/dt |_{x=1}=3
bndval2 = torch.from_numpy(np.array([[3]], dtype=np.float64))

# Putting all bconds together
bconds = [[bnd1, bop1, bndval1], [bnd2, bop2, bndval2]]

"""
Defining Legendre polynomials generating equations

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


# 1-t^2
def c1(grid):
    return 1 - grid ** 2


# -2t
def c2(grid):
    return -2 * grid

def c3(n):
    return n*(n-1)

n=3

# # operator is  (1-t^2)*d2u/dt2-2t*du/dt+n*(n-1)*u=0 (n=3)
# legendre_poly= {
#     '(1-t^2)*d2u/dt2**1':
#         {
#             'coeff': c1(grid), #coefficient is a torch.Tensor
#             'du/dt': [0, 0],
#             'pow': 1
#         },
#     '-2t*du/dt**1':
#         {
#             'coeff': c2(grid),
#             'u*du/dx': [0],
#             'pow':1
#         },
#     'n*(n-1)*u**1':
#         {
#             'coeff': 6,
#             'u':  [None],
#             'pow': 1
#         }
# }

# this one is to show that coefficients may be a function of grid as well
legendre_poly= {
    '(1-t^2)*d2u/dt2**1':
        {
            'coeff': c1, #coefficient is a function
            'du/dt': [0, 0],
            'pow': 1
        },
    '-2t*du/dt**1':
        {
            'coeff': c2,
            'u*du/dx': [0],
            'pow':1
        },
    'n*(n-1)*u**1':
        {
            'coeff': 6,
            'u':  [None],
            'pow': 1
        }
}
    
    
    
model = torch.rand(grid.shape)



if type(legendre_poly) == dict:
    legendre_poly = op_dict_to_list(legendre_poly)
legendre_poly = operator_unify(legendre_poly)

def bnd_unify(bconds):
    """
    Serves to add None instead of empty operator

    Parameters
    ----------
    bconds : list
        
        boundary in conventional form (see examples)

    Returns
    -------
    unified_bconds : list
        
        boundary in input-friendly form

    """
    if bconds==None:
        return None
    unified_bconds = []
    for bcond in bconds:
        if len(bcond) == 2:
            unified_bconds.append([bcond[0], None, bcond[1]])
        elif len(bcond) == 3:
            unified_bconds.append(bcond)
    return unified_bconds

unified_bnds=bnd_unify(bconds)

def bnd_prepare_matrix(bconds, grid):
    """
    

    Parameters
    ----------
    bconds : list
        boundary in conventional form (see examples)
    grid : torch.Tensor
        grid with sotred nodes (see grid_prepare)
    h : float
        derivative precision parameter. The default is 0.001.

    Returns
    -------
    prepared_bnd : list
        
        boundary in input form

    """
    bconds = bnd_unify(bconds)
    if bconds==None:
        return None
    prepared_bnd = []
    for bcond in bconds:
        b_coord = bcond[0]
        bop = bcond[1]
        bval = bcond[2]
        bpos = bndpos(grid, b_coord)
        if bop == [[1, [None], 1]]:
            bop = None
        if bop != None:
            if type(bop)==dict:
                bop=op_dict_to_list(bop)
            bop1 = operator_unify(bop)
        else:
            bop1 = None
        prepared_bnd.append([bpos, bop1, bval])

    return prepared_bnd





b_prepared = bnd_prepare_matrix(bconds, grid)


unified_operator = operator_unify(legendre_poly)

optimizer = torch.optim.LBFGS([model.requires_grad_()], lr=0.001)

optimizer.zero_grad()

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
    h = grid
    for j, scheme in enumerate(operator_product):
        prod=model
        if scheme!=None:
            for axis in scheme:
                if axis is None:
                    continue
                if grid.shape[1] != 1:
                    h = grid[axis]
                prod=derivative(prod, h, axis, scheme_order=1, boundary_order=1)
        der_term = der_term * prod ** power[j]

    if callable(coeff) is True:
        der_term = coeff(h) * der_term
    else:
        der_term = coeff * der_term
    return der_term


def derivative(u_tensor, h_tensor, axis, scheme_order=1, boundary_order=1):
    u_tensor = torch.transpose(u_tensor, 0, axis)
    h_tensor = torch.transpose(h_tensor, 0, axis)

    du_forward = (-torch.roll(u_tensor, -1) + u_tensor) / \
                 (-torch.roll(h_tensor, -1) + h_tensor)
    du_backward = (torch.roll(u_tensor, 1) - u_tensor) / \
                  (torch.roll(h_tensor, 1) - h_tensor)

    du = (1 / 2) * (du_forward + du_backward)
    
    if du.shape[1]!=1:
        du[:, 0] = du_forward[:, 0]
        du[:, -1] = du_backward[:, -1]
    else:
        du[0] = du_forward[0]
        du[-1] = du_backward[-1]

    du = torch.transpose(du, 0, axis)

    return du


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






op = apply_matrix_diff_operator(model, grid, unified_operator)


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

        if model.dim() != 1:
            b_val=torch.cat(b_val_list)
            true_b_val=torch.cat(true_b_val_list)
        else:
            b_val = b_val_list
            true_b_val = true_b_val_list
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
                b_val_list.append(b_op_val[position])
                # print(b_op_val)
            true_b_val_list.append(true_boundary_val)
        # print(b_val_list)
        
        b_val=torch.cat(b_val_list)
        if grid.shape[1]==1:
            b_val=b_val.reshape(-1,1)
        true_b_val=torch.cat(true_b_val_list)
        
    return b_val,true_b_val






b_val,true_b_val=apply_matrix_bcond_operator(model,grid,b_prepared)







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



loss = matrix_loss(model, grid, unified_operator, b_prepared, lambda_bound=1)

