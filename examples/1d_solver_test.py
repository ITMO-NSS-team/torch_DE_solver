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


loss = matrix_loss(model, grid, unified_operator, b_prepared, lambda_bound=1)

op = apply_matrix_diff_operator(model, grid, unified_operator)
