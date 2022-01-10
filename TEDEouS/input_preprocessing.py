# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 21:31:03 2021

@author: user
"""
import torch

from TEDEouS.points_type import point_typization
from TEDEouS.points_type import grid_sort
from TEDEouS.finite_diffs import scheme_build
from TEDEouS.finite_diffs import sign_order
from TEDEouS.finite_diffs import second_order_scheme_build
from TEDEouS.finite_diffs import second_order_sign_order
import numpy as np

"""
Functions here serve to make all possible forms of input into an inner representation of solver
"""


def grid_prepare(grid):
    """
    In the practice we want to apply different finite difference schemes to a
    inner and boundary points. This one is axilluary function that allow to
    reorder grid nodes with respect to the point types - inner points first
    then boundary points of different types

    Parameters
    ----------
    grid : torch.Tensor (torch.float64)


    Returns
    -------
    sorted_grid : torch.Tensor (torch.float64)

    """
    point_type = point_typization(grid)

    grid_dict = grid_sort(point_type)

    sorted_grid = torch.cat(list(grid_dict.values()))
    
    
    return sorted_grid,grid_dict,point_type


def operator_unify(operator):
    """
    I just was annoyed adding additional square brackets to the operators.
    This one allows to make operator form simpler.

    Parameters
    ----------
    operator : list
        Operator in form ... .

    Returns
    -------
    unified_operator : list
        DESCRIPTION.

    """
    unified_operator = []
    for term in operator:
        const = term[0]
        vars_set = term[1]
        power = term[2]
        variables=[]
        if len(term)==4:
            variables = term[3]
        elif type(power)==int:
            variables=0
        elif type(power)==list:
            for _ in range(len(power)):
                variables.append(0)
        if variables is None:
            if type(power) is list:
                unified_operator.append([const, vars_set, power,0])
            else:
                unified_operator.append([const, [vars_set], [power],[0]])
        else:
            if type(power) is list:
                unified_operator.append([const, vars_set, power,variables])
            else:
                unified_operator.append([const, [vars_set], [power],[variables]])
        # if type(power) is list:
        #         unified_operator.append([const, vars_set, power])
        # else:
        #         unified_operator.append([const, [vars_set], [power]])
    return unified_operator


def operator_to_type_op(unified_operator, nvars, axes_scheme_type, h=1 / 2,inner_order=1, boundary_order=2):
    """
    Function serves applying different schemes to a different point types for
    entire operator

    Parameters
    ----------
    unified_operator : list
        operator in a proper form
    nvars : int
        Dimensionality of the problem.
    axes_scheme_type : string
        'central' or combination of 'f' and 'b'.
    h : float, optional
        Derivative precision parameter (to simplify h in the expression
                                        (f(x+h)-f(x))/h). The default is 1/2.
    boundary_order : int, optional
        Order of finite difference scheme taken at the domain boundary points. 
        The default is 1.

    Returns
    -------
    fin_diff_op : list
        list, where the conventional operator changed to steps and signs
        (see scheme_build function description).

    """
    fin_diff_op = []
    for term in unified_operator:
        fin_diff_list = []
        s_order_list = []
        const = term[0]
        vars_set = term[1]
        power = term[2]
        variables = term[3]
        for k, term in enumerate(vars_set):
            # None is for case where we need the fuction without derivatives
            if term != [None]:
                if axes_scheme_type == 'central':
                    if inner_order==1:
                        scheme, direction_list = scheme_build(term, nvars, 'central')
                        s_order = sign_order(len(term), 'central', h=h)
                    elif inner_order==2:
                        scheme, direction_list = second_order_scheme_build(term, nvars, 'central')
                        s_order = second_order_sign_order(len(term), 'central', h=h)
                else:
                    if boundary_order == 1:
                        scheme, direction_list = scheme_build(term, nvars, axes_scheme_type)
                        s_order = sign_order(len(term), direction_list, h=h)
                    elif boundary_order == 2:
                        scheme, direction_list = second_order_scheme_build(term, nvars, axes_scheme_type)
                        s_order = second_order_sign_order(len(term), direction_list, h=h)
            else:
                scheme = [None]
                s_order = [1]
            fin_diff_list.append(scheme)
            s_order_list.append(s_order)
        fin_diff_op.append([const, fin_diff_list, s_order_list, power,variables])
    return fin_diff_op


def shift_points(grid, axis, shift):
    """
    Shifts all values of an array 'grid' on a value 'shift' in a direcion of
    axis 'axis', somewhat is equivalent to a np.roll
    """
    grid_shift = grid.clone()
    grid_shift[:, axis] = grid[:, axis] + shift
    return grid_shift


def finite_diff_scheme_to_grid_list(finite_diff_scheme, grid, h=0.001):
    """
    Axiluary function that converts integer grid steps in term described in
    finite-difference scheme to a grids with shifted points, i.e.
    from field (x,y) -> (x,y+h).

    Parameters
    ----------
    finite_diff_scheme : list
        operator_to_type_op one term
    grid : torch.Tensor
    h : float
        derivative precision parameter. The default is 0.001.

    Returns
    -------
    s_grid_list : list
        list, where the the steps and signs changed to grid and signs
    """
    s_grid_list = []
    for i, shifts in enumerate(finite_diff_scheme):
        s_grid = grid
        for j, axis in enumerate(shifts):
            if axis != 0:
                s_grid = shift_points(s_grid, j, axis * h)
        s_grid_list.append(s_grid)
    return s_grid_list




def type_op_to_grid_shift_op(fin_diff_op, grid, h=0.001, true_grid=None):
    """
    Converts operator to a grid_shift form. Includes term coefficient
    conversion.
    Coeff may be integer, function or array, last two are mapped to a 
    subgrid that corresponds point type

    Parameters
    ----------
    fin_diff_op : list
        operator_to_type_op result.
    grid : torch.Tensor
        grid with sotred nodes (see grid_prepare)
    h : float
        derivative precision parameter. The default is 0.001.
    true_grid : TYPE, optional
        initial grid for coefficient in form of torch.Tensor mapping

    Returns
    -------
    shift_grid_op : list
        final form of differential operator used in the algorithm for single 
        grid type
    """
    shift_grid_op = []
    for term1 in fin_diff_op:
        shift_grid_list = []
        coeff1 = term1[0]
        if type(coeff1) == int or type(coeff1) == float:
            coeff = coeff1
        elif callable(coeff1):
            coeff = coeff1(grid)
            coeff = coeff.reshape(-1, 1)
        elif type(coeff1) == torch.Tensor:
            if true_grid != None:
                pos = bndpos(true_grid, grid)
            else:
                pos = bndpos(grid, grid)
            coeff = coeff1[pos].reshape(-1, 1)
        finite_diff_scheme = term1[1]
        s_order = term1[2]
        power = term1[3]
        variables = term1[4]
        for k, term in enumerate(finite_diff_scheme):
            if term != [None]:
                grid_op = finite_diff_scheme_to_grid_list(term, grid, h=h)
            else:
                grid_op = [grid]
            shift_grid_list.append(grid_op)
        shift_grid_op.append([coeff, shift_grid_list, s_order, power,variables])
    return shift_grid_op


def apply_all_operators(operator, grid_dict, h=0.001, subset=None, true_grid=None):
    """
    

    Parameters
    ----------
    operator : list
        operator_unify result.
    grid : torch.Tensor
        grid with sotred nodes (see grid_prepare)
    h : float
        derivative precision parameter. The default is 0.001.
    subset : list, optional
        grid subsets used for the operator ,e.g. ['central','fb','ff']
    true_grid : TYPE, optional
        initial grid for coefficient in form of torch.Tensor mapping

    Returns
    -------
    operator_list :  list
        final form of differential operator used in the algorithm for subset 
        grid types

    """
    operator_list = []
    # nvars = grid.shape[1]
    # point_type = point_typization(grid)
    # grid_dict = grid_sort(point_type)
    nvars =list(grid_dict.values())[0].shape[-1]
    a = operator_unify(operator)
    for operator_type in list(grid_dict.keys()):
        if subset == None or operator_type in subset:
            b = operator_to_type_op(a, nvars, operator_type, h=h)
            c = type_op_to_grid_shift_op(b, grid_dict[operator_type], h=h, true_grid=true_grid)
            operator_list.append(c)
    return operator_list


def operator_prepare(op, grid_dict, subset=['central'], true_grid=None, h=0.001):
    """
    Changes the operator in conventional form to the input one
    
    Parameters
    ----------
    op : list
        operator in conventional form.
    grid : torch.Tensor
        grid with sotred nodes (see grid_prepare)
    h : float
        derivative precision parameter. The default is 0.001.
    subset : list, optional
        grid subsets used for the operator ,e.g. ['central','fb','ff']
    true_grid : torch.Tensor, optional
        initial grid for coefficient in form of torch.Tensor mapping
    Returns
    -------
    operator_list :  list
        final form of differential operator used in the algorithm for subset 
        grid types

    """
    if type(op)==dict:
        op=op_dict_to_list(op)
    op1 = operator_unify(op)
    prepared_operator = apply_all_operators(op1, grid_dict, subset=subset, true_grid=true_grid, h=h)
    return prepared_operator





def op_dict_to_list(opdict):
    return list([list(term.values()) for term in opdict.values()])

def closest_point(grid,target_point):
    min_dist=np.inf
    pos=0
    min_pos=0
    for point in grid:
        dist=torch.linalg.norm(point-target_point)
        if dist<min_dist:
            min_dist=dist
            min_pos=pos
        pos+=1
    return min_pos


def bndpos(grid, bnd):
    """
    
    Returns the position of the boundary points on the grid
    
    Parameters
    ----------
    grid : torch.Tensor
        grid for coefficient in form of torch.Tensor mapping
    bnd : torch.Tensor
        boundary

    Returns
    -------
    bndposlist : list (int)
        positions of boundaty points in grid

    """
    if grid.shape[0]==1:
        grid=grid.reshape(-1,1)
    bndposlist = []
    grid = grid.double()
    if type(bnd) == np.array:
        bnd = torch.from_numpy(bnd).double()
    else:
        bnd = bnd.double()
    for point in bnd:
        try:
            pos = int(torch.where(torch.all(torch.isclose(grid, point), dim=1))[0])
        except Exception:
            pos=closest_point(grid,point)
        bndposlist.append(pos)
    return bndposlist


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


def bnd_prepare(bconds,grid,grid_dict, h=0.001):
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
            bop2 = apply_all_operators(bop1, grid_dict, h=h)
        else:
            bop2 = None
        prepared_bnd.append([bpos, bop2, bval])

    return prepared_bnd


def grid_intersect(t1, t2):
    t1=list(t1.cpu().numpy())
    t2=list(t2.cpu().numpy())    
    t1_set=[]
    t2_set=[]
    for item in t1:
        t1_set.append(tuple(item))
    for item in t2:
        t2_set.append(tuple(item))
    t1_set=set(t1_set)
    t2_set=set(t2_set)
    intersect=t1_set.intersection(t2_set)
    intersect=list(intersect)
    for i in range(len(intersect)):
        intersect[i]=list(intersect[i])
    intersect=torch.Tensor(intersect)
    return intersect

    
def batch_bconds_transform(batch_grid,bconds):
    bconds = bnd_unify(bconds)
    batch_bconds=[]
    for bcond in bconds:
        b_coord = bcond[0]
        bop = bcond[1]
        bval = bcond[2]
        grid_proj=grid_intersect(b_coord, batch_grid)
        if len(grid_proj)>0:
            proj_pos=bndpos(b_coord, grid_proj)
            bval=bval[proj_pos]
            batch_bconds.append([grid_proj,bop,bval])
    if len(batch_bconds)==0:
        batch_bconds=None
    return batch_bconds



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
        # bpos=bndpos(grid,bpts)
        for pt in bpts:
            if grid.shape[0]==1:
                point_pos=(torch.tensor(bndpos(grid,pt)),)
            else:
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

def operator_prepare_matrix(operator):
    if type(operator) == dict:
        operator = op_dict_to_list(operator)
    unified_operator = operator_unify(operator)
    return unified_operator


def expand_coeffs_autograd(op,grid):
    autograd_op=[]
    for term in op:
        coeff1 = term[0]
        if type(coeff1) == int:
            coeff = coeff1
        elif callable(coeff1):
            coeff = coeff1(grid)
            coeff = coeff.reshape(-1,1)
        elif type(coeff1) == torch.Tensor:
            coeff = coeff1.reshape(-1,1)
        prod = term[1]
        power = term[2]
        autograd_op.append([coeff, prod, power])
    return autograd_op

def operator_prepare_autograd(op,grid):
    """
    Changes the operator in conventional form to the input one
    
    Parameters
    ----------
    op : list
        operator in conventional form.
    Returns
    -------
    operator_list :  list
        final form of differential operator used in the algorithm 

    """
    if type(op)==dict:
        op=op_dict_to_list(op)
    unified_operator = operator_unify(op)
        
    prepared_operator=expand_coeffs_autograd(unified_operator,grid)
    
    return prepared_operator


def bnd_prepare_autograd(bconds,grid):
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

