# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 22:25:22 2021

@author: Sashka
"""
import numpy as np
import torch
from scipy.spatial import Delaunay


def shift_points(grid, axis, shift):
    """
    Shifts all values of an array 'grid' on a value 'shift' in a direcion of
    axis 'axis', somewhat is equivalent to a np.roll
    """
    grid_shift = grid.clone()
    grid_shift[:, axis] = grid[:, axis] + shift
    return grid_shift


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if p.shape[1] > 1:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)

        return hull.find_simplex(p) >= 0
    elif p.shape[1] == 1:
        """
        this one is not a snippet from a stackexchange it does the same 
        but for a 1-D case, which is not covered in a code above
        """
        upbound = torch.max(hull)
        lowbound = torch.min(hull)
        return np.array(((p <= upbound) & (p >= lowbound)).reshape(-1))


def point_typization(grid):
    """
    

    Parameters
    ----------
    grid : torch.Tensor (torch.float64)
        array of a n-D points

    Returns
    -------
    point_type : dict
        dictionary point:type with a points in a 'grid' above
        type may be 'central' - inner point
        and string of 'f' and 'b', where the length of the string 
        is a dimension n
        'f' means that if we add small number to a position of corresponding
        coordinate we stay in the 'hull'
        'b' means that if we subtract small number from o a position 
        of corresponding coordinate we stay in the 'hull'
    """
    direction_list = []
    for axis in range(grid.shape[1]):
        for direction in range(2):
            direction_list.append(in_hull(shift_points(grid, axis, (-1) ** direction * 0.0001), grid))

    direction_list = np.array(direction_list)
    direction_list = np.transpose(direction_list)

    point_type = {}

    for i, point in enumerate(grid):
        if np.all(direction_list[i]):
            point_type[point] = 'central'
        else:
            p_type = ''
            j = 0
            while j < len(direction_list[i]):
                if (j % 2 == 0 and direction_list[i, j]) or (
                        j % 2 == 0 and direction_list[i, j] and direction_list[i, j + 1]):
                    p_type += 'f'
                else:
                    p_type += 'b'
                j += 2
            point_type[point] = p_type
    return point_type


def grid_sort(point_type):
    """
    

    Parameters
    ----------
    point_type : dict
        dictionary point:type with a points in a 'grid' above
        type may be 'central' - inner point
        and string of 'f' and 'b', where the length of the string 
        is a dimension n
        'f' means that if we add small number to a position of corresponding
        coordinate we stay in the 'hull'
        'b' means that if we subtract small number from o a position 
        of corresponding coordinate we stay in the 'hull'

    Returns
    -------
    grid_dict : dict
        dictionart type:points list
        basically reversed dictionaty
    """
    point_types = set(point_type.values())
    grid_dict = {}
    for p_type in point_types:
        grid_dict[p_type] = []
    for point in list(point_type.keys()):
        p_type = point_type[point]
        grid_dict[p_type].append(point)
    for p_type in point_types:
        grid_dict[p_type] = torch.stack(grid_dict[p_type])
    return grid_dict
