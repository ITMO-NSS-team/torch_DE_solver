"""Module for determine types of grid points. Only for *NN* mode."""


from typing import Union
from scipy.spatial import Delaunay
import numpy as np
import torch

class Points_type():
    """
    Discretizing the grid and allocating subsets for Finite Difference method.
    """

    def __init__(self, grid: torch.Tensor):
        """
        Initializes the `Points_type` object with a grid of discretization points. These points represent the domain over which the differential equation is approximated using a neural network.
        
                Args:
                    grid (torch.Tensor): Discretization points of the domain where the differential equation is approximated.
        
                Returns:
                    None
        """

        self.grid = grid

    @staticmethod
    def shift_points(grid: torch.Tensor, axis: int, shift: float) -> torch.Tensor:
        """
        Adjusts the discretization grid by shifting points along a specified axis.
        
        This function modifies the grid by adding a 'shift' value to all points along the given 'axis'.
        This is useful for exploring the solution space of differential equations by slightly perturbing
        the discretization and observing the impact on the neural network's learned solution.
        
        Args:
            grid (torch.Tensor): Discretization of the domain. Each row represents a point, and each column
                                 represents a dimension.
            axis (int): The axis (dimension) along which to apply the shift.
            shift (float): The value by which to shift the points along the specified axis.
        
        Returns:
            torch.Tensor: A new tensor representing the shifted grid.
        """

        grid_shift = grid.clone()
        grid_shift[:, axis] = grid[:, axis] + shift
        return grid_shift

    @staticmethod
    def _in_hull(p: torch.Tensor, hull: torch.Tensor) -> np.ndarray:
        """
        Checks if points `p` lie within the convex hull defined by `hull`.
        
        This function determines whether a set of points lies within a defined boundary,
        which is crucial for evaluating the accuracy and stability of neural network solutions
        to differential equations. By verifying that the solution points remain within a valid
        region, we can assess the reliability of the neural network's approximation.
        
        Args:
            p (torch.Tensor): A tensor of shape (N, K) representing N points in K dimensions.
            hull (torch.Tensor): A tensor of shape (M, K) representing the coordinates of M points
                in K dimensions, which define the convex hull. It can also be a
                scipy.spatial.Delaunay object.
        
        Returns:
            np.ndarray: A boolean array of shape (N,) indicating whether each point in `p`
                lies within the convex hull defined by `hull`. True indicates that the point
                is inside the hull, and False indicates that it is outside.
        """

        if p.shape[1] > 1:
            if not isinstance(hull, Delaunay):
                hull = Delaunay(hull.cpu())

            return hull.find_simplex(p.cpu()) >= 0
        elif p.shape[1] == 1:
            # this one is not a snippet from a stackexchange it does the same
            # but for a 1-D case, which is not covered in a code above
            upbound = torch.max(hull).cpu()
            lowbound = torch.min(hull).cpu()
            return np.array(((p.cpu() <= upbound) & (p.cpu() >= lowbound)).reshape(-1))

    def point_typization(self) -> dict:
        """
        Identifies the type of each point in the grid based on its proximity to the solution boundary.
        
                This method determines whether a point is an interior point ('central') or a boundary point.
                For boundary points, it identifies the directions ('f' for forward, 'b' for backward) in which
                small perturbations would keep the point within the solution space. This classification aids in
                understanding the behavior of the neural network solution near the boundaries and in refining the
                solution accuracy.
        
                Args:
                    self: The instance of the Points_type class containing the grid of points.
        
                Returns:
                    dict: A dictionary where keys are the points in the grid (NumPy arrays) and values are their types.
                          The type can be 'central' for interior points or a string of 'f' and 'b' characters for
                          boundary points. The length of the string corresponds to the dimension of the grid, where
                          'f' indicates that a small positive change in the corresponding coordinate keeps the point
                          within the solution space, and 'b' indicates that a small negative change does.
        """

        direction_list = []
        for axis in range(self.grid.shape[1]):
            for direction in range(2):
                direction_list.append(
                    Points_type._in_hull(Points_type.shift_points(
                     self.grid, axis, (-1) ** direction * 0.0001), self.grid))

        direction_list = np.array(direction_list)
        direction_list = np.transpose(direction_list)

        point_type = {}

        for i, point in enumerate(self.grid):
            if np.all(direction_list[i]):
                point_type[point] = 'central'
            else:
                p_type = ''
                j = 0
                while j < len(direction_list[i]):
                    if (j % 2 == 0 and direction_list[i, j]) or (
                            j % 2 == 0 and direction_list[i, j] and
                            direction_list[i, j + 1]):
                        p_type += 'f'
                    else:
                        p_type += 'b'
                    j += 2
                if self.grid.shape[-1] == 1:
                    point_type[point] = 'central'
                else:
                    point_type[point] = p_type
        return point_type

    def grid_sort(self) -> dict:
        """
        Sorts the grid points based on their classification, grouping them into subsets.
        
                This method is crucial for organizing the solution space, enabling targeted analysis and refinement of the neural network's approximation within specific regions of the problem domain. By segregating points based on their characteristics, the solver can better understand and address local variations in the differential equation's solution.
        
                Args:
                    self: An instance of the `Points_type` class containing the grid points and their properties.
        
                Returns:
                    dict: A dictionary where keys are the point types and values are tensors containing the corresponding grid points. This structured organization facilitates subsequent processing and analysis of the solution.
        """

        point_type = self.point_typization()
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

    def bnd_sort(self, grid_dict: dict, b_coord: Union[torch.Tensor, list]) -> list:
        """
        Sorts boundary points into a dictionary based on their correspondence to grid points.
        
                This function organizes boundary points by associating them with specific grid locations.
                This is useful for applying boundary conditions when solving differential equations using neural networks.
        
                Args:
                    grid_dict (dict): A dictionary where keys represent grid identifiers and values are tensors
                                     representing the coordinates of grid points.
                    b_coord (Union[torch.Tensor, list]): Boundary points. If the problem has periodic boundary
                                                          conditions, this will be a list of tensors, each
                                                          representing a boundary.
        
                Returns:
                    list: A list of dictionaries (or a single dictionary if 'b_coord' is a tensor). Each dictionary
                          maps grid identifiers to the corresponding boundary points located at those grid
                          locations. Returns a list of such dictionaries when dealing with periodic boundary
                          conditions (i.e., when 'b_coord' is a list).
        """

        def bnd_to_dict(grid_dict, b_coord):
            bnd_dict = {}
            for k, v in grid_dict.items():
                bnd_dict[k] = []
                for bnd in b_coord:
                    if ((bnd == v).all(axis=1)).any():
                        bnd_dict[k].append(bnd)
                if bnd_dict[k] == []:
                    del bnd_dict[k]
                else:
                    bnd_dict[k] = torch.stack(bnd_dict[k])
            return bnd_dict

        if isinstance(b_coord, list):
            bnd_dict_list = [bnd_to_dict(grid_dict, bnd) for bnd in b_coord]
            return bnd_dict_list
        else:
            return bnd_to_dict(grid_dict, b_coord)
