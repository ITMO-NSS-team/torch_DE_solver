""" Module for construct a domain with complex or irregular geometry"""
import torch
from abc import ABC, abstractmethod

class Shape(ABC):
    """
    Represents a generic shape with methods for checking point containment and boundary status.
    
        Class Methods:
        - contains:
                Returns a mask (N,) indicating which pts are inside the shape.
        - boundary:
                Returns a mask (N,) indicating which pts lie on the boundary of the shape.
    """

    @abstractmethod
    def contains(self, pts: torch.Tensor) -> torch.BoolTensor:
        """
        Returns a boolean mask indicating which points lie within the shape, a crucial step in evaluating the neural network's approximation of the differential equation's solution within a defined domain.
        
                Args:
                    pts (torch.Tensor): A tensor of shape (N, D) representing N points in D-dimensional space.
        
                Returns:
                    torch.BoolTensor: A boolean tensor of shape (N,) where each element indicates whether the corresponding point is inside the shape. This is used to determine where the neural network solution satisfies the boundary conditions or domain constraints of the differential equation.
        """
        ...

    @abstractmethod
    def boundary(self, pts: torch.Tensor, rtol: float = 1e-4, atol: float = 0.0) -> torch.BoolTensor:
        """
        Returns a boolean mask indicating which points lie on the boundary of the shape,
                useful for refining the solution of differential equations near boundaries.
        
                Args:
                    pts: A tensor of shape (N, D) representing the points to check, where N is the number of points and D is the dimensionality.
                    rtol: Relative tolerance.
                    atol: Absolute tolerance.
        
                Returns:
                    A boolean tensor of shape (N,) indicating whether each point lies on the boundary.
        """
        ...

class Rectangle(Shape):
    """
    Represents a rectangle in a multi-dimensional space.
    
        Class Methods:
        - __init__:
    """

    def __init__(self, lower, upper, dims=None):
        """
        Initializes a Rectangle object representing a spatial region.
        
        This rectangle is used to define a specific area within the problem domain
        where the solution is of interest. It's defined by lower and upper bounds
        across specified dimensions.
        
        Args:
            lower (sequence): Sequence of lower bounds for each spatial dimension.
            upper (sequence): Sequence of upper bounds for each spatial dimension.
            dims (sequence, optional): Indices of dimensions in the grid to apply the rectangle.
                Defaults to the first `len(lower)` dimensions.
        """
        self.lower = torch.as_tensor(lower, dtype=torch.float32)
        self.upper = torch.as_tensor(upper, dtype=torch.float32)
        self.dims = dims if dims is not None else list(range(self.lower.numel()))

    def _select(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Selects the dimensions of the input tensor relevant to the rectangle's definition.
        
                This operation is crucial for extracting the coordinates that define the rectangle
                from a more general tensor of points.
        
                Args:
                    pts: A tensor containing point coordinates, where each row represents a point.
        
                Returns:
                    torch.Tensor: A tensor containing only the coordinates corresponding
                    to the rectangle's dimensions.
        """
        return pts[:, self.dims]

    def contains(self, pts: torch.Tensor) -> torch.BoolTensor:
        """
        Checks if given points lie within the rectangle's boundaries. This is crucial for evaluating the neural network's solution within the defined problem space, ensuring that the approximated solution remains valid within the domain of the differential equation.
        
                Args:
                    pts: A tensor of shape (N, 2) representing the points to check, where N is the number of points.
        
                Returns:
                    A boolean tensor of shape (N,) indicating whether each point is contained within the rectangle.
        """
        sub = self._select(pts)
        return ((sub >= self.lower) & (sub <= self.upper)).all(dim=1)

    def boundary(self, pts: torch.Tensor, rtol: float = 1e-4, atol: float = 0.0) -> torch.BoolTensor:
        """
        Determines which points lie on the boundary of the defined rectangular domain.
        
                This method identifies points that are either inside or very close to the
                domain and have at least one coordinate near the domain's boundary. This
                is crucial for accurately representing the solution space of the differential
                equation being solved by the neural network. By identifying boundary points,
                the method helps in enforcing boundary conditions, which are essential for
                obtaining a unique and physically meaningful solution.
        
                Args:
                    pts: The points to check, with shape (N, D).
                    rtol: The relative tolerance parameter for `torch.isclose`.
                    atol: The absolute tolerance parameter for `torch.isclose`.
        
                Returns:
                    A boolean tensor of shape (N,) indicating which points are on the boundary.
        """
        sub = self._select(pts)
        inside_or_bound = self.contains(pts)
        on_lower = torch.isclose(sub, self.lower.unsqueeze(0), rtol=rtol, atol=atol).any(dim=1)
        on_upper = torch.isclose(sub, self.upper.unsqueeze(0), rtol=rtol, atol=atol).any(dim=1)
        return inside_or_bound & (on_lower | on_upper)

class Circle(Shape):
    """
    Represents a circle in Euclidean space.
    
        Class Methods:
        - __init__:
    """

    def __init__(self, center, radius, dims=None):
        """
        Initializes a circle in a specified number of dimensions. This circle is used to define regions of interest within the problem space when evaluating the neural network's solution to the differential equation.
        
                Args:
                    center (sequence): A sequence of coordinates representing the center of the circle in each spatial dimension.  These coordinates define a specific location within the problem domain around which the circle is constructed.
                    radius (scalar): The radius of the circle. This determines the size of the region influenced by the circle.
                    dims (list, optional):  A list of indices specifying the dimensions of the grid to which the circle should be applied. If None, defaults to the first `len(center)` dimensions. This allows the circle to be defined in a subset of the available dimensions, focusing the evaluation on specific spatial aspects of the solution.
        
                Returns:
                    None
        """
        self.center = torch.as_tensor(center, dtype=torch.float32)
        self.radius_sq = float(radius) ** 2
        self.dims = dims if dims is not None else list(range(self.center.numel()))

    def _select(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Selects the dimensions relevant to the circle from a given tensor.
        
                This selection is crucial for extracting the circle's parameters 
                (e.g., center coordinates, radius) from a larger set of data.
        
                Args:
                    pts: A tensor containing data points, where each row represents a point
                         and columns represent different dimensions.
        
                Returns:
                    torch.Tensor: A tensor containing only the columns (dimensions)
                                  specified by `self.dims`, which define the circle's
                                  representation within the input data.
        """
        return pts[:, self.dims]

    def contains(self, pts: torch.Tensor) -> torch.BoolTensor:
        """
        Checks if points lie within the circle's boundaries. This is crucial for evaluating the neural network's solution within the defined spatial domain of the differential equation.
        
                Args:
                    pts: A tensor of points with shape (N, 2) to check for containment.
        
                Returns:
                    A boolean tensor of shape (N,) indicating whether each point is inside the circle. This result is used to assess the accuracy of the neural network's approximation within the problem's domain.
        """
        sub = self._select(pts)
        sqd = ((sub - self.center) ** 2).sum(dim=1)
        return sqd <= self.radius_sq

    def boundary(self, pts: torch.Tensor, rtol: float = 1e-4, atol: float = 0.0) -> torch.BoolTensor:
        """
        Determines if points are located on the boundary of the sphere, which is crucial for verifying the accuracy of the neural network's solution to the differential equation.
        
                This method assesses whether a given set of points lie on the sphere's boundary by comparing their squared distance from the sphere's center to the squared radius, using specified relative and absolute tolerances. This is done to ensure that the neural network-based solution adheres to the geometric constraints defined by the sphere.
        
                Args:
                    pts: The points to check, represented as a `torch.Tensor`.
                    rtol: The relative tolerance for comparing squared distances. Defaults to 1e-4.
                    atol: The absolute tolerance for comparing squared distances. Defaults to 0.0.
        
                Returns:
                    torch.BoolTensor: A boolean tensor indicating whether each point is on the boundary.
        """
        sub = self._select(pts)
        sqd = ((sub - self.center) ** 2).sum(dim=1)
        return torch.isclose(sqd, torch.tensor(self.radius_sq, dtype=pts.dtype), rtol=rtol, atol=atol)

# CSG operations

def csg_difference(grid: torch.Tensor, shape: Shape) -> torch.Tensor:
    """
    Filters a grid of points, returning only those that do not satisfy the shape's constraints.
    
    This is useful for refining the solution space by excluding regions defined by the shape.
    
    Args:
        grid: (N, D) tensor of coordinates representing the points to be filtered.
        shape: A Shape instance defining the region to be excluded.
    
    Returns:
        (M, D) tensor containing the coordinates of points from the grid that lie outside the shape,
        where M <= N.
    """
    mask_outside = ~shape.contains(grid)
    return grid[mask_outside]


def csg_boundary(grid: torch.Tensor, shape: Shape, rtol: float = 1e-4, atol: float = 0.0) -> torch.Tensor:
    """
    Returns points of `grid` that lie on the boundary of the given `shape`.
    
        This is useful for identifying points where the solution to a differential equation, as represented by the `shape`, transitions or changes rapidly.
    
        Args:
            grid: (N, D) tensor of coordinates.
            shape: a Shape instance.
            rtol: relative tolerance for boundary detection.
            atol: absolute tolerance for boundary detection.
    
        Returns:
            torch.Tensor: A tensor containing the coordinates of the grid points that lie on the boundary of the shape.
    """
    mask_bnd = shape.boundary(grid, rtol=rtol, atol=atol)
    return grid[mask_bnd]




