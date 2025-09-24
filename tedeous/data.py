"""module for working with inerface for initialize grid, conditions and equation"""

from typing import List, Union, Dict
import torch
import numpy as np
import sys
import os

from tedeous.device import check_device
from tedeous.input_preprocessing import EquationMixin
from tedeous.data_CSG import csg_difference, csg_boundary, Circle, Rectangle


def tensor_dtype(dtype: str):
    """
    Converts a string representation of a data type to its corresponding PyTorch data type.
    
    This function ensures that the data type used within the neural network models
    for solving differential equations is correctly interpreted by PyTorch.
    
    Args:
        dtype (str): A string representing the desired data type (e.g., 'float32', 'float64', 'float16').
    
    Returns:
        torch.dtype: The corresponding PyTorch data type.
    """
    if dtype == 'float32':
        dtype = torch.float32
    elif dtype == 'float64':
        dtype = torch.float64
    elif dtype == 'float16':
        dtype = torch.float16

    return dtype


class Domain():
    """
    class for grid building
    """


    def __init__(self, type='uniform'):
        """
        Initializes a new Domain instance.
        
                This method sets up the domain by initializing its type and preparing a dictionary to hold variables relevant to the differential equation being solved. The type parameter allows for specifying different domain configurations, influencing how the solution space is explored. The variable dictionary will store necessary parameters and initial conditions for the differential equation.
        
                Args:
                    type (str): Specifies the type of domain (e.g., 'uniform'). Different types may influence sampling strategies or domain-specific constraints. Defaults to 'uniform'.
        
                Returns:
                    None.
        """
        self.type = type
        self.variable_dict = {}

    def variable(
            self,
            variable_name: str,
            variable_set: Union[List, torch.Tensor],
            n_points: Union[None, int],
            dtype: str = 'float32') -> None:
        """
        Initializes a spatial variable for defining the domain.
        
        This method creates a tensor representing the spatial variable, which is then stored in the domain's variable dictionary.
        This tensor is used to define the domain over which the differential equation is solved.
        The variable can be defined either by providing a range and number of points for uniform discretization,
        or by directly providing a tensor of points.
        
        Args:
            variable_name (str): Name of the spatial variable.
            variable_set (Union[List, torch.Tensor]): Either a list [start, stop] defining the range for the variable,
                                                      or a torch.Tensor containing the specific points for the variable.
            n_points (int): Number of points to discretize the variable into if variable_set is a range.  Ignored if variable_set is a Tensor.
            dtype (str, optional): Data type of the resulting tensor. Defaults to 'float32'.
        
        Returns:
            None: The method updates the internal variable dictionary of the domain object.
        """
        dtype = tensor_dtype(dtype)

        if isinstance(variable_set, torch.Tensor):
            variable_tensor = check_device(variable_set)
            variable_tensor = variable_set.to(dtype)
            self.variable_dict[variable_name] = variable_tensor
        else:
            if self.type == 'uniform':
                n_points = n_points + 1
                start, end = variable_set
                variable_tensor = torch.linspace(start, end, n_points, dtype=dtype)
                self.variable_dict[variable_name] = variable_tensor

    def build(self,
              mode: str,
              removed_domains: list = None) -> torch.Tensor:
        """
        Builds the computational grid based on the specified mode and domain.
        
                This method constructs the grid used for solving differential equations.
                The grid is generated differently depending on the chosen mode ('autograd', 'NN', or 'mat').
                For 'autograd' and 'NN' modes, it creates a Cartesian product of the variables,
                optionally removing specified domains using constructive solid geometry (CSG) operations.
                For the 'mat' mode, it uses `np.meshgrid` to create the grid. The resulting grid is then
                converted to the appropriate device using `check_device`.
        
                Args:
                    mode (str): The mode for equation solution ('mat', 'autograd', or 'NN'). This determines the grid generation method.
                    removed_domains (list, optional): A list of dictionaries, where each dictionary describes a domain to be removed from the grid. Defaults to None.
        
                Returns:
                    torch.Tensor: The resulting computational grid, ready for use in the differential equation solver.
        """
        var_lst = list(self.variable_dict.values())
        var_lst = [i.cpu() for i in var_lst]

        if mode in ('autograd', 'NN'):
            if len(self.variable_dict) == 1:
                grid = check_device(var_lst[0].reshape(-1, 1))
            else:
                grid = check_device(torch.cartesian_prod(*var_lst))

                if removed_domains is not None:
                    for domain_dict in removed_domains:
                        figure_type = list(domain_dict.keys())[0]
                        if figure_type == 'rectangle':
                            coords_min = domain_dict[figure_type]['coords_min']
                            coords_max = domain_dict[figure_type]['coords_max']
                            shape = Rectangle(coords_min,coords_max)
                        elif figure_type == 'circle':
                            center = domain_dict[figure_type]['center']
                            radius = domain_dict[figure_type]['radius']
                            shape = Circle(center, radius)

                        grid = csg_difference(grid, shape).detach().clone()
        else:
            grid = np.meshgrid(*var_lst, indexing='ij')
            grid = check_device(grid)

        grid = check_device(grid)
        return grid


class Conditions():
    """
    class for adding the conditions: initial, boundary, and data.
    """


    def __init__(self):
        """
        Initializes a new instance of the Conditions class.
        
                The conditions list is initialized to store boundary or initial conditions that constrain the solution space of the differential equation. These conditions are essential for training the neural network to approximate the specific solution of interest.
        
                Args:
                    self: The object instance.
        
                Returns:
                    None.
        
                Class Fields:
                    conditions_lst (list): A list to store conditions.
        """
        self.conditions_lst = []

    def dirichlet(
            self,
            bnd: Union[torch.Tensor, dict],
            value: Union[callable, torch.Tensor, float],
            var: int = 0):
        """
        Determines a Dirichlet boundary condition for the differential equation. This condition enforces a specific value at the boundary, guiding the neural network to learn solutions that satisfy the given constraints.
        
                Args:
                    bnd (Union[torch.Tensor, dict]): Boundary points where the Dirichlet condition is applied. Can be a torch.Tensor or a dictionary with coordinate names as keys and coordinate values as values.
                    value (Union[callable, torch.Tensor, float]): The value(s) at the boundary (bnd). If callable, it's a function that takes `bnd` as input.
                    var (int, optional): Variable index for systems of equations. Defaults to 0 for single equations.
        
                Returns:
                    None: The method appends the defined Dirichlet boundary condition to the internal list of conditions, which will be used during the training process to ensure the neural network solution adheres to the specified boundary constraints.
        """

        self.conditions_lst.append({'bnd': bnd,
                                    'bop': None,
                                    'bval': value,
                                    'var': var,
                                    'type': 'dirichlet'})

    def operator(self,
                 bnd: Union[torch.Tensor, dict],
                 operator: dict,
                 value: Union[callable, torch.Tensor, float]):
        """
        Adds an operator boundary condition to the list of conditions. This condition specifies a relationship involving a differential operator that must be satisfied on a given boundary. This is crucial for accurately representing the problem's constraints when solving differential equations using neural networks.
        
                Args:
                    bnd (Union[torch.Tensor, dict]): Boundary points where the condition applies. Can be a tensor or a dictionary with coordinate names as keys and coordinate values as values.
                    operator (dict): A dictionary defining the differential operator. It contains terms with coefficients, derivatives, powers, and variables.
                    value (Union[callable, torch.Tensor, float]): The value of the operator on the boundary. Can be a constant, a tensor, or a callable function that takes the boundary points as input.
        
                Returns:
                    None: This method adds the boundary condition to the internal list of conditions.
        """
        try:
            var = operator[operator.keys()[0]]['var']
        except:
            var = 0
        operator = EquationMixin.equation_unify(operator)
        self.conditions_lst.append({'bnd': bnd,
                                    'bop': operator,
                                    'bval': value,
                                    'var': var,
                                    'type': 'operator'})

    def periodic(self,
                 bnd: Union[List[torch.Tensor], List[dict]],
                 operator: dict = None,
                 var: int = 0):
        """
        Adds a periodic boundary condition to the problem definition. This ensures that the solution exhibits a repeating pattern across the domain boundaries. This is achieved by equating the solution or its derivatives at corresponding points on the boundary.
        
                Args:
                    bnd (Union[List[torch.Tensor], List[dict]]):  A list specifying the boundary locations. Can be a list of tensors representing coordinates or a list of dictionaries defining boundary regions.
                    operator (dict, optional): A dictionary defining the differential operator for the periodic condition. If None, a Dirichlet-type periodic condition is applied (i.e., the solution values are equated). Defaults to None.
                    var (int, optional): The index of the variable to which the periodic condition applies (for systems of equations). Defaults to 0.
        
                Returns:
                    None. The periodic condition is added to the internal list of conditions.
        
                Why:
                    This method enforces periodicity in the solution, which is a crucial characteristic for many physical systems and ensures that the neural network learns solutions that respect this fundamental property.
        """
        value = torch.tensor([0.])
        if operator is None:
            self.conditions_lst.append({'bnd': bnd,
                                        'bop': operator,
                                        'bval': value,
                                        'var': var,
                                        'type': 'periodic'})
        else:
            try:
                var = operator[operator.keys()[0]]['var']
            except:
                var = 0
            operator = EquationMixin.equation_unify(operator)
            self.conditions_lst.append({'bnd': bnd,
                                        'bop': operator,
                                        'bval': value,
                                        'var': var,
                                        'type': 'periodic'})

    def robin(self,
              bnd: Union[torch.Tensor, dict],
              value: Union[callable, torch.Tensor, float],
              operator: Dict = None,
              var: int = 0):
        """
        Defines a Robin boundary condition for the differential equation.
        
                This condition constrains a combination of the solution and its derivative
                on the boundary of the domain. This is particularly useful when modeling
                heat transfer with convection or when specifying impedance boundary conditions
                in electromagnetics.
        
                Args:
                    bnd: Boundary points where the condition is applied (torch.Tensor or dict).
                    value: The right-hand side value of the Robin condition. It can be a constant,
                           a function, or a tensor (callable, Tensor, or float).
                    operator: A dictionary defining the differential operator involved in the
                              Robin condition. Defaults to None, in which case a default operator
                              might be applied.
                    var: The variable index for systems of equations (default: 0). Specifies
                         which component of the solution vector this condition applies to.
        
                Returns:
                    None. The method appends the Robin boundary condition to the internal list
                    of conditions (`self.conditions_lst`) for later use in the solution process.
        """

        operator = EquationMixin.equation_unify(operator)
        self.conditions_lst.append({'bnd': bnd,
                                    'bop': operator,
                                    'bval': value,
                                    'var': var,
                                    'type': 'robin'})

    def data(
            self,
            bnd: Union[torch.Tensor, dict],
            operator: Union[dict, None],
            value: torch.Tensor,
            var: int = 0):
        """
        Registers known solution data as a boundary condition.
        
        This method stores the provided data, which represents known values of the solution at specific locations,
        as a boundary condition. This information is crucial for guiding the neural network to learn the correct
        solution by enforcing that the approximation matches the known values at these boundaries.
        
        Args:
            bnd (Union[torch.Tensor, dict]): Boundary points where the solution is known. Can be a tensor or a dictionary
                with coordinate names as keys and coordinate values as values.
            operator (Union[dict, None]): Dictionary defining the differential operator. Each entry represents a term
                in the operator, specified by coefficient, term, power, and variable. Can be None if no operator
                is involved in the boundary condition.
            value (torch.Tensor): Known values of the solution at the boundary points specified in `bnd`.
            var (int, optional): Index of the variable for system cases or periodic Dirichlet conditions. Defaults to 0.
        
        Returns:
            None
        """
        if operator is not None:
            operator = EquationMixin.equation_unify(operator)
        self.conditions_lst.append({'bnd': bnd,
                                    'bop': operator,
                                    'bval': value,
                                    'var': var,
                                    'type': 'data'})

    def _bnd_grid(self,
                  bnd: Union[torch.Tensor, dict],
                  variable_dict: dict,
                  dtype) -> torch.Tensor:
        """
        Constructs a subgrid tailored to the specified boundary conditions.
        
        This method generates a refined grid focusing on the boundaries
        defined within the problem, allowing for a more accurate approximation
        of the solution near these critical regions. It adapts to different
        types of boundary specifications, including tensor-based and
        dictionary-based definitions, to create a grid that precisely
        represents the boundary conditions. This is crucial for accurately
        training the neural network to satisfy the constraints imposed by
        the differential equation at the boundaries.
        
        Args:
            bnd (Union[torch.Tensor, dict]): Boundary points, which can be a
                torch.Tensor or a dictionary. If a dictionary, keys represent
                coordinate names, and values represent coordinate values.
            variable_dict (dict): A dictionary containing torch.Tensors for
                each domain variable.
            dtype (dtype): The desired data type for the grid.
        
        Returns:
            torch.Tensor: A subgrid representing the boundary conditions.
        """

        dtype = variable_dict[list(variable_dict.keys())[0]].dtype

        coords = [variable_dict[var] for var in variable_dict.keys()]
        grid = torch.cartesian_prod(*coords)

        if isinstance(bnd, torch.Tensor):
            bnd_grid = bnd.to(dtype)
        else:
            var_lst = []

            if list(bnd.keys())[0] == 'circle':
                shape = Circle(bnd['circle']['center'],bnd['circle']['radius'])
                result = csg_boundary(grid, shape)
                return result

            for var in variable_dict.keys():
                if isinstance(bnd[var], torch.Tensor):
                    var_lst.append(check_device(bnd[var]).to(dtype))
                elif isinstance(bnd[var], (float, int)):
                    var_lst.append(check_device(torch.tensor([bnd[var]])).to(dtype))
                elif isinstance(bnd[var], list):
                    lower_bnd = bnd[var][0]
                    upper_bnd = bnd[var][1]
                    grid_var = variable_dict[var]
                    bnd_var = grid_var[(grid_var >= lower_bnd) & (grid_var <= upper_bnd)]
                    var_lst.append(check_device(bnd_var).to(dtype))
            bnd_grid = torch.cartesian_prod(*var_lst).to(dtype)
        if len(bnd_grid.shape) == 1:
            bnd_grid = bnd_grid.reshape(-1, 1)
        return bnd_grid

    def build(self,
              variable_dict: dict) -> List[dict]:
        """
        Processes boundary conditions to prepare them for neural network training.
        
        This method prepares boundary conditions by converting them into a format suitable
        for use with neural network models. It handles different types of boundary conditions,
        including periodic boundaries, and ensures that boundary values are represented as
        PyTorch tensors with the correct data type and device. This preprocessing step is
        crucial for ensuring that the boundary conditions are properly enforced during the
        training process, leading to accurate solutions of the differential equation.
        
        Args:
            variable_dict (dict): A dictionary containing the domain variables as torch.Tensors.
        
        Returns:
            List[dict]: A list of dictionaries, where each dictionary contains information
                        about a boundary condition, including the boundary points and values,
                        formatted as torch.Tensors. Returns None if no conditions are present.
        """
        if self.conditions_lst == []:
            return None

        try:
            dtype = variable_dict[list(variable_dict.keys())[0]].dtype
        except:
            dtype = variable_dict[list(variable_dict.keys())[0]][0].dtype  # if periodic

        for cond in self.conditions_lst:
            if cond['type'] == 'periodic':
                cond_lst = []
                for bnd in cond['bnd']:
                    cond_lst.append(self._bnd_grid(bnd, variable_dict, dtype))
                cond['bnd'] = cond_lst
            else:
                cond['bnd'] = self._bnd_grid(cond['bnd'], variable_dict, dtype)

            if isinstance(cond['bval'], torch.Tensor):
                cond['bval'] = check_device(cond['bval']).to(dtype)
            elif isinstance(cond['bval'], (float, int)):
                cond['bval'] = check_device(
                    torch.ones_like(cond['bnd'][:, 0]) * cond['bval']).to(dtype)
            elif callable(cond['bval']):
                cond['bval'] = check_device(cond['bval'](cond['bnd'])).to(dtype)

        return self.conditions_lst


class Equation():
    """
    class for adding eqution.
    """


    def __init__(self):
        """
        Initializes the EquationList object.
        
        The EquationList is used to store and manage a collection of equations that define the differential equation system.
        This initialization creates an empty list ready to hold these equation objects, which are then used to define the neural network model.
        
        Args:
            self: The object instance.
        
        Returns:
            None
        """
        self.equation_lst = []

    def add(self, eq: dict):
        """
        Adds a differential equation to the list of equations to be solved.
        
        Args:
            eq (dict): A dictionary representing the differential equation in operator form.
        
        Returns:
            None
        
        Why:
            This method allows users to define a system of differential equations that the neural network will learn to solve. By appending the equation to the internal list, it becomes part of the overall problem definition for the solver.
        """
        self.equation_lst.append(eq)


