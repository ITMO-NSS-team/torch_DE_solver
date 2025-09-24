"""preprocessing module for operator (equation) and boundaries.
"""

from copy import deepcopy
from typing import Union, Callable
import numpy as np
import torch

from tedeous.points_type import Points_type
from tedeous.finite_diffs import Finite_diffs
from tedeous.device import check_device


def lambda_prepare(val: torch.Tensor,
                   lambda_: Union[int, list, torch.Tensor],
                   dtype: torch.dtype = None) -> torch.Tensor:
    """
    Prepares regularization parameters to match the dimensions of the equation or boundary condition terms. This ensures that each term in the loss function is appropriately weighted during the training process.
    
        Args:
            val (torch.Tensor): Operator tensor or boundary value tensor, representing the terms in the equation or boundary condition.
            lambda_ (Union[int, list, torch.Tensor]): Regularization parameter(s). Can be a single value, a list of values, or a tensor.
            dtype (torch.dtype, optional): The desired data type for the regularization parameters. Defaults to None, inferring from the input.
    
        Returns:
            torch.Tensor: A tensor containing the prepared regularization parameters, reshaped to have dimensions (1, number of terms in val).
    """

    if isinstance(lambda_, torch.Tensor):
        return lambda_

    if isinstance(lambda_, (int, float)):
        try:
            lambdas = torch.ones(val.shape[-1]) * lambda_
        except:
            lambdas = torch.tensor(lambda_)
    elif isinstance(lambda_, list):
        lambdas = torch.tensor(lambda_)

    if dtype:
        lambdas = torch.tensor(lambdas, dtype=dtype)

    return lambdas.reshape(1, -1)


class EquationMixin:
    """
    Auxiliary class. This one contains some methods that uses in other classes.
    """


    @staticmethod
    def equation_unify(equation: dict) -> dict:
        """
        Unifies the equation format to ensure compatibility with the neural network solver.
        
                This method standardizes the representation of differential equation operators
                by ensuring the presence of 'var' and converting 'pow' and other directional
                variables to lists when necessary. This ensures that the equation is in a
                format suitable for processing by the neural network-based solver. It is
                necessary to bring all equations to the same format before passing them
                to the solver.
        
                Args:
                    equation (dict): A dictionary representing the differential equation in its initial form.
        
                Returns:
                    dict: A dictionary representing the differential equation with unified formatting for use with the solver.
        """

        for operator_label in equation.keys():
            operator = equation[operator_label]
            dif_dir = list(operator.keys())[1]
            try:
                operator['var']
            except:
                if isinstance(operator['pow'], (int, float, Callable)):
                    operator[dif_dir] = [operator[dif_dir]]
                    operator['pow'] = [operator['pow']]
                    operator['var'] = [0]
                elif isinstance(operator['pow'], list):
                    operator['var'] = [0 for _ in operator['pow']]
                continue
            if isinstance(operator['pow'], (int, float, Callable)):
                operator[dif_dir] = [operator[dif_dir]]
                operator['pow'] = [operator['pow']]
                operator['var'] = [operator['var']]

        return equation

    @staticmethod
    def closest_point(grid: torch.Tensor, target_point: float) -> int:
        """
        Locates the grid point closest to a specified boundary point.
        
        This function is crucial for accurately representing boundary conditions
        when solving differential equations using neural networks. By identifying
        the nearest grid point to the boundary, the solver can effectively
        enforce the given constraints, leading to more accurate solutions.
        
        Args:
            grid (torch.Tensor): A tensor representing the discretization of the domain.
            target_point (float): The coordinate of the boundary point.
        
        Returns:
            int: The index of the grid point closest to the target point.
        """

        min_dist = np.inf
        pos = 0
        min_pos = 0
        for point in grid:
            dist = torch.linalg.norm(point - target_point)
            if dist < min_dist:
                min_dist = dist
                min_pos = pos
            pos += 1
        return min_pos

    @staticmethod
    def convert_to_double(bnd: Union[list, np.array]) -> float:
        """
        Converts input data (either a list or a NumPy array) to a double-precision PyTorch tensor.
        
                Args:
                    bnd (Union[list, np.array]): The input data, which can be a list of arrays or a NumPy array, representing points or boundaries.
        
                Returns:
                    torch.Tensor: A double-precision PyTorch tensor representing the converted input data. If the input is a list, the function recursively converts each element of the list.
        
                Why:
                    This conversion is essential for ensuring numerical stability and precision when solving differential equations using neural networks in PyTorch. Double-precision tensors provide a higher level of accuracy during computations, which is crucial for the convergence and reliability of the neural network-based solver.
        """

        if isinstance(bnd, list):
            for i, cur_bnd in enumerate(bnd):
                bnd[i] = EquationMixin.convert_to_double(cur_bnd)
            return bnd
        elif isinstance(bnd, np.ndarray):
            return torch.from_numpy(bnd).double()
        return bnd.double()

    @staticmethod
    def search_pos(grid: torch.Tensor, bnd) -> list:
        """
        Identifies the indices in the grid that correspond to the given boundary conditions.
        
        This method is crucial for mapping the boundary conditions of the differential equation
        onto the discrete grid used by the neural network solver. It ensures that the boundary
        conditions are accurately enforced during the training process. If exact matches are not
        found, it identifies the closest grid points to approximate the boundary conditions.
        
        Args:
            grid (torch.Tensor): A tensor representing the spatial or temporal grid on which the differential equation is discretized.
            bnd (torch.Tensor or list): The boundary condition(s) to locate on the grid. Can be a single point or a list of points.
        
        Returns:
            list: A list of integer indices representing the positions of the boundary conditions on the grid.
                  If `bnd` is a list, it returns a nested list of positions.
        """

        if isinstance(bnd, list):
            for i, cur_bnd in enumerate(bnd):
                bnd[i] = EquationMixin.search_pos(grid, cur_bnd)
            return bnd
        pos_list = []
        for point in bnd:
            try:
                pos = int(torch.where(torch.all(
                    torch.isclose(grid, point), dim=1))[0])
            except Exception:
                pos = EquationMixin.closest_point(grid, point)
            pos_list.append(pos)
        return pos_list

    @staticmethod
    def bndpos(grid: torch.Tensor, bnd: torch.Tensor) -> Union[list, int]:
        """
        Returns the indices of grid points that correspond to boundary conditions. This is crucial for enforcing solution constraints at the edges of the domain when solving differential equations using neural networks.
        
                Args:
                    grid (torch.Tensor): The computational grid represented as a tensor. Each point in the grid is a potential location for applying boundary conditions.
                    bnd (torch.Tensor): Tensor containing the values of the boundary conditions.
        
                Returns:
                    Union[list, int]: A list of indices representing the positions on the grid where the boundary conditions are applied. Returns integer if boundary conditions not found.
        """

        if grid.shape[0] == 1:
            grid = grid.reshape(-1, 1)
        grid = grid.double()
        bnd = EquationMixin.convert_to_double(bnd)
        bndposlist = EquationMixin.search_pos(grid, bnd)
        return bndposlist


class Equation_NN(EquationMixin, Points_type):
    """
    Class for preprocessing input data: grid, operator, bconds in unified
        form. Then it will be used for determine solution by 'NN' method.
    """


    def __init__(self,
                 grid: torch.Tensor,
                 operator:  Union[dict, list],
                 bconds: list,
                 h: float = 0.001,
                 inner_order: str = '1',
                 boundary_order: str = '2'):
        """
        Prepares the problem setup for solving differential equations using a neural network. It initializes the equation, boundary conditions, and discretization parameters necessary for training the neural network to approximate the solution.
        
                Args:
                    grid (torch.Tensor): Tensor representing the spatial or temporal domain where the solution is sought.
                    operator (Union[dict, list]): Definition of the differential equation to be solved.
                    bconds (list): Boundary conditions that constrain the solution space.
                    h (float, optional): Discretization parameter (step size) used in finite difference approximations. Defaults to 0.001.
                    inner_order (str, optional): Accuracy order for the finite difference scheme within the domain. Defaults to '1'.
                    boundary_order (str, optional): Accuracy order for the finite difference scheme at the boundaries. Defaults to '2'.
        
                Returns:
                    None
        """

        super().__init__(grid)
        self.grid = grid
        self.operator = operator
        self.bconds = bconds
        self.h = h
        self.inner_order = inner_order
        self.boundary_order = boundary_order

    def _operator_to_type_op(self,
                            dif_direction: list,
                            nvars: int,
                            axes_scheme_type: str) -> list:
        """
        Converts a symbolic differentiation operator into a concrete finite difference scheme tailored for neural network-based differential equation solving.
        
                This function translates a symbolic representation of a derivative (e.g., d2/dx2) into a numerical approximation suitable for use in training a neural network to solve the differential equation. It selects an appropriate finite difference scheme based on the desired accuracy order and the location within the domain (interior or boundary).
        
                Args:
                    dif_direction (list): Differentiation direction, represented as a list of lists (e.g., `[[0, 0]]` for d2/dx2).
                    nvars (int): Dimensionality of the problem (number of independent variables).
                    axes_scheme_type (str): Type of finite difference scheme to use ('central' for central difference, or a combination of 'f' and 'b' for forward/backward differences, typically used at boundaries).
        
                Returns:
                    list: A list containing two lists:
                        - The first list contains the finite difference schemes corresponding to each term in the differentiation direction. Each scheme is represented as a list of steps and signs.
                        - The second list contains the orders of accuracy corresponding to each finite difference scheme.
        
                Why:
                    This conversion is crucial for translating the symbolic form of the differential equation into a form that can be evaluated numerically by the neural network. The choice of finite difference scheme affects the accuracy and stability of the solution, and this function allows for different schemes to be used in different parts of the domain to optimize the solution process.
        """
        if axes_scheme_type == 'central':
            scheme_variant = self.inner_order
        else:
            scheme_variant = self.boundary_order

        fin_diff_list = []
        s_order_list = []
        for term in dif_direction:
            scheme, s_order = Finite_diffs(
                term, nvars, axes_scheme_type).scheme_choose(
                scheme_variant, h=self.h)
            fin_diff_list.append(scheme)
            s_order_list.append(s_order)
        return [fin_diff_list, s_order_list]

    def _finite_diff_scheme_to_grid_list(self,
                                        finite_diff_scheme: list,
                                        grid_points: torch.Tensor) -> list:
        """
        Converts a finite difference scheme to a list of shifted grid points.
        
                This method transforms the finite difference representation of a term
                into a set of grid points shifted according to the scheme. This is
                crucial for evaluating the differential equation using the neural network
                approximation by providing the locations where the solution needs to be
                evaluated.
        
                Args:
                    finite_diff_scheme (list): A list representing the finite difference
                        scheme for a single term, where each element corresponds to shifts
                        along different axes.
                    grid_points (torch.Tensor): The original grid points that will be
                        shifted according to the finite difference scheme.
        
                Returns:
                    list: A list of torch.Tensor, where each tensor contains the grid
                        points shifted according to the corresponding element in the
                        finite_diff_scheme.
        """

        s_grid_list = []
        for shifts in finite_diff_scheme:
            if shifts is None:
                s_grid_list.append(grid_points)
            else:
                s_grid = grid_points
                for j, axis in enumerate(shifts):
                    s_grid = self.shift_points(s_grid, j, axis * self.h)
                s_grid_list.append(s_grid)
        return s_grid_list

    def _checking_coeff(self,
                       coeff: Union[int, float, torch.Tensor, callable],
                       grid_points: torch.Tensor) -> torch.Tensor:
        """
        Checks and prepares the coefficient for use in the neural network-based differential equation solver. The method ensures that the coefficient is in a suitable format (torch.Tensor) for efficient computation within the neural network.
        
                Args:
                    coeff (Union[int, float, torch.Tensor, callable]): The coefficient in the differential equation. It can be a constant (int, float), a tensor, or a callable function.
                    grid_points (torch.Tensor): The grid points at which the coefficient needs to be evaluated if it's a callable or a tensor.
        
                Raises:
                    NameError: If the provided `coeff` is not of the allowed types (int, float, torch.Tensor, or callable).
        
                Returns:
                    torch.Tensor: The processed coefficient, ready for use in the neural network. If the input is a constant, it remains a constant. If it's a callable, it returns a tuple of callable and grid_points. If it's a tensor, it returns a tensor reshaped based on grid points.
        
                Why:
                    This method ensures that all coefficients, regardless of their initial type, are converted into a format that can be efficiently processed by the neural network, which is crucial for accurate and fast solving of differential equations.
        """

        if isinstance(coeff, (int, float)):
            coeff1 = coeff
        elif callable(coeff):
            coeff1 = (coeff, grid_points)
        elif isinstance(coeff, torch.Tensor):
            coeff = check_device(coeff)
            pos = self.bndpos(self.grid, grid_points)
            coeff1 = coeff[pos].reshape(-1, 1)
        elif isinstance(coeff, torch.nn.parameter.Parameter):
            coeff1 = coeff
        else:
            raise NameError('"coeff" should be: torch.Tensor or callable or int or float!')
        return coeff1

    def _type_op_to_grid_shift_op(self, fin_diff_op: list, grid_points) -> list:
        """
        Converts a finite difference operator for a specific grid type into a grid-shifted operator, preparing it for neural network-based differential equation solving. This involves mapping coefficients (which can be integers, functions, or arrays) to the appropriate subgrid points. This conversion is crucial for aligning the finite difference scheme with the neural network's grid representation, enabling the network to learn the solution effectively.
        
                Args:
                    fin_diff_op (list): The finite difference operator for a specific grid type.
                    grid_points (list): The grid points associated with the finite difference scheme.
        
                Returns:
                    list: A list of grid-shifted operators, ready for use in the neural network solver.
        """

        shift_grid_op = []
        for term1 in fin_diff_op:
            grid_op = self._finite_diff_scheme_to_grid_list(term1, grid_points)
            shift_grid_op.append(grid_op)
        return shift_grid_op

    def _one_operator_prepare(self,
                             operator: dict,
                             grid_points: torch.Tensor,
                             points_type: str) -> dict:
        """
        Prepares a single operator term for use in the neural network-based differential equation solver.
        
                This method standardizes the operator's structure, converts coefficient
                functions to appropriate tensor representations, and transforms
                differential operators into grid-shift operations suitable for
                evaluation on the computational grid. This ensures that each term in
                the differential equation is correctly represented and can be efficiently
                evaluated by the neural network.
        
                Args:
                    operator (dict): A dictionary representing the operator term,
                        containing the coefficient and differential operator.
                    grid_points (torch.Tensor): The coordinates of the grid points
                        where the solution is evaluated.
                    points_type (str): The type of grid points (e.g., 'uniform',
                        'random').
        
                Returns:
                    dict: The prepared operator term, with standardized structure,
                        tensor-based coefficients, and grid-shift differential
                        operators.
        """

        nvars = self.grid.shape[-1]
        operator = self.equation_unify(operator)
        for operator_label in operator:
            term = operator[operator_label]
            dif_term = list(term.keys())[1]
            term['coeff'] = self._checking_coeff(term['coeff'], grid_points)
            term[dif_term] = self._operator_to_type_op(term[dif_term],
                                                      nvars, points_type)
            term[dif_term][0] = self._type_op_to_grid_shift_op(
                term[dif_term][0], grid_points)
        return operator

    def operator_prepare(self) -> list:
        """
        Prepares the differential operator for the neural network solver.
        
                This method adapts the operator based on the problem setup,
                handling both single-equation and multi-equation systems.
                It ensures the operator is compatible with the chosen grid points
                before the neural network processes it.
        
                Args:
                    self: The instance of the Equation_NN class.
        
                Returns:
                    list: A list of dictionaries, where each dictionary represents a prepared
                          differential operator for a corresponding equation in the system.
                          If it is a single equation, the list contains only one dictionary.
        """

        grid_points = self.grid_sort()['central']
        if isinstance(self.operator, list) and isinstance(self.operator[0], dict):
            num_of_eq = len(self.operator)
            prepared_operator = []
            for i in range(num_of_eq):
                equation = self._one_operator_prepare(
                    self.operator[i], grid_points, 'central')
                prepared_operator.append(equation)
        else:
            equation = self._one_operator_prepare(
                self.operator, grid_points, 'central')
            prepared_operator = [equation]

        return prepared_operator

    def _apply_bnd_operators(self, bnd_operator: dict, bnd_dict: dict) -> list:
        """
        Applies the boundary operator to specific boundary points to prepare the equation for neural network-based solution.
        
                This method iterates through different types of boundary points and applies the corresponding boundary operator to each.
                This prepares the boundary conditions to be incorporated into the neural network's training process, guiding it towards
                solutions that satisfy the given constraints.
        
                Args:
                    bnd_operator (dict): Boundary operator in symbolic form, ready for numerical evaluation.
                    bnd_dict (dict): Dictionary where keys are point types and values are the corresponding boundary points.
        
                Returns:
                    list: A list of prepared equations, each representing the boundary condition applied to a specific subset of grid points.
        """

        operator_list = []
        for points_type in list(bnd_dict.keys()):
            equation = self._one_operator_prepare(
                deepcopy(bnd_operator), bnd_dict[points_type], points_type)
            operator_list.append(equation)
        return operator_list

    def bnd_prepare(self) -> list:
        """
        Prepares boundary conditions for use in the neural network-based differential equation solver. This involves sorting the grid and applying boundary operators to ensure compatibility with the network's input requirements.
        
                Args:
                    self: Instance of the Equation_NN class, containing the grid and boundary condition data.
        
                Returns:
                    list: A list of dictionaries, where each dictionary represents a boundary condition, processed and ready for use in the neural network solver. The boundary operators are applied to these conditions to align them with the network's expected input format, facilitating accurate solution approximation.
        """

        grid_dict = self.grid_sort()

        for bcond in self.bconds:
            bnd_dict = self.bnd_sort(grid_dict, bcond['bnd'])
            if bcond['bop'] is not None:
                if bcond['type'] == 'periodic':
                    bcond['bop'] = [self._apply_bnd_operators(
                        bcond['bop'], i) for i in bnd_dict]
                else:
                    bcond['bop'] = self._apply_bnd_operators(
                        bcond['bop'], bnd_dict)
        return self.bconds


class Equation_autograd(EquationMixin):
    """
    Prepares equation for autograd method (i.e., from conventional form to input form).
    """


    def __init__(self,
                 grid: torch.Tensor,
                 operator: Union[dict, list],
                 bconds: list):
        """
        Prepares the differential equation for the neural network-based solver. It stores the grid, operator, and boundary conditions for subsequent use in the training process. This setup is crucial for defining the loss function, which quantifies how well the neural network's output satisfies the given differential equation and boundary conditions.
        
                Args:
                    grid (torch.Tensor): Tensor representing the spatial or temporal grid where the solution is sought.
                    operator (Union[dict, list]): Definition of the differential equation to be solved.
                    bconds (list): Boundary conditions that constrain the solution space.
        """

        self.grid = grid
        self.operator = operator
        self.bconds = bconds

    def _checking_coeff(self,
                       coeff: Union[int, float, torch.Tensor]) -> Union[int, float, torch.Tensor]:
        """
        Validates and prepares the coefficient for use in the differential equation.
        
        This method ensures that the provided coefficient is of a supported type (int, float, torch.Tensor, or callable) and prepares it for use within the neural network-based differential equation solver.  It reshapes torch.Tensor coefficients to a column vector to ensure compatibility with subsequent operations.
        
        Args:
            coeff (Union[int, float, torch.Tensor]): The coefficient to be checked and prepared.
        
        Raises:
            NameError: If the coefficient is not of a supported type (int, float, torch.Tensor, or callable).
        
        Returns:
            Union[int, float, torch.Tensor]: The validated and potentially reshaped coefficient.
        """

        if isinstance(coeff, (int, float)):
            coeff1 = coeff
        elif callable(coeff):
            coeff1 = coeff
        elif isinstance(coeff, torch.Tensor):
            coeff = check_device(coeff)
            coeff1 = coeff.reshape(-1, 1)
        elif isinstance(coeff, torch.nn.parameter.Parameter):
            coeff1 = coeff
        else:
            raise NameError('"coeff" should be: torch.Tensor or callable or int or float!')
        return coeff1

    def _one_operator_prepare(self, operator: dict) -> dict:
        """
        Prepares a single operator by unifying its equation form and checking the coefficients of its terms.
        
        This method ensures that the operator is in a consistent format suitable for neural network-based differential equation solving.
        It unifies the equation representation and verifies the validity of the coefficients, ensuring numerical stability and correctness during the training process.
        
        Args:
            operator (dict): The operator in its initial input form.
        
        Returns:
            dict: The processed operator with unified equation form and checked coefficients.
        """

        operator = self.equation_unify(operator)
        for operator_label in operator:
            term = operator[operator_label]
            term['coeff'] = self._checking_coeff(term['coeff'])
        return operator

    def operator_prepare(self) -> list:
        """
        Prepares the operators for solving the differential equation by adapting them to the neural network solver.
        
                If the system involves multiple equations, it prepares each operator individually.
        
                Args:
                    self: The Equation_autograd object containing the operators and equation details.
        
                Returns:
                    list: A list of dictionaries, where each dictionary represents a prepared operator
                          ready for use in the neural network solver. The length of the list corresponds
                          to the number of equations in the system.
        
                Why:
                    This method ensures that the operators are in a format compatible with the neural network
                    solver, allowing the solver to accurately approximate the solution to the differential
                    equation. It handles both single-equation and multi-equation systems.
        """

        if isinstance(self.operator, list) and isinstance(self.operator[0], dict):
            num_of_eq = len(self.operator)
            prepared_operator = []
            for i in range(num_of_eq):
                equation = self.equation_unify(self.operator[i])
                prepared_operator.append(self._one_operator_prepare(equation))
        else:
            equation = self.equation_unify(self.operator)
            prepared_operator = [self._one_operator_prepare(equation)]

        return prepared_operator

    def bnd_prepare(self) -> list:
        """
        Prepares boundary conditions for use in the neural network-based differential equation solver.
        
        This method ensures that the boundary conditions are in the correct format
        for use during the training process. It essentially acts as a pass-through,
        returning the boundary conditions if they exist, or None otherwise. This
        is crucial for setting up the problem so that the neural network can learn
        the solution that satisfies the given constraints.
        
        Args:
            None
        
        Returns:
            list: A list of dictionaries, where each dictionary represents a boundary condition.
                  Returns None if no boundary conditions are defined.
        """

        if self.bconds is None:
            return None
        else:
            return self.bconds


class Equation_mat(EquationMixin):
    """
    Class realizes input data preprocessing (operator and boundary conditions
        preparing) for 'mat' method.
    """


    def __init__(self,
                 grid: torch.Tensor,
                 operator: Union[list, dict],
                 bconds: list):
        """
        Initializes the equation with the problem domain, differential operator, and boundary conditions.
        
        This setup is crucial for defining the problem that the neural network will learn to solve.
        The grid represents the domain, the operator describes the differential equation,
        and the boundary conditions constrain the solution space.
        
        Args:
            grid (torch.Tensor): The computational grid where the solution is approximated.
            operator (Union[list, dict]): Definition of the differential operator.
            bconds (list): Boundary conditions that the solution must satisfy.
        """

        self.grid = grid
        self.operator = operator
        self.bconds = bconds

    def operator_prepare(self) -> list:
        """
        Prepares the differential operator for solving the equation.
        
        This method unifies the format of the operator to ensure compatibility
        with the neural network-based solver. It processes the operator,
        converting it into a standardized form suitable for subsequent
        calculations within the solution process.
        
        Args:
            self (Equation_mat): An instance of the Equation_mat class
              containing the differential operator to be prepared.
        
        Returns:
            list: A list containing the prepared differential operator(s)
            in a unified format, ready for use in the neural network-based
            solution process.
        """

        if isinstance(self.operator, list) and isinstance(self.operator[0], dict):
            num_of_eq = len(self.operator)
            prepared_operator = []
            for i in range(num_of_eq):
                equation = self.equation_unify(self.operator[i])
                prepared_operator.append(equation)
        else:
            equation = self.equation_unify(self.operator)
            prepared_operator = [equation]

        return prepared_operator

    def _point_position(self, bnd: torch.Tensor) -> list:
        """
        Locates the grid indices corresponding to boundary points.
        
        This method determines the discrete grid positions that most closely
        represent the given boundary points. It is crucial for enforcing
        boundary conditions when solving differential equations using a neural
        network on a discretized domain.
        
        Args:
            bnd (torch.Tensor): A tensor containing the coordinates of the
                boundary points.
        
        Returns:
            list: A list of tuples, where each tuple contains the indices of the
                grid points that correspond to a boundary point. The length of
                the tuple corresponds to dimension of the grid.
        """

        bpos = []
        for pt in bnd:
            if self.grid.shape[0] == 1:
                point_pos = (torch.tensor(self.bndpos(self.grid, pt)),)
            else:
                prod = (torch.zeros_like(self.grid[0]) + 1).bool()
                for axis in range(self.grid.shape[0]):
                    axis_intersect = torch.isclose(
                        pt[axis].float(), self.grid[axis].float())
                    prod *= axis_intersect
                    point_pos = torch.where(prod)
            bpos.append(point_pos)
        return bpos

    def bnd_prepare(self) -> list:
        """
        Prepares boundary conditions for use in the neural network-based differential equation solver.
        
        This method transforms the user-provided boundary conditions into a format suitable
        for evaluating losses within the neural network training loop. It determines the
        numerical position of each boundary condition and unifies the boundary operators
        to ensure compatibility with the equation structure. This ensures that boundary
        conditions are correctly applied during the training process, guiding the neural
        network to learn solutions that satisfy the problem's constraints.
        
        Args:
            None
        
        Returns:
            list: A list of dictionaries, where each dictionary represents a boundary condition
                  with updated 'bnd' (boundary position) and 'bop' (boundary operator) keys,
                  ready for use in loss calculations.
        """

        for bcond in self.bconds:
            if bcond['type'] == 'periodic':
                bpos = []
                for bnd in bcond['bnd']:
                    bpos.append(self._point_position(bnd))
            else:
                bpos = self._point_position(bcond['bnd'])
            if bcond['bop'] is not None:
                bcond['bop'] = self.equation_unify(bcond['bop'])
            bcond['bnd'] = bpos
        return self.bconds


class Operator_bcond_preproc():
    """
    Interface for preparing equations due to chosen calculation method.
    """

    def __init__(self,
                 grid: torch.Tensor,
                 operator: Union[dict, list],
                 bconds: list,
                 h: float = 0.001,
                 inner_order: str ='1',
                 boundary_order: str ='2'):
        """
        Initializes the preprocessor with the problem definition.
        
                This preprocessor prepares the problem definition, including the equation, boundary conditions, and discretization parameters, for use in the neural network-based differential equation solver. It stores the grid, operator, boundary conditions, and discretization parameters for subsequent use in setting up the loss function and training the neural network.
        
                Args:
                    grid (torch.Tensor): Grid representing the domain of the differential equation.  This is typically generated using `cartesian_prod` or `meshgrid`.
                    operator (Union[dict, list]): Definition of the differential equation to be solved.
                    bconds (list): List of boundary conditions that constrain the solution.
                    h (float, optional): Discretization parameter (grid resolution). Defaults to 0.001.
                    inner_order (str, optional): Accuracy order for finite difference approximation within the domain. Defaults to '1'.
                    boundary_order (str, optional): Accuracy order for finite difference approximation at the boundaries. Defaults to '2'.
        
                Returns:
                    None
        """

        self.grid = check_device(grid)
        self.operator = operator
        self.bconds = bconds
        self.h = h
        self.inner_order = inner_order
        self.boundary_order = boundary_order

    def set_strategy(self, strategy: str) -> Union[Equation_NN, Equation_mat, Equation_autograd]:
        """
        Selects the appropriate equation formulation for solving the differential equation.
        
        This method determines how the differential equation will be represented and solved,
        choosing between a neural network-based approach, a matrix-based approach, or an
        automatic differentiation approach. The choice impacts how the solution is approximated
        and optimized.
        
        Args:
            strategy (str):  Specifies the calculation method to use for solving the equation
                             ('NN' for neural network, 'mat' for matrix-based, 'autograd' for
                             automatic differentiation).
        
        Returns:
            Union[Equation_NN, Equation_mat, Equation_autograd]: An instance of the selected
                                                                 equation type, configured with
                                                                 the problem's grid, operator,
                                                                 and boundary conditions.
        """

        if strategy == 'NN':
            return Equation_NN(self.grid, self.operator, self.bconds, h=self.h,
                               inner_order=self.inner_order,
                               boundary_order=self.boundary_order)
        if strategy == 'mat':
            return Equation_mat(self.grid, self.operator, self.bconds)
        if strategy == 'autograd':
            return Equation_autograd(self.grid, self.operator, self.bconds)
