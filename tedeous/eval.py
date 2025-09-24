"""Module for operatoins with operator and boundaru con-ns."""

from typing import Tuple, Union, List, Callable
import torch

from tedeous.points_type import Points_type
from tedeous.derivative import Derivative
from tedeous.device import device_type, check_device
from tedeous.utils import PadTransform

from torch.utils.data import DataLoader

def integration(func: torch.Tensor,
                grid: torch.Tensor,
                power: int = 2) \
                -> Union[Tuple[float, float], Tuple[list, torch.Tensor]]:
    """
    Performs numerical integration of subintegrands along one axis of the grid.
    
    This function integrates the product of the differential equation's residual and a test function
    over a single spatial dimension. It iteratively computes the integral using the trapezoidal rule,
    accumulating the result until a boundary condition is encountered along the integration axis.
    This process effectively reduces the dimensionality of the integration domain, facilitating the
    solution of the differential equation using neural network approximations.
    
    Args:
        func (torch.Tensor): The differential equation's residual evaluated at each grid point,
                             representing (L(u)-f) * weak_form.
        grid (torch.Tensor): A tensor representing the spatial grid points where the residual is evaluated.
        power (int, optional): The power to which the residual is raised before integration. Defaults to 2.
    
    Returns:
        Tuple[Union[float, list], torch.Tensor]: A tuple containing the integration result and the updated grid.
            - If the grid has only one column, returns a float representing the final integral and 0.
            - Otherwise, returns a list of floats representing the integral over each segment
              and a tensor representing the grid with the last column removed.
    """
    if grid.shape[-1] == 1:
        column = -1
    else:
        column = -2
    marker = grid[0][column]
    index = [0]
    result = []
    u = 0.
    for i in range(1, len(grid)):
        if grid[i][column] == marker or column == -1:
            u += (grid[i][-1] - grid[i - 1][-1]).item() * \
                 (func[i] ** power + func[i - 1] ** power) / 2
        else:
            result.append(u)
            marker = grid[i][column]
            index.append(i)
            u = 0.
    if column == -1:
        return u, 0.
    else:
        result.append(u)
        grid = grid[index, :-1]
        return result, grid


def dict_to_matrix(bval: dict, true_bval: dict)\
    -> Tuple[torch.Tensor, torch.Tensor, List, List]:
    """
    Transforms dictionaries of boundary values into matrix representations for neural network training.
    
    This function prepares boundary data for use in neural network-based differential equation solvers.
    It converts dictionaries of predicted and true boundary values into matrices, padding shorter
    sequences to ensure consistent dimensions for efficient processing by neural networks. This
    is crucial for training models to accurately approximate solutions to differential equations
    by learning from the provided boundary conditions.
    
    Args:
        bval (dict): Dictionary containing predicted boundary values for different boundary types.
                     Keys represent boundary types, and values are tensors of predicted values.
        true_bval (dict): Dictionary containing true boundary values for different boundary types.
                          Keys represent boundary types, and values are tensors of true values.
    
    Returns:
        matrix_bval (torch.Tensor): Matrix where each column contains padded predicted boundary values
                                     for a specific boundary type.
        matrix_true_bval (torch.Tensor): Matrix where each column contains padded true boundary values
                                          for a specific boundary type.
        keys (list): List of boundary types corresponding to the columns in `matrix_bval` and
                     `matrix_true_bval`.
        len_list (list): List of original lengths of the boundary value tensors for each boundary
                         type before padding.
    """

    keys = list(bval.keys())
    max_len = max([len(i) for i in bval.values()])
    pad = PadTransform(max_len, 0)
    matrix_bval = pad(bval[keys[0]]).reshape(-1,1)
    matrix_true_bval = pad(true_bval[keys[0]]).reshape(-1,1)
    len_list = [len(bval[keys[0]])]
    for key in keys[1:]:
        bval_i = pad(bval[key]).reshape(-1,1)
        true_bval_i = pad(true_bval[key]).reshape(-1,1)
        matrix_bval = torch.hstack((matrix_bval, bval_i))
        matrix_true_bval = torch.hstack((matrix_true_bval, true_bval_i))
        len_list.append(len(bval[key]))

    return matrix_bval, matrix_true_bval, keys, len_list


class Operator():
    """
    Class for differential equation calculation.
    """

    def __init__(self,
                 grid: torch.Tensor,
                 prepared_operator: Union[list,dict],
                 model: Union[torch.nn.Sequential, torch.Tensor],
                 mode: str,
                 weak_form: list[callable],
                 derivative_points: int,
                 batch_size: int = None):
        """
        Initializes the Operator instance, preparing it for the differential equation solving process.
        
                This involves setting up the computational grid, defining the operator (differential equation),
                specifying the neural network model, and configuring the mode of operation (e.g., 'NN', 'autograd', 'mat').
                The initialization also handles the creation of mini-batches for efficient processing, if a batch size is provided.
                This setup is crucial for efficiently approximating solutions to differential equations using neural networks.
        
                Args:
                    grid (torch.Tensor): grid (domain discretization).
                    prepared_operator (Union[list,dict]): prepared (after Equation class) operator.
                    model (Union[torch.nn.Sequential, torch.Tensor]): *mat or NN or autograd* model.
                    mode (str): *mat or NN or autograd*
                    weak_form (list[callable]): list with basis functions (if the form is *weak*).
                    derivative_points (int): points number for derivative calculation.
                                             For details to Derivative_mat class.
                    batch_size (int): size of batch.
        
                Returns:
                    None
        """
        self.grid = check_device(grid)
        self.prepared_operator = prepared_operator
        self.model = model.to(device_type())
        self.mode = mode
        self.weak_form = weak_form
        self.derivative_points = derivative_points
        if self.mode == 'NN':
            self.grid_dict = Points_type(self.grid).grid_sort()
            self.sorted_grid = torch.cat(list(self.grid_dict.values()))
        elif self.mode in ('autograd', 'mat'):
            self.sorted_grid = self.grid
        self.batch_size = batch_size
        if self.batch_size is not None:
            self.grid_loader =  DataLoader(self.sorted_grid, batch_size=self.batch_size, shuffle=True,
                                      generator=torch.Generator(device=device_type()))
            self.n_batches = len(self.grid_loader)
            del self.sorted_grid
            torch.cuda.empty_cache()
            self.init_mini_batches()
            self.current_batch_i = 0
        self.derivative = Derivative(self.model,
                                self.derivative_points).set_strategy(self.mode).take_derivative

    def init_mini_batches(self):
        """
        Initializes the mini-batch iterator for training. This prepares the data loader to provide batches of data points sampled across the problem domain, which are used to iteratively refine the neural network's approximation of the differential equation's solution.
        
                Args:
                    self: The Operator instance.
        
                Returns:
                    None. The method initializes the `grid_iter` and `grid_batch` attributes of the Operator instance.
        
                Why: To prepare batches of data points sampled across the problem domain, which are used to iteratively refine the neural network's approximation of the differential equation's solution.
        """
        self.grid_iter = iter(self.grid_loader)
        self.grid_batch = next(self.grid_iter)

    def apply_operator(self,
                       operator: list,
                       grid_points: Union[torch.Tensor, None]) -> torch.Tensor:
        """
        Applies a preprocessed differential operator to a grid subset to approximate the solution field.
        
        This method iterates through the terms of a differential operator, calculates the
        derivative of each term, and accumulates the results to approximate the overall
        effect of the operator on the solution field within a specific grid subset. This
        process is crucial for evaluating how well the neural network's output satisfies
        the differential equation within that region.
        
        Args:
            operator (list): A list of preprocessed terms representing the differential operator.
                             See `input_preprocessing.operator_prepare()` for details on the expected format.
            grid_points (torch.Tensor, optional): The coordinates within the grid subset where the
                                                   derivatives are evaluated. Required for 'autograd' and 'mat' modes.
                                                   Defaults to None.
        
        Returns:
            torch.Tensor: The approximated result of applying the differential operator to the
                          solution field within the grid subset. This represents the residual
                          or error of the neural network's solution at these points.
        """

        for term in operator:
            term = operator[term]
            dif = self.derivative(term, grid_points)
            try:
                total += dif
            except NameError:
                total = dif
        return total

    def _pde_compute(self) -> torch.Tensor:
        """
        Computes the residual of the differential equation. This involves applying the defined differential operator(s) to the input grid points. The method handles both single and multiple equation systems by iterating through the prepared operators and concatenating the results. Mini-batching is used when a batch size is specified, allowing for efficient processing of large datasets.
        
                Args:
                    None
        
                Returns:
                    torch.Tensor: The computed residual of the differential equation(s) on the given grid. This represents how well the neural network's output satisfies the equation(s).
        """

        if self.batch_size is not None:
            sorted_grid = self.grid_batch
            try:
                self.grid_batch = next(self.grid_iter)
            except: # if no batches left then reinit
                self.init_mini_batches()
                self.current_batch_i = -1
        else:
            sorted_grid = self.sorted_grid
        num_of_eq = len(self.prepared_operator)
        if num_of_eq == 1:
            op = self.apply_operator(
                self.prepared_operator[0], sorted_grid).reshape(-1,1)
        else:
            op_list = []
            for i in range(num_of_eq):
                op_list.append(self.apply_operator(
                    self.prepared_operator[i], sorted_grid).reshape(-1,1))
            op = torch.cat(op_list, 1)
        return op

    def _weak_pde_compute(self) -> torch.Tensor:
        """
        Computes the weak form of the PDE residual by integrating the product of the PDE operator and test functions over the domain. This process transforms the differential equation into an integral equation, suitable for numerical solution using neural network approximations. The weak form allows for solutions that are not necessarily differentiable in the classical sense, expanding the range of solvable problems.
        
                Args:
                    None
        
                Returns:
                    torch.Tensor: weak PDE residual.
        """

        device = device_type()
        if self.mode == 'NN':
            grid_central = self.grid_dict['central']
        elif self.mode == 'autograd':
            grid_central = self.grid

        op = self._pde_compute()
        sol_list = []
        for i in range(op.shape[-1]):
            sol = op[:, i]
            for func in self.weak_form:
                sol = sol * func(grid_central).to(device).reshape(-1)
            grid_central1 = torch.clone(grid_central)
            for _ in range(grid_central.shape[-1]):
                sol, grid_central1 = integration(sol, grid_central1)
            sol_list.append(sol.reshape(-1, 1))
        if len(sol_list) == 1:
            return sol_list[0]
        else:
            return torch.cat(sol_list).reshape(1,-1)

    def operator_compute(self):
        """
        Calculates the residual of the differential operator, serving as a measure of how well the neural network satisfies the equation.
        
        This computation is central to training the neural network to approximate the solution of the differential equation.
        The residual is calculated based on either a strong or weak formulation of the equation.
        
        Args:
            None
        
        Returns:
            torch.Tensor: The operator residual, a tensor representing the error in satisfying the differential equation.
        """
        if self.weak_form is None or self.weak_form == []:
            return self._pde_compute()
        else:
            return self._weak_pde_compute()


class Bounds():
    """
    Class for boundary and initial conditions calculation.
    """

    def __init__(self,
                 grid: torch.Tensor,
                 prepared_bconds: Union[list, dict],
                 model: Union[torch.nn.Sequential, torch.Tensor],
                 mode: str,
                 weak_form: list[callable],
                 derivative_points: int):
        """
        Initializes the Bounds object.
        
        This class manages the boundary conditions and the operator associated with the differential equation.
        It prepares the necessary components for solving the equation within the specified domain.
        
        Args:
            grid (torch.Tensor): The computational grid representing the domain discretization.
            prepared_bconds (Union[list, dict]): Boundary conditions, preprocessed by the Equation class.
            model (Union[torch.nn.Sequential, torch.Tensor]): The neural network or matrix model used to approximate the solution.
            mode (str): Specifies the solution approach ('mat', 'NN', or 'autograd').
            weak_form (list[callable]): Basis functions for the weak formulation of the equation (if applicable).
            derivative_points (int): Number of points used for derivative calculations in the matrix approach.
        
        Returns:
            None
        """
        self.grid = check_device(grid)
        self.prepared_bconds = prepared_bconds
        self.model = model.to(device_type())
        self.mode = mode
        self.operator = Operator(self.grid, self.prepared_bconds,
                                       self.model, self.mode, weak_form,
                                       derivative_points)

    def _apply_bconds_set(self, operator_set: list) -> torch.Tensor:
        """
        Applies a set of boundary conditions to the solution field.
        
        This method iterates through a list of boundary operators, applies each operator to the solution field,
        and concatenates the results. This effectively enforces the specified boundary conditions on the neural network's
        solution, guiding it towards satisfying the constraints of the differential equation.
        
        Args:
            operator_set (list): A list of prepared boundary operators, typically generated by the Equation_NN.operator_prepare method.
                                 Each operator represents a specific boundary condition to be applied.
        
        Returns:
            torch.Tensor: A tensor representing the combined effect of all boundary operators on the solution field.
                          This tensor is used to penalize deviations from the specified boundary conditions during training,
                          ensuring that the neural network solution adheres to the problem's constraints.
        """

        field_part = []
        for operator in operator_set:
            field_part.append(self.operator.apply_operator(operator, None))
        field_part = torch.cat(field_part)
        return field_part

    def _apply_dirichlet(self, bnd: torch.Tensor, var: int) -> torch.Tensor:
        """
        Applies Dirichlet boundary conditions by evaluating the neural network model at the boundary points. This ensures that the solution adheres to the specified values at the domain boundaries, a crucial step in accurately solving the differential equation.
        
                Args:
                    bnd (torch.Tensor): Terms (boundary points) of prepared boundary conditions.
                        For more details, refer to input_preprocessing (bnd_prepare method).
                    var (int): Indicates the dependent variable for which to apply the boundary condition.
                        For a single equation, this is typically 0.
        
                Returns:
                    torch.Tensor: The calculated boundary condition values, obtained by evaluating the neural network at the boundary points.
        """

        if self.mode == 'NN' or self.mode == 'autograd':
            b_op_val = self.model(bnd)[:, var].reshape(-1, 1)
        elif self.mode == 'mat':
            b_op_val = []
            for position in bnd:
                b_op_val.append(self.model[var][position])
            b_op_val = torch.cat(b_op_val).reshape(-1, 1)
        return b_op_val

    def _apply_neumann(self, bnd: torch.Tensor, bop: list) -> torch.Tensor:
        """
        Applies derivative operators to the boundary conditions based on the chosen mode.
        
                This method calculates the boundary condition value by applying the derivative operator.
                The specific calculation depends on the selected mode ('NN', 'autograd', or 'mat'),
                allowing for different approaches to enforce boundary conditions when solving
                differential equations with neural networks.
        
                Args:
                    bnd (torch.Tensor): Terms (boundary points) of the prepared boundary conditions.
                    bop (list): Terms of the prepared boundary derivative operator.
        
                Returns:
                    torch.Tensor: Calculated boundary condition.
        """

        if self.mode == 'NN':
            b_op_val = self._apply_bconds_set(bop)
        elif self.mode == 'autograd':
            b_op_val = self.operator.apply_operator(bop, bnd)
        elif self.mode == 'mat':
            var = bop[list(bop.keys())[0]]['var'][0]
            b_op_val = self.operator.apply_operator(bop, self.grid)
            b_val = []
            for position in bnd:
                b_val.append(b_op_val[var][position])
            b_op_val = torch.cat(b_val).reshape(-1, 1)
        return b_op_val

    def _apply_periodic(self, bnd: torch.Tensor, bop: list, var: int) -> torch.Tensor:
        """
        Applies periodic boundary conditions by evaluating the difference between boundary points,
                ensuring continuity of the solution across the domain boundaries.
        
                Args:
                    bnd (torch.Tensor): Terms (boundary points) of prepared boundary conditions.
                    bop (list): Terms of prepared boundary derivative operator.
                    var (int): Indicates for which dependent variable it is necessary to apply
                        the boundary condition. For single equation is 0.
        
                Returns:
                    torch.Tensor: Calculated boundary condition, representing the difference
                        enforcing periodicity.
        
                Why:
                This method enforces periodicity by calculating the difference between the solution
                or its derivatives at opposing boundaries. This difference is driven towards zero
                during training, ensuring a smooth, continuous solution across the periodic domain,
                which is crucial for accurately solving differential equations with periodic constraints
                using neural networks.
        """

        if bop is None:
            b_op_val = self._apply_dirichlet(bnd[0], var).reshape(-1, 1)
            for i in range(1, len(bnd)):
                b_op_val -= self._apply_dirichlet(bnd[i], var).reshape(-1, 1)
        else:
            if self.mode == 'NN':
                b_op_val = self._apply_neumann(bnd, bop[0]).reshape(-1, 1)
                for i in range(1, len(bop)):
                    b_op_val -= self._apply_neumann(bnd, bop[i]).reshape(-1, 1)
            elif self.mode in ('autograd', 'mat'):
                b_op_val = self._apply_neumann(bnd[0], bop).reshape(-1, 1)
                for i in range(1, len(bnd)):
                    b_op_val -= self._apply_neumann(bnd[i], bop).reshape(-1, 1)
        return b_op_val

    def _apply_robin(self, bnd: torch.Tensor, bop: Union[list, dict], var: int) -> torch.Tensor:
        """
        Applies Robin boundary conditions by combining the function value and its derivative at the boundary. This is done to enforce a mixed-type boundary constraint, where the solution is related to its derivative on the boundary.
        
                Args:
                    bnd (torch.Tensor): Boundary points where the condition is applied.
                    bop (Union[list, dict]): Dictionary containing the coefficients for the boundary condition, including alpha (coefficient for the function value) and betas (coefficients for the derivative terms).
                    var (int): Index of the variable to which the boundary condition applies.
        
                Returns:
                    torch.Tensor: The calculated Robin boundary condition value at the specified boundary points.
        """

        alpha, *betas = [bop[list(bop.keys())[i]]['coeff'] for i in range(len(bop))]

        value_term = alpha * self._apply_dirichlet(bnd, var)

        derivative_term = 0
        for beta in betas:
            if self.mode == 'NN':
                if isinstance(beta, (int, float)):
                    derivative_term += beta * self._apply_bconds_set(bop)
                elif isinstance(beta, Callable):
                    derivative_term += beta(bnd) * self._apply_bconds_set(bop)
            else:
                if isinstance(beta, (int, float)):
                    derivative_term += beta * self._apply_neumann(bnd, bop)
                elif isinstance(beta, Callable):
                    derivative_term += beta(bnd) * self._apply_neumann(bnd, bop)

        b_op_val = value_term + derivative_term
        return b_op_val

    def _apply_data(self, bnd: torch.Tensor, bop: list, var: int) -> torch.Tensor:
        """
        Applies boundary conditions to enforce known solution behavior.
        
        This method determines how to apply the provided boundary conditions,
        choosing between Dirichlet (value-based) and Neumann (derivative-based)
        conditions based on the provided operator. This ensures that the neural
        network solution adheres to the specified constraints at the boundaries
        of the problem domain, guiding the training process towards a physically
        accurate solution.
        
        Args:
            bnd (torch.Tensor): Terms (data points) of prepared boundary conditions.
            bop (list): Terms of prepared data derivative operator. If None, Dirichlet
                boundary conditions are applied; otherwise, Neumann conditions are used.
            var (int): Indicates for which dependent variable to apply the data condition.
                For a single equation, this is 0.
        
        Returns:
            torch.Tensor: Calculated data condition, representing the enforced
                boundary values or derivative constraints.
        """
        if bop is None:
            b_op_val = self._apply_dirichlet(bnd, var).reshape(-1, 1)
        else:
            b_op_val = self._apply_neumann(bnd, bop).reshape(-1, 1)
        return b_op_val

    def b_op_val_calc(self, bcond: dict) -> torch.Tensor:
        """
        Calculates the boundary operator value based on the specified boundary condition type.
        
        This function acts as a dispatcher, selecting the appropriate method to compute the boundary operator value
        based on the 'type' key within the provided boundary condition dictionary. This allows the framework
        to handle various types of boundary conditions, such as Dirichlet, Neumann, Periodic, Robin, and Data-driven
        conditions, enabling the neural network to learn the solution that satisfies the given constraints.
        
        Args:
            bcond (dict): A dictionary containing the terms of the prepared boundary conditions,
                          as generated by the `bnd_prepare` method in the `input_preprocessing` module.
                          This dictionary must include a 'type' key specifying the boundary condition type
                          (e.g., 'dirichlet', 'neumann', 'periodic', 'robin', 'data') and other keys
                          relevant to that type.
        
        Returns:
            torch.Tensor: The calculated value of the boundary operator, represented as a PyTorch tensor.
                          The specific meaning and shape of this tensor depend on the boundary condition type
                          and the underlying method used for its calculation.
        """

        b_op_val = None

        if bcond['type'] == 'dirichlet':
            b_op_val = self._apply_dirichlet(bcond['bnd'], bcond['var'])
        elif bcond['type'] == 'operator':
            b_op_val = self._apply_neumann(bcond['bnd'], bcond['bop'])
        elif bcond['type'] == 'periodic':
            b_op_val = self._apply_periodic(bcond['bnd'], bcond['bop'], bcond['var'])
        elif bcond['type'] == 'robin':
            b_op_val = self._apply_robin(bcond['bnd'], bcond['bop'], bcond['var'])
        elif bcond['type'] == 'data':
            b_op_val = self._apply_data(bcond['bnd'], bcond['bop'], bcond['var'])
        return b_op_val

    def apply_bcs(self) -> Tuple[torch.Tensor, torch.Tensor, list, list]:
        """
        Applies boundary and data conditions to prepare data for training the neural network to solve differential equations.
        
                The method iterates through the prepared boundary conditions, calculates the predicted boundary values using the specified operators, and organizes them along with the true boundary values. This prepares the data in a suitable format for training the neural network to approximate the solution of the differential equation subject to these conditions.
        
                Args:
                    self (Bounds): An instance of the Bounds class containing the prepared boundary conditions.
        
                Returns:
                    Tuple[torch.Tensor, torch.Tensor, list, list]: A tuple containing:
                        - bval (torch.Tensor): A matrix where each column represents the predicted boundary values for a specific boundary type.
                        - true_bval (torch.Tensor): A matrix where each column represents the true boundary values for a specific boundary type.
                        - keys (list): A list of boundary types corresponding to the columns in the `bval` and `true_bval` matrices.
                        - bval_length (list): A list containing the length of each boundary type column.
        """

        bval_dict = {}
        true_bval_dict = {}

        for bcond in self.prepared_bconds:
            try:
                bval_dict[bcond['type']] = torch.cat((bval_dict[bcond['type']],
                                                    self.b_op_val_calc(bcond).reshape(-1)))
                true_bval_dict[bcond['type']] = torch.cat((true_bval_dict[bcond['type']],
                                                    bcond['bval'].reshape(-1)))
            except:
                bval_dict[bcond['type']] = self.b_op_val_calc(bcond).reshape(-1)
                true_bval_dict[bcond['type']] = bcond['bval'].reshape(-1)

        bval, true_bval, keys, bval_length = dict_to_matrix(
                                                    bval_dict, true_bval_dict)

        return bval, true_bval, keys, bval_length