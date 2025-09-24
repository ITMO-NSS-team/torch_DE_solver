"""Module of derivative calculations.
"""

from typing import Any, Union, List, Tuple, Callable
import numpy as np
from scipy import linalg
import torch


class DerivativeInt():
    """
    Interface class
    """

    def take_derivative(self, value):
        """
        Calculates the derivative of the neural network's output with respect to its input.
        
        This method is essential for training the neural network to approximate the solution of a differential equation.
        It leverages automatic differentiation in PyTorch to compute the derivatives required for the loss function,
        which measures the difference between the network's output and the desired solution.
        
        Args:
            value (torch.Tensor): The input value(s) at which to compute the derivative.
        
        Returns:
            torch.Tensor: The derivative of the network's output with respect to the input value(s).
        
        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError


class Derivative_NN(DerivativeInt):
    """
    Taking numerical derivative for 'NN' method.
    """


    def __init__(self, model: Any):
        """
        Initializes the Derivative_NN with a given neural network model. This model will be used to approximate the solution of a differential equation.
        
                Args:
                    model: The neural network model to be used as the solver.
        
                Returns:
                    None
        """
        self.model = model

    def take_derivative(self, term: Union[list, int, torch.Tensor], *args) -> torch.Tensor:
        """
        Computes the contribution of a single differential operator term to the overall differential equation residual.
        
        This function evaluates a component of the differential equation defined by the provided term,
        effectively calculating a part of the equation that the neural network aims to satisfy.
        
        Args:
            term (Union[list, int, torch.Tensor]): A dictionary representing a differential operator term,
                containing information about the coefficient, variable, differentiation direction,
                stencil grids, weights, and power.
        
        Returns:
            torch.Tensor: The computed field representing the contribution of the term to the differential equation,
                evaluated on the grid.
        """

        dif_dir = list(term.keys())[1]
        if isinstance(term['coeff'], tuple):
            coeff = term['coeff'][0](term['coeff'][1]).reshape(-1, 1)
        else:
            coeff = term['coeff']

        der_term = 1.
        for j, scheme in enumerate(term[dif_dir][0]):
            grid_sum = 0.
            for k, grid in enumerate(scheme):
                grid_sum += self.model(grid)[:, term['var'][j]].reshape(-1, 1)\
                    * term[dif_dir][1][j][k]
            if isinstance(term['pow'][j], (int, float)):
                der_term = der_term * grid_sum ** term['pow'][j]
            elif isinstance(term['pow'][j], Callable):
                der_term = term['pow'][j](der_term * grid_sum)
        der_term = coeff * der_term

        return der_term


class Derivative_autograd(DerivativeInt):
    """
    Taking numerical derivative for 'autograd' method.
    """


    def __init__(self, model: torch.nn.Module):
        """
        Initializes the Derivative_autograd class.
        
                This class prepares a given neural network model for use in the differential equation solving process. It essentially wraps the provided model, making it ready for subsequent computations required for approximating solutions to differential equations.
        
                Args:
                    model (torch.nn.Module): The neural network model to be used for solving the differential equation. This model should be defined using PyTorch's autograd functionality.
        
                Returns:
                    None
        """
        self.model = model

    @staticmethod
    def _nn_autograd(model: torch.nn.Module,
                     points: torch.Tensor,
                     var: int,
                     axis: List[int] = [0]):
        """
        Computes the derivative of the neural network output with respect to the input points using PyTorch's autograd functionality. This is a core component for calculating the loss function when training neural networks to solve differential equations, as it allows us to compare the network's predicted derivatives with the derivatives specified by the equation.
        
                Args:
                    model (torch.nn.Module): The neural network model.
                    points (torch.Tensor): The input points at which to calculate the derivative.
                    var (int): The index of the output variable to differentiate (for systems of equations).
                    axis (List[int], optional): The axes with respect to which to differentiate. Defaults to [0].
        
                Returns:
                    torch.Tensor: The computed derivative of the model's output at the given points.
        """

        points.requires_grad = True
        fi = model(points)[:, var].sum(0)
        for ax in axis:
            grads, = torch.autograd.grad(fi, points, create_graph=True)
            fi = grads[:, ax].sum()
        gradient_full = grads[:, axis[-1]].reshape(-1, 1)
        return gradient_full

    def take_derivative(self, term: dict, grid_points:  torch.Tensor) -> torch.Tensor:
        """
        Computes the contribution of a single differential operator term to the overall solution.
        
        This function calculates the value of a term in the differential equation,
        involving derivatives of the neural network's output with respect to specified
        variables. It handles cases where the coefficient of the term is a constant,
        a function of the grid points, or a tensor.  It also accounts for different
        orders of derivatives and powers of derivative terms.
        
        Args:
            term (dict): A dictionary defining the differential operator term.  It
                         specifies the variable to differentiate with respect to, the
                         order of the derivative, the power to raise the derivative to,
                         and the coefficient of the term.
            grid_points (torch.Tensor): The points at which to evaluate the term.
                                         These points serve as input to the neural network
                                         and are used to compute the derivatives.
        
        Returns:
            torch.Tensor: The value of the differential operator term evaluated at the
                          given grid points. This represents the contribution of this
                          specific term to the overall differential equation.
        """

        dif_dir = list(term.keys())[1]
        # it is may be int, function of grid or torch.Tensor
        if callable(term['coeff']):
            coeff = term['coeff'](grid_points).reshape(-1, 1)
        else:
            coeff = term['coeff']

        der_term = 1.
        for j, derivative in enumerate(term[dif_dir]):
            if derivative == [None]:
                der = self.model(grid_points)[:, term['var'][j]].reshape(-1, 1)
            else:
                der = self._nn_autograd(
                    self.model, grid_points, term['var'][j], axis=derivative)
            if isinstance(term['pow'][j], (int, float)):
                der_term = der_term * der ** term['pow'][j]
            elif isinstance(term['pow'][j], Callable):
                der_term = term['pow'][j](der_term * der)
        der_term = coeff * der_term

        return der_term


class Derivative_mat(DerivativeInt):
    """
    Taking numerical derivative for 'mat' method.
    """

    def __init__(self, model: torch.Tensor, derivative_points: int):
        """
        Initializes the Derivative_mat object.
        
        This method prepares the derivative calculation by pre-computing coefficients
        used in approximating derivatives with finite difference methods. It sets up
        the necessary data structures for both backward and forward difference schemes,
        allowing for efficient computation of derivatives during the solving of
        differential equations.
        
        Args:
            model (torch.Tensor): The model (e.g., a neural network) whose output
                derivatives are to be computed. This represents the solution
                approximation at a given point.
            derivative_points (int): The number of points to use in the finite
                difference approximation of the derivative. More points generally
                lead to a more accurate approximation but increase computational cost.
        
        Returns:
            None
        """
        self.model = model
        self.backward, self.farward = Derivative_mat._labels(derivative_points)

        self.alpha_backward = Derivative_mat._linear_system(self.backward)
        self.alpha_farward = Derivative_mat._linear_system(self.farward)

        num_points = int(len(self.backward) - 1)

        self.back = [int(0 - i) for i in range(1, num_points + 1)]

        self.farw = [int(i) for i in range(num_points)]

    @staticmethod
    def _labels(derivative_points: int) -> Tuple[List, List]:
        """
        Determines the indices of points used in approximating derivatives.
        
        This function generates index sets for calculating derivatives using backward and forward difference schemes.
        These indices are essential for constructing the differentiation matrices, which approximate derivative operators
        on a discrete grid. The backward and forward indices facilitate the calculation of derivatives at each point
        by referencing neighboring points.
        
        Args:
            derivative_points (int): The number of points used in the derivative calculation.
        
        Returns:
            Tuple[List[int], List[int]]: A tuple containing two lists.
                - The first list (labels_backward) contains the indices for the backward difference scheme.
                - The second list (labels_forward) contains the indices for the forward difference scheme.
        """
        labels_backward = list(i for i in range(-derivative_points + 1, 1))
        labels_farward = list(i for i in range(derivative_points))
        return labels_backward, labels_farward

    @staticmethod
    def _linear_system(labels: list) -> np.ndarray:
        """
        Solves a linear system to determine the coefficients for approximating derivatives using a set of points.
        
                This method constructs and solves a linear system of equations to find the weights
                that best approximate the derivative at a given point based on the provided stencil.
                These coefficients are crucial for accurately representing derivatives within the
                neural network-based differential equation solver.
        
                Args:
                    labels (list): A list of points representing the stencil for derivative approximation.
        
                Returns:
                    np.ndarray: The coefficients for the numerical scheme, obtained by solving the linear system.
        """
        points_num = len(labels) # num_points=number of equations
        labels = np.array(labels)
        A = []
        for i in range(points_num):
            A.append(labels**i)
        A = np.array(A)

        b = np.zeros_like(labels)
        b[1] = 1

        alpha = linalg.solve(A, b)

        return alpha

    def _derivative_1d(self, u_tensor: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Computes the derivative of a tensor along one dimension using a finite difference scheme. This is a key step in neural differential equation solvers, where derivatives are needed to calculate loss functions and update model parameters.
        
                Args:
                    u_tensor (torch.Tensor): The input tensor for which the derivative is computed. Represents the dependent variable of the differential equation.
                    h (torch.Tensor): The step size or increment used in the finite difference approximation.
        
                Returns:
                    du (torch.Tensor): The computed derivative of the input tensor along the specified dimension.
        """

        shape = u_tensor.shape
        u_tensor = u_tensor.reshape(-1)

        du_back = 0
        du_farw = 0
        i = 0
        for shift_b, shift_f in zip(self.backward, self.farward):
            du_back += torch.roll(u_tensor, -shift_b) * self.alpha_backward[i]
            du_farw += torch.roll(u_tensor, -shift_f) * self.alpha_farward[i]
            i += 1
        du = (du_back + du_farw) / (2 * h)
        du[self.back] = du_back[self.back] / h
        du[self.farw] = du_farw[self.farw] / h

        du = du.reshape(shape)

        return du

    def _step_h(self, h_tensor: torch.Tensor) -> list[torch.Tensor]:
        """
        Calculates the grid spacing along each axis.
        
        This function determines the resolution of the grid used for solving
        differential equations by computing the difference between
        consecutive unique points along each axis. This information is crucial
        for accurately calculating derivatives using finite difference methods
        within the neural network solver.
        
        Args:
            h_tensor (torch.Tensor): A tensor representing the grid in *mat* mode,
                                     where each row corresponds to a dimension and
                                     each column represents a point on the grid.
        
        Returns:
            list[torch.Tensor]: A list containing the grid spacing (h) for each
                                 axis of the grid. Each element in the list is a
                                 torch.Tensor representing the grid spacing along
                                 the corresponding axis.
        """
        h = []

        nn_grid = torch.vstack([h_tensor[i].reshape(-1) for i in \
                                range(h_tensor.shape[0])]).T.float()

        for i in range(nn_grid.shape[-1]):
            axis_points = torch.unique(nn_grid[:,i])
            h.append(abs(axis_points[1]-axis_points[0]))
        return h

    def _derivative(self,
                    u_tensor: torch.Tensor,
                    h: torch.Tensor,
                    axis: int) -> torch.Tensor:
        """
        Computes the numerical derivative of a tensor along a specified axis, leveraging a finite difference scheme.
        
        This method approximates the derivative of `u_tensor` with respect to a given axis `axis`
        using a combination of forward and backward differences. The `h` parameter controls the step size
        of the finite difference approximation. The method handles both 1D and multi-dimensional tensors,
        applying different calculations at the boundaries to improve accuracy.
        
        Args:
            u_tensor (torch.Tensor): The tensor for which the derivative is computed. This represents the
                dependent variable in the differential equation being solved.
            h (torch.Tensor): The step size used in the finite difference approximation.  It represents
                the increment in the independent variable.
            axis (int): The axis along which the derivative is calculated.  This specifies the direction
                in which the rate of change is being computed.
        
        Returns:
            torch.Tensor: The computed derivative of `u_tensor` along the specified axis. This approximates
                the rate of change of the solution to the differential equation.
        
        Why:
            This method is crucial for approximating derivatives within the neural network-based differential
            equation solver.  By numerically estimating derivatives, the neural network can learn to satisfy
            the differential equation, even when an analytical solution is unavailable. The finite difference
            scheme provides a way to relate the network's output to the derivatives required by the equation.
        """

        if len(u_tensor.shape)==1 or u_tensor.shape[0]==1:
            du = self._derivative_1d(u_tensor, h)
            return du

        pos = len(u_tensor.shape) - 1

        u_tensor = torch.transpose(u_tensor, pos, axis)

        du_back = 0
        du_farw = 0
        i = 0
        for shift_b, shift_f in zip(self.backward, self.farward):
            du_back += torch.roll(u_tensor, -shift_b) * self.alpha_backward[i]
            du_farw += torch.roll(u_tensor, -shift_f) * self.alpha_farward[i]
            i += 1
        du = (du_back + du_farw) / (2 * h)

        if pos == 1:
            du[:,self.back] = du_back[:,self.back] / h
            du[:, self.farw] = du_farw[:, self.farw] / h
        elif pos == 2:
            du[:,:, self.back] = du_back[:,:, self.back] / h
            du[:,:, self.farw] = du_farw[:,:, self.farw] / h

        du = torch.transpose(du, pos, axis)

        return du

    def take_derivative(self, term: torch.Tensor, grid_points: torch.Tensor) -> torch.Tensor:
        """
        Auxiliary function to compute the contribution of a single term in the differential operator.
        
                This function calculates the derivative of a field with respect to specified variables and orders,
                effectively evaluating one component of the overall differential equation on the given grid points.
                It is a crucial step in constructing the complete solution by combining the contributions of all terms.
        
                Args:
                    term (torch.Tensor): A dictionary representing a single term in the differential operator,
                                         containing information about the variable, derivative order, and coefficient.
                    grid_points (torch.Tensor): The coordinates at which the solution is to be evaluated.
        
                Returns:
                    der_term (torch.Tensor): The computed value of the term on the grid.
        """

        dif_dir = list(term.keys())[1]
        der_term = torch.zeros_like(self.model) + 1
        for j, scheme in enumerate(term[dif_dir]):
            prod=self.model[term['var'][j]]
            if scheme!=[None]:
                for axis in scheme:
                    if axis is None:
                        continue
                    h = self._step_h(grid_points)[axis]
                    prod = self._derivative(prod, h, axis)
            if isinstance(term['pow'][j], (int, float)):
                der_term = der_term * prod ** term['pow'][j]
            elif isinstance(term['pow'][j], Callable):
                der_term = term['pow'][j](der_term * prod)
        if callable(term['coeff']) is True:
            der_term = term['coeff'](grid_points) * der_term
        else:
            der_term = term['coeff'] * der_term
        return der_term


class Derivative():
    """
    Abstract base class for numerical differentiation. Provides a consistent interface for computing derivatives using different numerical methods within a neural network-based differential equation solving framework. Enables flexible selection and implementation of derivative calculation techniques.
    """

    def __init__(self,
                 model: Union[torch.nn.Module, torch.Tensor],
                 derivative_points: int):
        """
        Initializes the Derivative class with a neural network model and the number of points to use for derivative calculation. This setup is crucial for accurately approximating derivatives within the neural network-based differential equation solver. The number of derivative points influences the precision of the numerical scheme used to estimate derivatives, which directly affects the solver's ability to find accurate solutions.
        
                Args:
                    model (Union[torch.nn.Module, torch.Tensor]): The neural network or matrix representing the system being modeled.
                    derivative_points (int): The number of points to use in the numerical scheme for derivative calculation.  A higher number of points can increase accuracy but also computational cost.
        
                Returns:
                    None
        """

        self.model = model
        self.derivative_points = derivative_points

    def set_strategy(self,
                     strategy: str) -> Union[Derivative_NN, Derivative_autograd, Derivative_mat]:
        """
        Sets the strategy for calculating derivatives within the neural network-based differential equation solver.
        
        This method configures the derivative calculation approach, allowing users to select from different techniques 
        like neural networks, autograd, or matrix-based methods. This choice impacts how the derivatives required for 
        solving the differential equation are computed within the neural network framework.
        
        Args:
            strategy (str): Specifies the derivative calculation method to use. 
                            Valid options are "NN" (neural network), "autograd" (automatic differentiation), and "mat" (matrix-based).
        
        Returns:
            Union[Derivative_NN, Derivative_autograd, Derivative_mat]: An instance of the class corresponding to the chosen derivative calculation strategy.
                                                                        This object encapsulates the logic for computing derivatives using the selected method.
        """
        if strategy == 'NN':
            return Derivative_NN(self.model)

        elif strategy == 'autograd':
            return Derivative_autograd(self.model)

        elif strategy == 'mat':
            return Derivative_mat(self.model, self.derivative_points)











