"""Module for connecting *eval.py*, *losses.py*"""

from __future__ import annotations

from copy import deepcopy
from typing import Tuple, Union, Any
import torch

from tedeous.derivative import Derivative
from tedeous.points_type import Points_type
from tedeous.eval import Operator, Bounds
from tedeous.losses import Losses
from tedeous.device import device_type, check_device
from tedeous.input_preprocessing import lambda_prepare, Equation_NN, Equation_mat, Equation_autograd


flatten_list = lambda t: [item for sublist in t for item in sublist]

class Solution():
    """
    class for different loss functions calculation.
    """

    def __init__(
        self,
        grid: torch.Tensor,
        equal_cls: Union[Equation_NN, Equation_mat, Equation_autograd],
        model: Union[torch.nn.Sequential, torch.Tensor],
        mode: str,
        weak_form: Union[None, list[callable]],
        lambda_operator,
        lambda_bound,
        tol: float = 0,
        derivative_points: int = 2,
        batch_size: int = None):
        """
        Initializes the Solution class, preparing the components needed to solve the differential equation.
        
                This involves setting up the computational grid, defining the equation to be solved,
                constructing the neural network model, and configuring the loss functions. The class
                also handles the preparation of boundary conditions and the operator, ensuring that
                all necessary elements are in place for the solution process.
        
                Args:
                    grid (torch.Tensor): Discretization of the computational domain.
                    equal_cls (Union[Equation_NN, Equation_mat, Equation_autograd]): Equation object defining the differential equation.
                    model (Union[torch.nn.Sequential, torch.Tensor]): Neural network model used to approximate the solution.
                    mode (str): Specifies the solution mode (*mat, NN, or autograd*).
                    weak_form (Union[None, list[callable]]): List of basis functions for weak form solutions, if applicable.
                    lambda_operator: Regularization parameter for the operator term in the loss function.
                    lambda_bound: Regularization parameter for the boundary term in the loss function.
                    tol (float, optional): Penalty value used in the *casual loss* calculation. Defaults to 0.
                    derivative_points (int, optional): Number of points used for derivative calculation. Defaults to 2.
                    batch_size (int, optional): Size of the batches used during training. Defaults to None.
        """

        self.grid = check_device(grid)
        if mode == 'NN':
            sorted_grid = Points_type(self.grid).grid_sort()
            self.n_t = len(sorted_grid['central'][:, 0].unique())
            self.n_t_operation = lambda sorted_grid: len(sorted_grid['central'][:, 0].unique())
        elif mode == 'autograd':
            self.n_t = len(self.grid[:, 0].unique())
            self.n_t_operation = lambda grid: len(grid[:, 0].unique())
        elif mode == 'mat':
            self.n_t = grid.shape[1]
            self.n_t_operation = lambda grid: grid.shape[1]
        
        equal_copy = deepcopy(equal_cls)
        prepared_operator = equal_copy.operator_prepare()
        self._operator_coeff(equal_cls, prepared_operator)
        self.prepared_bconds = equal_copy.bnd_prepare()
        self.model = model.to(device_type())
        self.mode = mode
        self.weak_form = weak_form
        self.lambda_operator = lambda_operator
        self.lambda_bound = lambda_bound
        self.tol = tol
        self.derivative_points = derivative_points
        self.batch_size = batch_size
        if self.batch_size is None:
            self.n_t_operation = None
        

        self.operator = Operator(self.grid, prepared_operator, self.model,
                                   self.mode, weak_form, derivative_points, 
                                   self.batch_size)
        self.boundary = Bounds(self.grid,self.prepared_bconds, self.model,
                                   self.mode, weak_form, derivative_points)

        self.loss_cls = Losses(self.mode, self.weak_form, self.n_t, self.tol, 
                               self.n_t_operation) # n_t calculate for each batch 
        self.op_list = []
        self.bval_list = []
        self.loss_list = []

    @staticmethod
    def _operator_coeff(equal_cls: Any, operator: list):
        """
        Checks and transfers the coefficients of the differential equation operator to the appropriate device (CPU or GPU). This ensures that the coefficients are compatible with the device being used for neural network computations, preventing potential errors during the training or evaluation process.
        
                Args:
                    equal_cls (Any): Equation_{NN, mat, autograd} object containing the differential equation definition.
                    operator (list): Prepared operator (result of operator_prepare()) representing the terms of the equation.
        
                Returns:
                    None
        """
        for i, _ in enumerate(equal_cls.operator):
            eq = equal_cls.operator[i]
            for key in eq.keys():
                if isinstance(eq[key]['coeff'], torch.nn.Parameter):
                    try:
                        operator[i][key]['coeff'] = eq[key]['coeff'].to(device_type())
                    except:
                        operator[key]['coeff'] = eq[key]['coeff'].to(device_type())
                elif isinstance(eq[key]['coeff'], torch.Tensor):
                    eq[key]['coeff'] = eq[key]['coeff'].to(device_type())

    def _model_change(self, new_model: torch.nn.Module) -> None:
        """
        Updates the neural network model used for solving the differential equation, ensuring consistency across different components. This method is called during cache updates to propagate the retrained model to the operator and boundary conditions.
        
                Args:
                    new_model (torch.nn.Module): The updated neural network model.
        
                Returns:
                    None
        """
        self.model = new_model
        self.operator.model = new_model
        self.operator.derivative = Derivative(new_model, self.derivative_points).set_strategy(
            self.mode).take_derivative
        self.boundary.model = new_model
        self.boundary.operator = Operator(self.grid,
                                          self.prepared_bconds,
                                          new_model,
                                          self.mode,
                                          self.weak_form,
                                          self.derivative_points,
                                          self.batch_size)

    def evaluate(self,
                 save_graph: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the loss function for evaluating the neural network's solution to the differential equation.
        
                This method calculates the loss by comparing the network's output against the governing equations and boundary conditions.
                The computed loss provides a measure of how well the neural network approximates the true solution of the differential equation.
                It is a crucial step in the training process, guiding the network to refine its parameters and improve its accuracy in solving the equation.
        
                Args:
                    save_graph (bool, optional): Determines whether to save the computational graph for visualization or debugging. Defaults to True.
        
                Returns:
                    Tuple[torch.Tensor, torch.Tensor]: A tuple containing the computed loss and the normalized loss.
        """
        self.op = self.operator.operator_compute()
        self.bval, self.true_bval,\
            self.bval_keys, self.bval_length = self.boundary.apply_bcs()
        dtype = self.op.dtype
        self.lambda_operator = lambda_prepare(self.op, self.lambda_operator).to(dtype)
        self.lambda_bound = lambda_prepare(self.bval, self.lambda_bound).to(dtype)

        self.loss, self.loss_normalized = self.loss_cls.compute(
            self.op,
            self.bval,
            self.true_bval,
            self.lambda_operator,
            self.lambda_bound,
            save_graph)
        if self.batch_size is not None: 
            if self.operator.current_batch_i == 0: # if first batch in epoch
                self.save_op = self.op
            else:
                self.save_op = torch.cat((self.save_op, self.op), 0) # cat curent losses to previous
            self.operator.current_batch_i += 1
            del self.op
            torch.cuda.empty_cache()

        return self.loss, self.loss_normalized
