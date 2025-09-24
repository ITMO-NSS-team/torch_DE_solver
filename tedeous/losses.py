"""Module for losses calculation"""

from typing import Tuple, Union
import numpy as np
import torch

from tedeous.input_preprocessing import lambda_prepare


class Losses():
    """
    Class which contains all losses.
    """


    def __init__(self,
                 mode: str,
                 weak_form: Union[None, list],
                 n_t: int,
                 tol: Union[int, float],
                 n_t_operation: callable = None):
        """
        Initializes the loss function with specified parameters for solving differential equations using neural networks. This setup configures how the error between the neural network's approximation and the true solution (or its known properties) is calculated.
        
                Args:
                    mode (str): Calculation mode (*NN*, *autograd*, *mat*) defining the approach for loss computation.
                    weak_form (Union[None, list]): List of basis functions if using a weak formulation of the differential equation.
                    n_t (int): Number of unique time points in the temporal dimension, relevant for time-dependent problems.
                    tol (Union[int, float]): Tolerance value used as a penalty in the *causal loss* calculation.
                    n_t_operation (callable): Function to calculate the number of time points for each batch, useful in dynamic scenarios.
        
                Returns:
                    None: The method initializes the loss function object.
        
                Why:
                    This initialization configures the loss function based on the chosen solution approach (mode),
                    the mathematical formulation of the differential equation (weak_form), and parameters related to
                    the problem's dimensions and acceptable error levels (n_t, tol). The n_t_operation allows for
                    dynamic adjustments during training.
        """

        self.mode = mode
        self.weak_form = weak_form
        self.n_t = n_t
        self.n_t_operation = n_t_operation
        self.tol = tol
        # TODO: refactor loss_op, loss_bcs into one function, carefully figure out when bval
        # is None + fix causal_loss operator crutch (line 76).

    def _loss_op(self,
                 operator: torch.Tensor,
                 forcing_function: torch.Tensor,
                 lambda_op: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the operator loss term, penalizing deviations from the differential equation.
        
        This function calculates the loss associated with how well the neural network satisfies the differential equation.
        It compares the output of the operator (representing the differential equation evaluated by the network)
        with the forcing function (representing the right-hand side of the equation).
        The loss is then scaled by a regularization parameter. This ensures that the neural network's solution
        adheres to the governing differential equation.
        
        Args:
            operator (torch.Tensor): The result of applying the differential operator, as computed by the neural network.
                See `eval` module -> `operator_compute()` for details.
            forcing_function (torch.Tensor): Represents the right-hand side of the differential equation.
            lambda_op (torch.Tensor): Regularization parameter to control the weight of the operator loss term.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - loss_operator (torch.Tensor): The operator loss term, scaled by the regularization parameter.
                - op (torch.Tensor): The mean squared error of the operator on the entire grid.
        """
        if self.weak_form is not None and self.weak_form != []:
            op = operator
        else:
            op = torch.mean((operator - forcing_function) ** 2, 0)

        loss_operator = op @ lambda_op.T
        return loss_operator, op

    def _loss_bcs(self,
                  bval: torch.Tensor,
                  true_bval: torch.Tensor,
                  lambda_bound: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the boundary loss, quantifying how well the neural network satisfies the specified boundary conditions of the differential equation. This loss is crucial for guiding the network to learn solutions that adhere to the problem's constraints at the boundaries.
        
                Args:
                    bval (torch.Tensor): The neural network's predicted values at the boundary points.
                    true_bval (torch.Tensor): The actual (target) values at the boundary points.
                    lambda_bound (torch.Tensor):  A weighting factor to adjust the importance of each boundary condition in the overall loss.
        
                Returns:
                    loss_bnd (torch.Tensor): The calculated boundary loss, representing the weighted error between predicted and actual boundary values.
                    bval_diff (torch.Tensor): The mean squared error (MSE) between the predicted and actual values at each boundary point, providing a measure of individual boundary condition satisfaction.
        """

        bval_diff = torch.mean((bval - true_bval) ** 2, 0)

        loss_bnd = bval_diff @ lambda_bound.T
        return loss_bnd, bval_diff

    def _default_loss(self,
                      operator: torch.Tensor,
                      bval: torch.Tensor,
                      true_bval: torch.Tensor,
                      lambda_op: torch.Tensor,
                      lambda_bound: torch.Tensor,
                      save_graph: bool = True,
                      forcing_function: torch.Tensor = None,
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the overall loss by combining the operator loss and boundary condition loss, with optional regularization.
        
                This function calculates the weighted sum of the operator loss and boundary loss,
                providing a measure of how well the neural network solution satisfies the differential equation
                and its boundary conditions. Regularization parameters allow weighting the importance of
                satisfying the equation versus the boundary conditions.
        
                Args:
                    operator (torch.Tensor): The result of applying the differential operator to the neural network's output.
                    bval (torch.Tensor): The neural network's predicted values at the boundaries.
                    true_bval (torch.Tensor): The true values of the boundary conditions.
                    lambda_op (torch.Tensor): Regularization parameter weighting the operator loss.
                    lambda_bound (torch.Tensor): Regularization parameter weighting the boundary loss.
                    save_graph (bool, optional): Whether to save the computational graph for later analysis. Defaults to True.
                    forcing_function (torch.Tensor): Represents the right-hand side of the differential equation. Defaults to None.
        
                Returns:
                    Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                        - loss (torch.Tensor): The total loss, a weighted sum of operator and boundary losses.
                        - loss_normalized (torch.Tensor): The total loss with regularization parameters set to 1, useful for comparing different problem setups.
        """

        if bval is None:
            return torch.sum(torch.mean((operator) ** 2, 0))

        if forcing_function is None:
            forcing_function = torch.zeros(operator.shape)

        loss_oper, op = self._loss_op(operator, forcing_function, lambda_op)
        dtype = op.dtype
        loss_bnd, bval_diff = self._loss_bcs(bval, true_bval, lambda_bound)
        loss = loss_oper + loss_bnd

        lambda_op_normalized = lambda_prepare(operator, 1).to(dtype)
        lambda_bound_normalized = lambda_prepare(bval, 1).to(dtype)

        with torch.no_grad():
            loss_normalized = op @ lambda_op_normalized.T + \
                              bval_diff @ lambda_bound_normalized.T

        # TODO make decorator and apply it for all losses.
        if not save_graph:
            temp_loss = loss.detach()
            del loss
            torch.cuda.empty_cache()
            loss = temp_loss

        return loss, loss_normalized

    def _causal_loss(self,
                     operator: torch.Tensor,
                     bval: torch.Tensor,
                     true_bval: torch.Tensor,
                     lambda_op: torch.Tensor,
                     lambda_bound: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes a loss function that accounts for the temporal evolution of the solution, weighting the squared error at each time step based on the cumulative error up to that point. This approach is particularly useful for differential equations where the accuracy of the solution at earlier times influences the solution at later times.
        
                Args:
                    operator (torch.Tensor): The result of applying the differential operator, representing the residual of the equation.
                        For more details see eval module -> operator_compute().
                    bval (torch.Tensor): The calculated values of the boundary conditions, obtained from the neural network.
                    true_bval (torch.Tensor): The true values of the boundary conditions, used for comparison.
                    lambda_op (torch.Tensor): Regularization parameter for the operator term in the loss function, controlling the importance of minimizing the equation residual.
                    lambda_bound (torch.Tensor): Regularization parameter for the boundary term in the loss function, controlling the importance of satisfying the boundary conditions.
        
                Returns:
                    loss (torch.Tensor): The total loss, combining the weighted operator loss and the boundary condition loss.
                    loss_normalized (torch.Tensor): The total loss, calculated with regularization parameters set to 1, allowing for comparison of different loss components.
        """
        if self.n_t_operation is not None:  # calculate if batch mod
            self.n_t = self.n_t_operation(operator)
        try:
            res = torch.sum(operator ** 2, dim=1).reshape(self.n_t, -1)
        except:  # if n_t_operation calculate bad n_t then change n_t to batch size
            self.n_t = operator.size()[0]
            res = torch.sum(operator ** 2, dim=1).reshape(self.n_t, -1)
        m = torch.triu(torch.ones((self.n_t, self.n_t), dtype=res.dtype), diagonal=1).T
        with torch.no_grad():
            w = torch.exp(- self.tol * (m @ res))

        loss_oper = torch.mean(w * res)

        loss_bnd, bval_diff = self._loss_bcs(bval, true_bval, lambda_bound)

        loss = loss_oper + loss_bnd

        lambda_bound_normalized = lambda_prepare(bval, 1)
        with torch.no_grad():
            loss_normalized = loss_oper + \
                              lambda_bound_normalized @ bval_diff

        return loss, loss_normalized

    def _weak_loss(self,
                   operator: torch.Tensor,
                   bval: torch.Tensor,
                   true_bval: torch.Tensor,
                   lambda_op: torch.Tensor,
                   lambda_bound: torch.Tensor,
                   forcing_function: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the loss function for the weak formulation of the differential equation.
        
                This method calculates the loss based on the operator's output, boundary values,
                true boundary values, and regularization parameters. It combines the loss from
                the operator and boundary conditions to quantify how well the neural network
                satisfies the differential equation and boundary conditions. The method also
                computes a normalized loss for comparison purposes. This loss function is
                crucial for training the neural network to approximate the solution of the
                differential equation.
        
                Args:
                    operator (torch.Tensor): The result of applying the differential operator,
                        obtained using the `operator_compute` method.
                    bval (torch.Tensor): The calculated values of the solution at the boundaries.
                    true_bval (torch.Tensor): The true (target) values of the solution at the boundaries.
                    lambda_op (torch.Tensor): Regularization parameter for the operator term in the loss function.
                        Controls the weight of the operator loss relative to other loss components.
                    lambda_bound (torch.Tensor): Regularization parameter for the boundary term in the loss function.
                        Controls the weight of the boundary loss relative to other loss components.
                    forcing_function (torch.Tensor, optional): Represents the right-hand side of the differential equation.
                        Defaults to None, which is equivalent to a zero forcing function.
        
                Returns:
                    Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                        - loss (torch.Tensor): The total loss, a combination of the operator loss and boundary loss.
                        - loss_normalized (torch.Tensor): The loss computed with regularization parameters set to 1,
                          allowing for comparison of loss components without regularization effects.
        """

        if bval is None:
            return sum(operator)

        if forcing_function is None:
            forcing_function = torch.zeros(operator.shape)

        loss_oper, op = self._loss_op(operator, forcing_function, lambda_op)

        loss_bnd, bval_diff = self._loss_bcs(bval, true_bval, lambda_bound)
        loss = loss_oper + loss_bnd

        op_dtype = op.dtype
        bval_dtype = bval_diff.dtype

        lambda_op_normalized = lambda_prepare(operator, 1, dtype=op_dtype)
        lambda_bound_normalized = lambda_prepare(bval, 1, dtype=bval_dtype)

        with torch.no_grad():
            loss_normalized = op @ lambda_op_normalized.T + \
                              bval_diff @ lambda_bound_normalized.T

        return loss, loss_normalized

    def compute(self,
                operator: torch.Tensor,
                bval: torch.Tensor,
                true_bval: torch.Tensor,
                lambda_op: torch.Tensor,
                lambda_bound: torch.Tensor,
                save_graph: bool = True) -> Union[_default_loss, _weak_loss, _causal_loss]:
        """
        Selects the appropriate loss calculation method based on the specified mode and form.
        
        This method acts as a dispatcher, choosing between different loss calculation strategies
        depending on whether a weak form is specified or a tolerance level is set. This allows the
        framework to adapt the loss calculation to the specific requirements of the differential
        equation being solved and the desired solution approach.
        
        Args:
            operator (torch.Tensor): The result of the operator calculation, typically obtained from the `operator_compute` method.
            bval (torch.Tensor): Calculated values of boundary conditions.
            true_bval (torch.Tensor): True values of boundary conditions.
            lambda_op (torch.Tensor): Regularization parameter for the operator term in the loss function.
            lambda_bound (torch.Tensor): Regularization parameter for the boundary term in the loss function.
            save_graph (bool, optional): Whether to save the computational graph for later analysis. Defaults to True.
        
        Returns:
            Union[_default_loss, _weak_loss, _causal_loss]: The selected loss calculation method.
        """

        if self.mode in ('mat', 'autograd'):
            if bval is None:
                print('No bconds is not possible, returning infinite loss')
                return np.inf
        inputs = [operator, bval, true_bval, lambda_op, lambda_bound]

        if self.weak_form is not None and self.weak_form != []:
            return self._weak_loss(*inputs)
        elif self.tol != 0:
            return self._causal_loss(*inputs)
        else:
            return self._default_loss(*inputs, save_graph)
