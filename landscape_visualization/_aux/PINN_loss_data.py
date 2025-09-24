import torch
import numpy as np
from typing import Tuple, Union, Any
from tedeous.input_preprocessing import lambda_prepare


class PINNLossData:
    """
    The PINNLossData class is a base class for calculating the loss in PINN methods.
    
            Class Methods:
            - _loss_op:
    """

    def __init__(self, solution_cls):
        """
        Initializes the SolutionChecker with a Solution class.
        
        This initialization is crucial for setting up the framework to approximate solutions to differential equations using neural networks.
        By storing the Solution class, the checker prepares to evaluate how well the neural network's output matches the expected solution behavior.
        
        Args:
            solution_cls: The Solution class, defining the structure and parameters of the neural network model used to approximate the solution.
        
        Returns:
            None.
        
        Class Fields:
            solution_cls: The Solution class to be used for checking.
        """
        # Храним экземпляр Solution
        self.solution_cls = solution_cls

    # def __getattr__(self, name):
    #     # Делегируем вызовы методов и атрибутов к экземпляру Solution
    #     return getattr(self.solution_cls, name)

    def evaluate(self, save_graph: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates the PINN model by computing the loss based on the defined operator, boundary conditions, and loss type.
        
                This method orchestrates the loss calculation process by preparing the necessary components such as the operator,
                boundary values, and lambda functions. It then dispatches the computation to the appropriate loss function
                (_weak_loss, _causal_loss, or _default_loss) based on the solver's configuration. This evaluation step is crucial
                for training the neural network to accurately approximate the solution of the differential equation.
        
                Args:
                    save_graph (bool, optional): A flag indicating whether to save the computation graph. Defaults to True.
        
                Returns:
                    Tuple[torch.Tensor, torch.Tensor]: A dictionary containing the computed loss values.
        """

        # Используем напрямую атрибуты и методы Solution

        self.op = self.solution_cls.operator.operator_compute()
        self.bval, self.true_bval, \
        self.bval_keys, self.bval_length = self.solution_cls.boundary.apply_bcs()

        dtype = self.op.dtype
        self.lambda_operator = lambda_prepare(self.op.detach(), self.solution_cls.lambda_operator).to(dtype)
        self.lambda_bound = lambda_prepare(self.bval, self.solution_cls.lambda_bound).to(dtype)

        if self.solution_cls.mode in ('mat', 'autograd'):
            if self.bval is None:
                print('No bconds is not possible, returning infinite loss')
                return np.inf

        inputs = [self.op.detach(),
                  self.bval,
                  self.true_bval,
                  self.lambda_operator,
                  self.lambda_bound, ]

        if self.solution_cls.weak_form is not None and self.solution_cls.weak_form != []:
            loss_dict = self._weak_loss(*inputs)
        elif self.solution_cls.tol != 0:
            loss_dict = self._causal_loss(*inputs)
        else:
            loss_dict = self._default_loss(*inputs, save_graph)

        return loss_dict

    def _loss_op(self,
                 operator: torch.Tensor,
                 lambda_op: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the loss term associated with the differential operator.
        
        This function calculates the loss contribution from how well the neural network satisfies the differential equation.
        It measures the discrepancy between the network's output and the expected behavior dictated by the equation.
        The loss is computed using the operator's mean squared error (MSE) on the grid points.
        This loss term encourages the network to learn a solution that adheres to the governing differential equation.
        
        Args:
            operator (torch.Tensor): The result of applying the differential operator to the network's output.
                This represents how well the network's solution satisfies the differential equation at each point.
                See `operator_compute()` in the evaluation module for more details.
            lambda_op (torch.Tensor): A regularization parameter that weights the importance of the operator loss term
                in the overall loss function. This allows for balancing the satisfaction of the differential equation
                with other constraints or data fitting terms.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - loss_operator (torch.Tensor): The operator loss term, representing the overall discrepancy between
                  the network's solution and the differential equation.
                - op (torch.Tensor): The mean squared error of the operator on the entire grid. This provides a measure
                  of how well the network satisfies the differential equation across the domain.
        """
        with torch.no_grad():
            if self.solution_cls.weak_form is not None and self.solution_cls.weak_form != []:
                op = operator
            else:
                op = torch.mean(operator ** 2, 0)

            loss_operator = op @ lambda_op.T
        return loss_operator, op

    def _loss_bcs(self,
                  bval: torch.Tensor,
                  true_bval: torch.Tensor,
                  lambda_bound: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the boundary loss, penalizing deviations from the specified boundary conditions.
        
        This loss term ensures that the neural network solution adheres to the constraints imposed
        at the boundaries of the problem domain, guiding the network towards a physically plausible solution.
        
        Args:
            bval (torch.Tensor): Calculated values of boundary conditions predicted by the neural network.
            true_bval (torch.Tensor): True values of boundary conditions as defined by the problem.
            lambda_bound (torch.Tensor): Regularization parameter for the boundary term in the loss function,
                                         allowing control over the strength of the boundary condition enforcement.
        
        Returns:
            loss_bnd (torch.Tensor): The boundary loss term, a scalar representing the magnitude of the boundary condition violation.
            bval_diff (torch.Tensor): Mean squared error (MSE) between the predicted and true boundary values for each boundary condition.
        """
        with torch.no_grad():
            bval_diff = torch.mean((bval - true_bval) ** 2, 0)

            loss_bnd = bval_diff @ lambda_bound.T
        return loss_bnd, bval_diff

    def _default_loss(self,
                      operator: torch.Tensor,
                      bval: torch.Tensor,
                      true_bval: torch.Tensor,
                      lambda_op: torch.Tensor,
                      lambda_bound: torch.Tensor,
                      save_graph: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the loss based on the differential equation and boundary conditions.
        
                This method calculates the overall loss by combining the loss from the operator (differential equation) and the loss from the boundary conditions.
                It also computes a normalized loss for comparison purposes, where the regularization parameters are set to 1.
                The loss is computed to evaluate how well the neural network solution satisfies both the differential equation and the specified boundary conditions.
        
                Args:
                    operator (torch.Tensor): The result of applying the differential operator, calculated by `eval module -> operator_compute()`.
                    bval (torch.Tensor): The calculated values of the boundary conditions.
                    true_bval (torch.Tensor): The true values of the boundary conditions.
                    lambda_op (torch.Tensor): Regularization parameter for the operator term in the loss.
                    lambda_bound (torch.Tensor): Regularization parameter for the boundary term in the loss.
                    save_graph (bool, optional): Whether to save the computational graph. Defaults to True.
        
                Returns:
                    loss_dict (dict): A dictionary containing the total loss, normalized loss, operator loss, boundary loss, operator values, and boundary value differences.
        """

        if bval is None:
            return torch.sum(torch.mean((operator) ** 2, 0))

        loss_oper, op = self._loss_op(operator, lambda_op)
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
        loss_dict = {
            "loss_total": loss,
            "loss_normalized": loss_normalized.detach(),
            "loss_oper": loss_oper.detach(),
            "loss_bnd": loss_bnd.detach(),
            "operator": operator.detach(),
            "bval_diff": bval_diff.detach()
        }
        torch.cuda.empty_cache()
        return loss_dict

    def _causal_loss(self,
                     operator: torch.Tensor,
                     bval: torch.Tensor,
                     true_bval: torch.Tensor,
                     lambda_op: torch.Tensor,
                     lambda_bound: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes a weighted loss, emphasizing earlier time points in the solution of differential equations. This is achieved by weighting the operator loss at each time point based on the cumulative loss up to that point. This approach is particularly useful when the accuracy of the solution is more critical at the initial stages of the simulation.
        
                Args:
                    operator (torch.Tensor): The result of the differential operator applied to the neural network's output.
                        Obtained from the `operator_compute()` method in the evaluation module.
                    bval (torch.Tensor): The calculated values of the boundary conditions, as predicted by the neural network.
                    true_bval (torch.Tensor): The true values of the boundary conditions, used for comparison.
                    lambda_op (torch.Tensor): Regularization parameter for the operator term in the loss function, balancing the
                        importance of satisfying the differential equation.
                    lambda_bound (torch.Tensor): Regularization parameter for the boundary term in the loss function, balancing the
                        importance of satisfying the boundary conditions.
        
                Returns:
                    loss_dict (Dict[str, torch.Tensor]): A dictionary containing the following:
                        - "loss_total" (torch.Tensor): The total loss, combining the operator and boundary losses.
                        - "loss_normalized" (torch.Tensor): The total loss with regularization parameters set to 1, useful for
                          comparing the inherent magnitudes of the operator and boundary losses.
                        - "loss_oper" (torch.Tensor): The operator loss, measuring how well the neural network satisfies the
                          differential equation.
                        - "loss_bnd" (torch.Tensor): The boundary loss, measuring how well the neural network satisfies the
                          boundary conditions.
                        - "operator" (torch.Tensor): The detached operator tensor.
                        - "bval_diff" (torch.Tensor): The detached difference between calculated and true boundary values.
        """

        res = torch.sum(operator ** 2, dim=1).reshape(self.n_t, -1)
        res = torch.mean(res, axis=1).reshape(self.n_t, 1)
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

        loss_dict = {
            "loss_total": loss,
            "loss_normalized": loss_normalized.detach(),
            "loss_oper": loss_oper.detach(),
            "loss_bnd": loss_bnd.detach(),
            "operator": operator.detach(),
            "bval_diff": bval_diff.detach()
        }

        return loss_dict

    def _weak_loss(self,
                   operator: torch.Tensor,
                   bval: torch.Tensor,
                   true_bval: torch.Tensor,
                   lambda_op: torch.Tensor,
                   lambda_bound: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the loss function for training the neural network to approximate the solution of a differential equation.
        
        This method calculates a weighted sum of the loss from the operator (differential equation) and the boundary conditions.
        It aims to minimize the difference between the neural network's output and the true solution, encouraging the network
        to satisfy both the differential equation and the given boundary conditions. The loss is calculated with regularization
        parameters and also normalized to provide a scale-invariant measure of the error.
        
        Args:
            operator (torch.Tensor): The result of applying the differential operator to the neural network's output.
                This represents how well the network satisfies the differential equation within the domain.
            bval (torch.Tensor): The neural network's predicted values at the boundary points.
            true_bval (torch.Tensor): The true values of the solution at the boundary points.
            lambda_op (torch.Tensor): The weight (regularization parameter) assigned to the operator loss term.
                This controls the importance of satisfying the differential equation.
            lambda_bound (torch.Tensor): The weight (regularization parameter) assigned to the boundary condition loss term.
                This controls the importance of satisfying the boundary conditions.
        
        Returns:
            loss_dict (dict): A dictionary containing the total loss, normalized loss (with regularization parameters set to 1),
                operator loss, boundary loss, the operator tensor, and the difference between predicted and true boundary values.
        """

        if bval is None:
            return sum(operator)

        loss_oper, op = self._loss_op(operator, lambda_op)

        loss_bnd, bval_diff = self._loss_bcs(bval, true_bval, lambda_bound)
        loss = loss_oper + loss_bnd

        lambda_op_normalized = lambda_prepare(operator, 1)
        lambda_bound_normalized = lambda_prepare(bval, 1)

        with torch.no_grad():
            loss_normalized = op @ lambda_op_normalized.T + \
                              bval_diff @ lambda_bound_normalized.T

        loss_dict = {
            "loss_total": loss,
            "loss_normalized": loss_normalized.detach(),
            "loss_oper": loss_oper.detach(),
            "loss_bnd": loss_bnd.detach(),
            "operator": operator.detach(),
            "bval_diff": bval_diff.detach()
        }

        return loss_dict


def get_PINN(layer_sizes, device):
    """
    Creates a Physics-Informed Neural Network (PINN) model.
        
        This method constructs a sequential neural network with Tanh activation
        functions between linear layers. The final layer does not have an
        activation function. The model is then moved to the specified device.
        This architecture is commonly used for solving differential equations
        as it provides a good balance between expressiveness and trainability.
    
        Args:
            layer_sizes (list): A list of integers representing the number of neurons in each layer.
            device (torch.device): The device (CPU or GPU) to which the model should be moved.
    
        Returns:
            torch.nn.Sequential: A PINN model.
    """
    layers = []
    for i, j in zip(layer_sizes[:-1], layer_sizes[1:]):
        layer = torch.nn.Linear(i, j)
        layers.append(layer)
        layers.append(torch.nn.Tanh())
    layers = layers[:-1]
    return torch.nn.Sequential(*layers).to(device)
