from typing import Tuple, Union

import tedeous.input_preprocessing
from tedeous.utils import *

class Losses():
    """
    Class which contains all losses.
    """
    def __init__(self, operator: dict,
                 bval: dict,
                 true_bval: dict,
                 lambda_op: Union[int, list, dict],
                 lambda_bound: Union[int, list, dict],
                 mode: str,
                 weak_form: Union[None, torch.Tensor],
                 n_t: int,
                 save_graph: bool):
        self.operator = operator
        self.mode = mode
        self.weak_form = weak_form
        self.bval = bval
        self.true_bval = true_bval
        self.lambda_bound = tedeous.input_preprocessing.lambda_prepare(bval, lambda_bound)
        self.lambda_op = tedeous.input_preprocessing.lambda_prepare(operator, lambda_op)
        self.n_t = n_t
        self.save_graph = save_graph
        # TODO: refactor loss_op, loss_bcs into one function, carefully figure out when bval is None + fix causal_loss operator crutch (line 76).

    def loss_op(self, lambda_op) -> torch.Tensor:
        loss_operator = 0
        for eq in self.operator:
            if self.weak_form != None and self.weak_form != []:
                loss_operator += lambda_op[eq] * torch.sum(self.operator[eq])
            else:
                loss_operator += lambda_op[eq] * torch.mean((self.operator[eq]) ** 2)
        return loss_operator


    def loss_bcs(self, lambda_bound) -> torch.Tensor:
        """
        Computes boundary loss for corresponding type.

        Returns:
            boundary loss
        """
        loss_bnd = 0
        for bcs_type in self.bval:
            loss_bnd += lambda_bound[bcs_type] * torch.mean((self.bval[bcs_type] - self.true_bval[bcs_type]) ** 2)
        return loss_bnd


    def default_loss(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes l2 loss.

        Args:
            lambda_bound: an arbitrary chosen constant, influence only convergence speed.
        Returns:
            model loss.
        """

        if self.bval == None:
            return torch.sum(torch.mean((self.operator) ** 2, 0))

        if self.mode == 'mat':
            loss = torch.mean((self.operator) ** 2) + self.loss_bcs(self.lambda_bound)
        else:
            loss = self.loss_op(self.lambda_op) + self.loss_bcs(self.lambda_bound)

        lambda_op_normalized = tedeous.input_preprocessing.lambda_prepare(self.operator, 1)
        lambda_bound_normalized = tedeous.input_preprocessing.lambda_prepare(self.bval, 1)

        with torch.no_grad():
            loss_normalized = self.loss_op(lambda_op_normalized) + self.loss_bcs(lambda_bound_normalized)

        # TODO make decorator and apply it for all losses.
        if not self.save_graph:
            temp_loss = loss.detach()
            del loss
            torch.cuda.empty_cache()
            loss = temp_loss

        return loss, loss_normalized

    def causal_loss(self, tol: float = 0) -> torch.Tensor:
        """
        Computes causal loss, which is calculated with weights matrix:
        W = exp(-tol*(Loss_i)) where Loss_i is sum of the L2 loss from 0
        to t_i moment of time. This loss function should be used when one
        of the DE independent parameter is time.

        Args:
            lambda_bound: an arbitrary chosen constant, influence only convergence speed.
            tol: float constant, influences on error penalty.
        Returns:
            model loss.
        """

        op = torch.hstack(list(self.operator.values())) ** 2

        if self.bval == None:
            return torch.sum(torch.mean((op) ** 2, 0))

        res = torch.sum(op, dim=1).reshape(self.n_t, -1)
        res = torch.mean(res, axis=1).reshape(self.n_t, 1)
        M = torch.triu(torch.ones((self.n_t, self.n_t)), diagonal=1).T
        with torch.no_grad():
            W = torch.exp(- tol * (M @ res))

        loss = torch.mean(W * res) + self.loss_bcs(self.lambda_bound)

        return loss

    def weak_loss(self) -> torch.Tensor:
        """
        Weak solution of O/PDE problem.

        Args:
            weak_form: list of basis functions.
            lambda_bound: const regularization parameter.
        Returns:
            model loss.
        """
        if self.bval == None:
            return sum(list(self.operator.values()))

        # we apply no  boundary conditions operators if they are all None

        loss = self.loss_op(self.lambda_op) + self.loss_bcs(self.lambda_bound)

        return loss

    def compute(self, tol: float = 0) -> \
        Union[default_loss, weak_loss, causal_loss]:
            """
            Setting the required loss calculation method.

            Args:
                tol: float constant, influences on error penalty.
            Returns:
                A given calculation method.
            """
            if self.mode == 'mat' or self.mode == 'autograd':
                if self.bval == None:
                    print('No bconds is not possible, returning infinite loss')
                    return np.inf

            if self.weak_form != None and self.weak_form != []:
                return self.weak_loss()
            elif tol != 0:
                return self.causal_loss(tol=tol)
            else:
                return self.default_loss()