import torch
from abc import ABC
from typing import Union, Any
from tedeous.optimizers import PSO

class Optimizer():
    def __init__(self, optimizer: Union[torch.optim.Optimizer, str], learning_rate:float = 1e-3, **params):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.mode = params.get('mode', 'NN')

    def _optimizer_choice(
        self,
        optimizer: Union[str, Any],
        learning_rate: float) -> \
            Union[torch.optim.Adam, torch.optim.SGD, torch.optim.LBFGS, PSO]:
        """ Setting optimizer. If optimizer is string type, it will get default settings,
            or it may be custom optimizer defined by user.

        Args:
           optimizer: optimizer choice (Adam, SGD, LBFGS, PSO).
           learning_rate: determines the step size at each iteration
           while moving toward a minimum of a loss function.

        Returns:
            optimzer: ready optimizer.
        """

        if optimizer == 'Adam':
            torch_optim = torch.optim.Adam
        elif optimizer == 'SGD':
            torch_optim = torch.optim.SGD
        elif optimizer == 'LBFGS':
            torch_optim = torch.optim.LBFGS
        else:
            torch_optim = optimizer

        if self.mode in ('NN', 'autograd'):
            optimizer = torch_optim(self.model.parameters(), lr=learning_rate)
        elif self.mode == 'mat':
            optimizer = torch_optim([self.model.requires_grad_()], lr=learning_rate)
        return optimizer