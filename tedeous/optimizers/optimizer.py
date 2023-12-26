import torch
from abc import ABC
from typing import Union

class Optimizer():
    def __init__(self, model, optimizer: Union[torch.optim.Optimizer], learning_rate:float = 1e-3, **params):
        self.model = model
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.mode = params.get('mode', 'NN')

    def _optimizer_choice(
        self,
        optimizer,
        learning_rate: float):
        """ Setting optimizer. If optimizer is string type, it will get default settings,
            or it may be custom optimizer defined by user.

        Args:
           optimizer: optimizer choice (Adam, SGD, LBFGS, PSO).
           learning_rate: determines the step size at each iteration
           while moving toward a minimum of a loss function.

        Returns:
            optimizer: ready optimizer.
        """

        if self.mode in ('NN', 'autograd'):
            optimizer = self.optimizer(self.model.parameters(), lr=learning_rate)
        elif self.mode == 'mat':
            optimizer = self.optimizer([self.model.requires_grad_()], lr=learning_rate)

        return optimizer