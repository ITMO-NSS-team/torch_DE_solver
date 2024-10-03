import torch
from abc import ABC
from typing import Union, Any
from tedeous.optimizers.pso import PSO
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import Adam, SGD, LBFGS, RMSprop, AdamW


class Optimizer():
    def __init__(
            self,
            optimizer: str,
            params: dict,
            gamma: Union[float, None] = None,
            decay_every: Union[int, None] = None):
        self.optimizer = optimizer
        self.params = params
        self.gamma = gamma
        self.decay_every = decay_every

    def optimizer_choice(
            self,
            mode,
            model) -> \
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

        if self.optimizer == 'Adam':
            torch_optim = Adam
        elif self.optimizer == 'AdamW':
            torch_optim = AdamW
        elif self.optimizer == 'SGD':
            torch_optim = SGD
        elif self.optimizer == 'LBFGS':
            torch_optim = LBFGS
        elif self.optimizer == 'PSO':
            torch_optim = PSO
        elif self.optimizer == 'RMSprop':
            torch_optim = RMSprop

        if mode in ('NN', 'autograd'):
            optimizer = torch_optim(model.parameters(), **self.params)
        elif mode == 'mat':
            optimizer = torch_optim([model.requires_grad_()], **self.params)

        if self.gamma is not None:
            self.scheduler = ExponentialLR(optimizer, gamma=self.gamma)

        return optimizer










