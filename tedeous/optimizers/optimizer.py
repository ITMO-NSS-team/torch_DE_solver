import torch
from abc import ABC
from typing import Union, Any
from tedeous.optimizers.pso import PSO
from tedeous.optimizers.ngd import NGD
from tedeous.optimizers.CSO import CSO
from tedeous.optimizers.nys_newton_cg import NysNewtonCG
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts


class Optimizer():
    def __init__(
            self,
            optimizer: Union[str, dict],
            params: dict,
            gamma: Union[float, None] = None,
            decay_every: Union[int, None] = None,
            cosine_scheduler_patience: Union[float, None] = None):
        self.optimizer = optimizer
        self.params = params
        self.gamma = gamma
        self.decay_every = decay_every
        self.cosine_scheduler_patience = cosine_scheduler_patience

    def optimizer_choice(
        self,
        mode,
        model) -> \
            Union[torch.optim.Adam, torch.optim.SGD, torch.optim.LBFGS, PSO, CSO, NysNewtonCG]:
        """ Setting optimizer. If optimizer is string type, it will get default settings,
            or it may be custom optimizer defined by user.

        Args:
           optimizer: optimizer choice (Adam, SGD, LBFGS, PSO).
           learning_rate: determines the step size at each iteration
           while moving toward a minimum of a loss function.

        Returns:
            optimzer: ready optimizer.
        """

        torch_optim = None

        if self.optimizer == 'Adam':
            torch_optim = torch.optim.Adam
        elif self.optimizer == 'SGD':
            torch_optim = torch.optim.SGD
        elif self.optimizer == 'LBFGS':
            torch_optim = torch.optim.LBFGS
        elif self.optimizer == 'NNCG':
            torch_optim = NysNewtonCG
        elif self.optimizer == 'PSO':
            torch_optim = PSO
        elif self.optimizer == 'NGD':
            torch_optim = NGD
        elif self.optimizer == 'CSO':
            torch_optim = CSO
        elif self.optimizer == 'RMSprop':
            torch_optim = torch.optim.RMSprop

        if mode in ('NN', 'autograd'):
            optimizer = torch_optim(model.parameters(), **self.params)
        elif mode == 'mat':
            optimizer = torch_optim([model.requires_grad_()], **self.params)
        
        if self.gamma is not None:
            self.scheduler = ExponentialLR(optimizer, gamma=self.gamma)

        if self.cosine_scheduler_patience is not None:
            self.scheduler = CosineAnnealingWarmRestarts(optimizer, self.cosine_scheduler_patience)

        return optimizer

