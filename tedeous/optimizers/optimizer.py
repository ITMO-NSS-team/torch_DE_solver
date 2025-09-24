import torch
from abc import ABC
from typing import Union, Any
from tedeous.optimizers.pso import PSO
from tedeous.optimizers.ngd import NGD
from tedeous.optimizers.CSO import CSO
from tedeous.optimizers.nys_newton_cg import NysNewtonCG
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts


class Optimizer():
    """
    Base class for creating optimizers.
    
         The Optimizer class serves as a base class for implementing various optimization algorithms.
         It encapsulates the optimization logic and provides a common interface for different optimizers.
    
         Class Methods:
             - __init__
             - optimizer_choice
    
         Class Fields:
             optimizer (Union[str, dict]): The optimizer to use.
             params (dict): Parameters for the optimizer.
             gamma (Union[float, None]): Gamma value for learning rate decay.
             decay_every (Union[int, None]): Frequency of learning rate decay.
             cosine_scheduler_patience (Union[float, None]): Patience for cosine annealing scheduler.
    
         Methods:
             __init__:
    Initializes the class with optimizer settings.
    
    Args:
        optimizer: The optimizer to use. Can be a string or a dictionary.
        params: Parameters for the optimizer.
        gamma: Gamma value for learning rate decay.
        decay_every: Frequency of learning rate decay.
        cosine_scheduler_patience: Patience for cosine annealing scheduler.
    
    Returns:
        None
    
             optimizer_choice:
    Setting optimizer. If optimizer is string type, it will get default settings,
    or it may be custom optimizer defined by user.
    
    Args:
        optimizer: optimizer choice (Adam, SGD, LBFGS, PSO).
        learning_rate: determines the step size at each iteration
        while moving toward a minimum of a loss function.
    
    Returns:
        optimzer: ready optimizer.
    """

    def __init__(
            self,
            optimizer: Union[str, dict],
            params: dict,
            gamma: Union[float, None] = None,
            decay_every: Union[int, None] = None,
            cosine_scheduler_patience: Union[float, None] = None):
        """
        Initializes the Optimizer with specified settings for training a neural network to solve differential equations.
        
                The optimizer, its parameters, and learning rate decay settings are configured here.
                These settings are crucial for effectively training the neural network to approximate
                the solution of the differential equation.
        
                Args:
                    optimizer (Union[str, dict]): The optimizer to use (e.g., 'Adam', 'SGD'). Can be a string
                        specifying the optimizer name or a dictionary with optimizer configuration.
                    params (dict): Parameters to be passed to the optimizer, such as learning rate and momentum.
                    gamma (Union[float, None]): Gamma value for learning rate decay, used in schedulers like
                        StepLR. If None, no learning rate decay is applied.
                    decay_every (Union[int, None]): Frequency (in epochs) at which to decay the learning rate.
                        If None, learning rate is not decayed at specific intervals.
                    cosine_scheduler_patience (Union[float, None]): Patience parameter for the CosineAnnealingLR
                        scheduler.  If None, cosine annealing is not used.
        
                Returns:
                    None
        
                Class Fields:
                    optimizer (Union[str, dict]): The optimizer to use.
                    params (dict): Parameters for the optimizer.
                    gamma (Union[float, None]): Gamma value for learning rate decay.
                    decay_every (Union[int, None]): Frequency of learning rate decay.
                    cosine_scheduler_patience (Union[float, None]): Patience for cosine annealing scheduler.
        """
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
        """
        Selects and configures the optimization algorithm for training the neural network model.
        
                This method determines the appropriate optimizer based on the user's choice,
                preparing it for use in the training loop. The selection of the optimizer
                is crucial for effectively minimizing the loss function and achieving accurate
                solutions to the differential equation.
        
                Args:
                    mode (str): Specifies the mode of operation ('NN', 'autograd', or 'mat'),
                        influencing how the optimizer is applied to the model parameters.
                    model (torch.nn.Module): The neural network model whose parameters will be optimized.
        
                Returns:
                    Union[torch.optim.Adam, torch.optim.SGD, torch.optim.LBFGS, PSO, CSO, NysNewtonCG]:
                        The configured optimizer instance, ready for use in training.
                        The type of optimizer returned depends on the `self.optimizer` attribute.
        """

        torch_optim = None

        if self.optimizer == 'Adam':
            torch_optim = torch.optim.Adam
        if self.optimizer == 'AdamW':
            torch_optim = torch.optim.AdamW
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

