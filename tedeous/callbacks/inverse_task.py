import numpy as np
from typing import Union
import torch
import datetime
from tedeous.callbacks.callback import Callback


class InverseTask(Callback):
    """
    Class for printing the parameters during inverse task solution.
    """

    def __init__(self,
                 parameters: dict,
                 info_string_every: Union[int, None] = None):
        """
        Initializes the inverse task with initial parameter guesses and an optional printing frequency.
        
                This setup is crucial for configuring the optimization process, allowing the solver to start from a reasonable initial state and providing a mechanism for monitoring progress during the iterative solution finding.
        
                Args:
                    parameters (dict): Dictionary containing the initial guess for each parameter to be optimized.
                    info_string_every (Union[int, None], optional):  If an integer is provided, the parameters will be printed to the console every `info_string_every` steps. If `None`, parameters will not be printed during the optimization process. Defaults to None.
        
                Returns:
                    None
        """
        super().__init__()
        self.parameters = parameters
        self.info_string_every = info_string_every
    
    def str_param(self):
        """
        Prints the tracked parameters of the neural network at specified intervals during training.
        
                This method is used to monitor the evolution of key parameters within the neural network
                as it learns to approximate the solution of the differential equation. The parameters to be
                tracked are specified during the initialization of the `InverseTask`. The output is printed
                only every `info_string_every` training steps to avoid excessive logging.
        
                Args:
                    None
        
                Returns:
                    None
        """
        if self.info_string_every is not None and self.model.t % self.info_string_every == 0:
            param = list(self.parameters.keys())
            for name, p in self.model.net.named_parameters():
                if name in param:
                    try:
                        param_str += name + '=' + str(p.item()) + ' '
                    except:
                        param_str = name + '=' + str(p.item()) + ' '
            print(param_str)
    
    def on_epoch_end(self, logs=None):
        """
        Performs actions at the end of each epoch.
        
        This method is called to perform specific actions required after each training epoch, 
        such as updating internal parameters or logging relevant information to monitor the training process 
        of the neural network approximating the solution of a differential equation.
        
        Calls the `str_param` method.
        
        Args:
          logs: Contains information about the epoch.
        
        Returns:
          None
        """
        self.str_param()