from abc import ABC, abstractmethod

class Callback(ABC):
    """
    Base class used to build new callbacks.
    """


    def __init__(self):
        """
        Initializes the KerasClassifierTrainer.
        
        The KerasClassifierTrainer is initialized with default values to prepare for training a neural network that approximates the solution of a differential equation. This setup ensures a clean state for configuring the model, setting verbosity levels, and preparing validation data, which are essential steps in training a neural network to solve differential equations.
        
        Args:
            self: The object instance.
        
        Returns:
            None.
        
        Class Fields:
            print_every (None): Frequency of printing training metrics. Initialized to None.
            verbose (int): Verbosity level during training. Initialized to 0.
            validation_data (None): Validation data to be used during training. Initialized to None.
            _model (None): The Keras model to be trained. Initialized to None.
        """
        self.print_every = None
        self.verbose = 0
        self.validation_data = None
        self._model = None

    def set_params(self, params):
        """
        Sets the parameters used during the training process. These parameters influence how the neural network learns to approximate the solution of the differential equation.
        
                Args:
                    params (dict): A dictionary containing the parameters to be set. These parameters might include learning rate, batch size, or other hyperparameters relevant to the optimization process.
        
                Returns:
                    None
        
                Class Fields:
                    params (dict): A dictionary storing the parameters used by the callback during training.
        """
        self.params = params

    def set_model(self, model):
        """
        Sets the neural network model to be used for solving differential equations.
        
                This method is crucial for associating the solver callback with a specific neural network architecture,
                enabling the training process to approximate solutions to the defined differential equation.
        
                Args:
                    model (torch.nn.Module): The neural network model that will be trained to solve the differential equation.
        
                Returns:
                    None
        """
        self._model = model

    @property
    def model(self):
        """
        Gets the neural network model used for approximating the solution of the differential equation.
                
                This property provides access to the underlying neural network model that is being trained to learn the solution of the differential equation.
                
                Returns:
                    torch.nn.Module: The neural network model.
        """
        return self._model

    def on_epoch_begin(self, logs=None):
        """
        Called at the start of each training epoch.
        
        Subclasses can override this method to implement custom logic that needs to be executed at the beginning of each epoch.
        This is useful for tasks such as adjusting learning rates, logging epoch-specific information, or preparing data for the upcoming epoch
        in the context of training a neural network to solve differential equations.
        
        Args:
            logs (dict, optional): Placeholder for future data logging. Currently unused. Defaults to None.
        """
        pass

    def on_epoch_end(self, logs=None):
        """
        Called at the end of each training epoch.
        
        This method is designed to be overridden by subclasses to implement custom actions
        that should be performed after each epoch during the training phase. It allows for
        monitoring and adapting the training process based on the epoch's performance.
        
        Args:
            epoch (int): The index of the epoch that has just finished.
            logs (dict, optional): A dictionary containing the metrics computed during the epoch.
                This includes training metrics and, if validation is performed, validation metrics
                (prefixed with `val_`). For example: `{'loss': 0.2, 'accuracy': 0.7}`. Defaults to None.
        
        Returns:
            None: This method does not return any value.
        
        Why:
            This method enables the customization of training loops, allowing users to implement
            callbacks for tasks such as saving model checkpoints, adjusting learning rates, or
            performing early stopping based on validation performance, thus tailoring the training
            process for solving differential equations with neural networks.
        """
        pass

    def on_train_begin(self, logs=None):
        """
        Called at the beginning of training.
        
        This method is invoked to set up the necessary configurations and initializations before the training loop commences,
        allowing for custom actions to be executed at the start of the training process.
        Subclasses should override for any actions to run.
        
        Args:
            logs (dict, optional): Placeholder for logging information. Currently, no data is passed to this argument,
              but it may be used in the future. Defaults to None.
        
        Returns:
            None
        """
        pass

    def on_train_end(self, logs=None):
        """
        Called at the end of the training process.
        
        Subclasses should override this method to implement any actions that need to be performed after the training is complete, such as saving the trained model or evaluating its performance. This is useful for post-training analysis and ensuring the trained model is properly stored for future use in solving differential equations.
        
        Args:
            logs (dict, optional):  Contains information about the training process, such as loss values and metrics. It might include the output of the last call to `on_epoch_end()`. Defaults to None.
        
        Returns:
            None
        """
        pass

    def during_epoch(self, logs=None):
        """
        This method is called at the end of each epoch to potentially perform actions based on the training progress.
        
                Args:
                    logs (dict, optional): The logs returned from the previous batch. Defaults to None.
        
                Returns:
                    None
        """
        pass