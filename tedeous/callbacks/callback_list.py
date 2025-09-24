from tedeous.callbacks.callback import Callback

# import tree

class CallbackList(Callback):
    """
    Container abstracting a list of callbacks.
    """

    def __init__(
        self,
        callbacks=None,
        model=None,
        **params,
    ):
        """
        Initializes a container to manage and coordinate a set of callbacks during the neural network training process for solving differential equations.
        
                This class streamlines the interaction with multiple callbacks by allowing them to be invoked simultaneously, ensuring consistent behavior across different stages of the training loop. This is particularly useful for tasks such as monitoring training progress, adjusting learning rates, or saving model checkpoints during the solution of differential equations.
        
                Args:
                    callbacks: A list of `Callback` instances to be managed.
                    model: The `Model` instance associated with these callbacks.
                    **params: Optional parameters that will be passed to each `Callback` via `Callback.set_params`.
        
                Returns:
                    None
        """
        self.callbacks = callbacks if callbacks else []

        if model:
            self.set_model(model)
        if params:
            self.set_params(params)

    def set_model(self, model):
        """
        Sets the Keras model for the callback and its children.
        
        This method propagates the Keras model instance to the callback and all its children,
        ensuring that all components of the training process have access to the model. This is
        crucial for callbacks that need to interact with the model's parameters or structure
        during training, enabling them to monitor, modify, or react to the model's state.
        
        Args:
            model: The Keras model instance.
        
        Returns:
            None.
        """
        super().set_model(model)
        for callback in self.callbacks:
            callback.set_model(model)

    def append(self, callback):
        """
        Appends a callback function to the list of callbacks.
        
        This allows users to extend the training loop with custom logic, such as logging intermediate results or applying specific constraints during the training process of neural network-based differential equation solvers.
        
        Args:
            callback: The callback function to append.
        
        Returns:
            None.
        """
        self.callbacks.append(callback)

    def set_params(self, params):
        """
        Sets the parameters of the training process and ensures all registered callbacks are aware of these parameters.
        
        This ensures that each callback has the necessary information to perform its intended function during the training process, such as early stopping or logging.
        
        Args:
            params (dict): A dictionary containing the parameters of the training process, such as the model, optimizer, and training configuration.
        
        Returns:
            None
        """
        self.params = params
        for callback in self.callbacks:
            callback.set_params(params)

    def on_epoch_begin(self, logs=None):
        """
        Called at the beginning of an epoch to propagate the event to all callbacks.
        
        This ensures that each callback in the list can perform necessary setup or logging
        at the start of every epoch during the training process of the neural network
        used to solve the differential equation.
        
        Args:
            logs (dict, optional): Log data. Defaults to None.
        
        Returns:
            None
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(logs)

    def on_epoch_end(self, logs=None):
        """
        Called at the end of an epoch.
        
        Iterates through the registered callbacks, triggering their `on_epoch_end` methods to allow them to perform any necessary actions at the end of each training epoch, such as logging metrics or adjusting hyperparameters, contributing to the overall training and evaluation process of the neural network-based differential equation solver.
        
        Args:
            logs (dict, optional): Metric results for this epoch. Defaults to None.
        
        Returns:
            None
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(logs)

    def on_train_begin(self, logs=None):
        """
        Called at the beginning of training.
        
        This method iterates through the registered callbacks and calls their `on_train_begin` methods,
        passing the training logs. This ensures that all callbacks are initialized and prepared
        before the training process starts, allowing them to track and respond to the training state.
        
        Args:
          logs: Dictionary of logs.
        
        Returns:
          None
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        """
        Propagates the `on_train_end` event to all callbacks.
        
        This method is called at the very end of the training process. It ensures that all registered callbacks are informed about the completion of training, allowing them to perform any necessary cleanup or finalization steps, such as logging final metrics or saving model checkpoints.
        
        Args:
            logs (dict, optional): Dictionary of logs collected during training. Defaults to None.
        
        Returns:
            None
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)
