from abc import ABC, abstractmethod

class Callback(ABC):
    """Base class used to build new callbacks.

    Callbacks can be passed to keras methods such as `fit()`, `evaluate()`, and
    `predict()` in order to hook into the various stages of the model training,
    evaluation, and inference lifecycle.

    To create a custom callback, subclass `keras.callbacks.Callback` and
    override the method associated with the stage of interest.

    If you want to use `Callback` objects in a custom training loop:

    1. You should pack all your callbacks into a single `callbacks.CallbackList`
       so they can all be called together.
    2. You will need to manually call all the `on_*` methods at the appropriate
       locations in your loop. Like this:


    Attributes:
        params: Dict. Training parameters
            (eg. verbosity, batch size, number of epochs...).
        model: Instance of `Model`.
            Reference of the model being trained.

    The `logs` dictionary that callback methods
    take as argument will contain keys for quantities relevant to
    the current batch or epoch (see method-specific docstrings).
    """

    def __init__(self):
        self.print_every = None
        self.verbose = 0
        self.validation_data = None
        self._model = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self._model = model

    @property
    def model(self):
        return self._model

    def on_epoch_begin(self, logs=None):
        """Called at the start of an epoch.

        Subclasses should override for any actions to run. This function should
        only be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """
        pass

    def on_epoch_end(self, logs=None):
        """Called at the end of an epoch.

        Subclasses should override for any actions to run. This function should
        only be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict, metric results for this training epoch, and for the
              validation epoch if validation is performed. Validation result
              keys are prefixed with `val_`. For training epoch, the values of
              the `Model`'s metrics are returned. Example:
              `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        pass

    def on_train_begin(self, logs=None):
        """Called at the beginning of training.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """
        pass

    def on_train_end(self, logs=None):
        """Called at the end of training.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently the output of the last call to
              `on_epoch_end()` is passed to this argument for this method but
              that may change in the future.
        """
        pass

    def during_epoch(self, logs=None):
        pass