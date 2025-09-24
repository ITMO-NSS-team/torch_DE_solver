# Code for early stopping is copied from https://github.com/lululxvi/deepxde/blob/master/deepxde/callbacks.py

import os
import numpy as np
import torch
from tedeous.callbacks.callback import Callback


def create_directory_if_not_exists(file_path):
    """
    Creates the necessary directory structure for a given file path, ensuring that the directory exists before any file operations are performed.
    
        Args:
            file_path (str): The path to the file. The directory containing this file will be created if it doesn't exist.
    
        Returns:
            None
    
        Why: This ensures that the program can write output files to the specified location, especially when dealing with dynamically generated paths or when the target directory might not exist beforehand. This is crucial for saving the trained models or generated data during the differential equation solving process.
    """
    dir_path = os.path.dirname(file_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class EarlyStopping(Callback):
    """
    Halt training when a monitored metric, such as training or validation loss, ceases to improve. This check is performed at each validation step during model training.
    
    
    Args:
        min_delta: Minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: Number of epochs with no improvement
            after which training will be stopped.
        baseline: Baseline value for the monitored quantity to reach.
            Training will stop if the model doesn't show improvement
            over the baseline.
        monitor: The loss function that is monitored. Either 'loss_train' or 'loss_test'
    """


    def __init__(self, min_delta=0, patience=0):
        """
        Initializes an instance of the EarlyStopping class.
        
        This method configures the early stopping criteria based on the specified patience and minimum delta.
        Early stopping is used to prevent overfitting and improve generalization by monitoring the training process
        and stopping when the model's performance on a validation set stops improving.
        
        Args:
            min_delta (float, optional): Minimum change in the monitored quantity to qualify as an improvement. Defaults to 0.
            patience (int, optional): Number of epochs with no improvement after which training will be stopped. Defaults to 0.
        
        Returns:
            None
        """
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.best_epoch = 0

        self.monitor_op = np.less
        self.min_delta *= -1

        self.stop_training = False

    def on_train_begin(self, logs=None):
        """
        Initializes the early stopping mechanism at the start of training.
        
                This method sets up the internal state of the `EarlyStopping` callback,
                preparing it to track the monitored metric and determine when to stop
                training to prevent overfitting and improve generalization. It resets the
                wait counter, the epoch at which training stopped, and the best observed
                value of the monitored metric.
        
                Args:
                    logs (dict, optional): Dictionary of training logs. Defaults to None.
        
                Returns:
                    None
        
                Class Fields Initialized:
                    wait (int): Number of epochs waited after the last time the monitored metric improved. Initialized to 0.
                    stopped_epoch (int): The epoch when training was stopped. Initialized to 0.
                    best (float): The best value of the monitored quantity. Initialized to positive infinity if the monitor operation is 'less', otherwise negative infinity.
        
                Why:
                    This initialization ensures that the early stopping logic starts fresh with each training run,
                    allowing the training process to be stopped when the model's performance on a monitored metric
                    plateaus or degrades, thus preventing overfitting and saving computational resources.
        """
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0

        self.best = np.inf if self.monitor_op == np.less else -np.inf

    def on_epoch_end(self, logs=None):
        """
        Called at the end of each training epoch to monitor performance and potentially halt training.
        
        This method checks if the current total loss is the best encountered so far. If it is, the best loss,
        epoch, and model are updated, and the model's state dictionary is saved. If the loss does not improve
        for a specified patience, training is stopped to prevent overfitting and save computational resources
        by avoiding unnecessary epochs.
        
        Args:
            logs (dict, optional): Metric results for the current epoch. Defaults to None.
        
        Returns:
            None
        """
        current = self.model.total_loss
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            self.best_epoch = self.model.epoch
            self.best_model = self.model.AE_model
            if self.model.path_to_plot_model is not None:
                create_directory_if_not_exists(self.model.path_to_plot_model)
                torch.save(self.best_model.state_dict(), self.model.path_to_plot_model)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = self.model.epoch
                self.stop_training = True

    def on_train_end(self, logs=None):
        """
        Finalizes the training process and saves the best model.
        
        This method is called upon completion of the training loop. It announces the epoch
        at which early stopping was triggered, if applicable, and reports the epoch
        where the best model was obtained based on the validation loss.  The state
        dictionary of the best autoencoder model is saved to the specified path,
        allowing for later use or analysis.
        
        Args:
            logs (dict, optional): Metric results from the final epoch of training. Defaults to None.
        
        Returns:
            torch.nn.Module: The best autoencoder model captured during training, based on validation loss.
        
        Why:
        This method is crucial for preserving the best-performing model found during training.
        By saving the model's state dictionary, we ensure that the most accurate solution
        to the differential equation can be retrieved and utilized for further analysis or prediction.
        The reporting of early stopping and best epoch provides valuable insights into the training process.
        """
        if self.stopped_epoch > 0:
            print("Epoch {}: early stopping".format(self.stopped_epoch))
        print("best model captured at epoch {} with loss={:.4f}".format(self.best_epoch, self.best))
        if self.model.path_to_plot_model is not None:
            create_directory_if_not_exists(self.model.path_to_plot_model)
            torch.save(self.model.AE_model.state_dict(), self.model.path_to_plot_model)
        return self.model.AE_model
