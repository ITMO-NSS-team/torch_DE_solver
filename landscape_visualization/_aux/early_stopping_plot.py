# Code for early stopping is copied from https://github.com/lululxvi/deepxde/blob/master/deepxde/callbacks.py

import os
import numpy as np
import torch
from tedeous.callbacks.callback import Callback


def create_directory_if_not_exists(file_path):
    dir_path = os.path.dirname(file_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class EarlyStopping(Callback):
    """Stop training when a monitored quantity (training or testing loss) has stopped improving.
    Only checked at validation step according to ``display_every`` in ``Model.train``.

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
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.best_epoch = 0

        self.monitor_op = np.less
        self.min_delta *= -1

        self.stop_training = False

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0

        self.best = np.inf if self.monitor_op == np.less else -np.inf

    def on_epoch_end(self, logs=None):
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
        if self.stopped_epoch > 0:
            print("Epoch {}: early stopping".format(self.stopped_epoch))
        print("best model captured at epoch {} with loss={:.4f}".format(self.best_epoch, self.best))
        if self.model.path_to_plot_model is not None:
            create_directory_if_not_exists(self.model.path_to_plot_model)
            torch.save(self.model.AE_model.state_dict(), self.model.path_to_plot_model)
        return self.model.AE_model
