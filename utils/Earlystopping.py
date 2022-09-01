"""
======================================================================================================================
0. Introduction
This script aims to stop training when the evaluated metrics meet pre-defined requirements:
    - Generalization errors (validation scores) keep decreasing after [patience] epochs
    - Validation scores or training losses lower than particular values.
By FAN FAN (s192217@dtu.dk)
======================================================================================================================
1. Load Package
Loading the necessary package:
torch, numpy
======================================================================================================================
2. Steps
======================================================================================================================
"""
import torch
import numpy as np

class EarlyStopping:
    """Early-stopping is a regularization method to avoid over-fitting when training a learner.
    It stops the training if validation loss doesn't improve after a given patience.
     The idea is introduced in
    'On Early Stopping in Gradient Descent Learning'
    https://link.springer.com/article/10.1007/s00365-006-0663-2
    Parameters
    ----------
    patience : int
        Number of iterations to stop. Default to 100.
    verbose : bool
        Whether to show messages about running details. Default to 0.
    delta : float
        Minimum change in the monitored quantity as an improvement. Default to 0.
    path : str
        Path for checkpoint to be saved to. Default: 'E:\\Deep Learning\\checkpoint.pt'
    trace_func : function
        Print trace messages. Default to print.

    Returns
    ----------
    """
    def __init__(self, patience = 100, verbose = 0, delta = 0, path = 'checkpoint.pt', trace_func = print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        """ Define the function to evaluate whether validation score meet the first requirement
        Parameters
        ----------
        val_loss : float
            Validation loss obtained in current epoch.
        model : pytorch.object
            Model for training.
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """ Save the model
        ----------
        val_loss : float
            Validation loss obtained in current epoch.
        model : pytorch.object
            Model for training.
        """
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.path))
        return model