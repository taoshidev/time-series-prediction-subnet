import numpy as np
from numpy import ndarray
import torch
from torch import Tensor

from typing import Union

def rmse(y_true: Union[ndarray, Tensor], y_pred: Union[ndarray, Tensor]) -> float:
    """
    Calculate the root mean squared error between the true and predicted values.
    :param y_true: The true values.
    :param y_pred: The predicted values.
    :return: The root mean squared error.
    """
    if not isinstance(y_true, (ndarray, Tensor)) or not isinstance(y_pred, (ndarray, Tensor)):
        raise ValueError("y_true and y_pred must be numpy arrays or torch tensors.")
    
    if isinstance(y_true, Tensor) or isinstance(y_pred, Tensor):
        return torch.sqrt(torch.mean(torch.square(y_true - y_pred)))
    else:
        return np.sqrt(np.mean(np.square(y_true - y_pred)))