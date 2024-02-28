# developer: taoshi-tdougherty
# Copyright © 2024 Taoshi Inc
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from typing import Union

def mae(y_true: Union[ndarray, Tensor], y_pred: Union[ndarray, Tensor]) -> float:
    """
    Calculate the mean absolute error between the true and predicted values.
    :param y_true: The true values.
    :param y_pred: The predicted values.
    :return: The mean absolute error.
    """
    if not isinstance(y_true, (ndarray, Tensor)) or not isinstance(y_pred, (ndarray, Tensor)):
        raise ValueError("y_true and y_pred must be numpy arrays or torch tensors.")
    
    if isinstance(y_true, Tensor) or isinstance(y_pred, Tensor):
        return torch.mean(torch.abs(y_true - y_pred))
    else:
        return np.mean(np.abs(y_true - y_pred))