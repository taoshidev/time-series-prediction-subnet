import numpy as np
from numpy import ndarray
import torch
from torch import Tensor

from typing import Union

def weighted_rmse(y_true: Union[ndarray, Tensor], y_pred: Union[ndarray, Tensor]) -> float:
    """Mimics the scoring mechanism from the ValiConfig class."""
    k = 0.001 # this comes from the current config for the RMSE_WEIGHT in the ValiConfig class
    if isinstance(y_true, Tensor) or isinstance(y_pred, Tensor):
        weights = torch.exp(-k * torch.arange(len(y_pred)))
        weighted_squared_errors = weights * (y_pred - y_true) ** 2
        weighted_rmse = torch.sqrt(torch.sum(weighted_squared_errors) / torch.sum(weights))
        return weighted_rmse.item()
    else:
        weights = np.exp(-k * np.arange(len(y_pred)))
        weighted_squared_errors = weights * (y_pred - y_true) ** 2
        weighted_rmse = np.sqrt(np.sum(weighted_squared_errors) / np.sum(weights))
        return weighted_rmse