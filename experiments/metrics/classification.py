# developer: taoshi-tdougherty
# Copyright © 2024 Taoshi Inc

import numpy as np
import torch

from numpy import ndarray
from torch import Tensor

from typing import Union, Tuple

def cast_bool(
        y_true: Union[ndarray, Tensor], 
        y_pred: Union[ndarray, Tensor],
        y_mean: Union[ndarray, Tensor] = None
    ) -> Tuple[Union[ndarray, Tensor], Union[ndarray, Tensor]]:
    """
    Cast the true and predicted values to boolean arrays or tensors.
    :param y_true: The true values.
    :param y_pred: The predicted values.
    :param y_mean: The mean values to use for thresholding. If the number is greater than the mean, it is True.
    :return: The true and predicted values as boolean arrays or tensors.
    """
    if y_mean is None:
        if isinstance(y_true, Tensor) or isinstance(y_pred, Tensor):
            return y_true.bool(), y_pred.bool()
        else:
            return y_true.astype(np.bool_), y_pred.astype(np.bool_)
    else:
        if isinstance(y_true, Tensor) or isinstance(y_pred, Tensor):
            return y_true > y_mean, y_pred > y_mean
        else:
            return y_true > y_mean, y_pred > y_mean

def accuracy(y_true: Union[ndarray, Tensor], y_pred: Union[ndarray, Tensor]) -> float:
    """
    Calculate the classification accuracy between the true and predicted values.
    :param y_true: The true values.
    :param y_pred: The predicted values.
    :return: The classification accuracy.
    """
    if not isinstance(y_true, (ndarray, Tensor)) or not isinstance(y_pred, (ndarray, Tensor)):
        raise ValueError("y_true and y_pred must be numpy arrays or torch tensors.")
    
    if not len(y_true) == len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    
    if isinstance(y_true, Tensor) or isinstance(y_pred, Tensor):
        assert y_true.dtype == torch.bool and y_pred.dtype == torch.bool, "y_true and y_pred must be boolean tensors."
        return (y_true == y_pred) / len(y_true) 
    else:
        assert y_true.dtype == np.bool_ and y_pred.dtype == np.bool_, "y_true and y_pred must be boolean arrays."
        return (y_true == y_pred) / len(y_true)

def precision(y_true: Union[ndarray, Tensor], y_pred: Union[ndarray, Tensor]) -> float:
    """
    Calculate the classification precision between the true and predicted values.
    :param y_true: The true values.
    :param y_pred: The predicted values.
    :return: The classification precision.
    """
    if not isinstance(y_true, (ndarray, Tensor)) or not isinstance(y_pred, (ndarray, Tensor)):
        raise ValueError("y_true and y_pred must be numpy arrays or torch tensors.")
    
    if not len(y_true) == len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    
    if isinstance(y_true, Tensor) or isinstance(y_pred, Tensor):
        assert y_true.dtype == torch.bool and y_pred.dtype == torch.bool, "y_true and y_pred must be boolean tensors."
        true_positives = torch.sum(y_true & y_pred)
        predicted_positives = torch.sum(y_pred)
        if predicted_positives == 0:
            return 0.0  # or return np.nan to indicate undefined precision
        
        return true_positives / predicted_positives
    else:
        assert y_true.dtype == np.bool_ and y_pred.dtype == np.bool_, "y_true and y_pred must be boolean arrays."
        true_positives = np.sum(y_true & y_pred)
        predicted_positives = np.sum(y_pred)

        if predicted_positives == 0:
            return 0.0  # or return np.nan to indicate undefined precision
        return true_positives / predicted_positives
    
def recall(y_true: Union[ndarray, Tensor], y_pred: Union[ndarray, Tensor]) -> float:
    """
    Calculate the classification recall between the true and predicted values.
    :param y_true: The true values.
    :param y_pred: The predicted values.
    :return: The classification recall.
    """
    if not isinstance(y_true, (ndarray, Tensor)) or not isinstance(y_pred, (ndarray, Tensor)):
        raise ValueError("y_true and y_pred must be numpy arrays or torch tensors.")
    
    if not len(y_true) == len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    
    if isinstance(y_true, Tensor) or isinstance(y_pred, Tensor):
        assert y_true.dtype == torch.bool and y_pred.dtype == torch.bool, "y_true and y_pred must be boolean tensors."
        true_positives = torch.sum(y_true & y_pred)
        actual_positives = torch.sum(y_true)
        if actual_positives == 0:
            return 0.0
        
        return true_positives / actual_positives
    else:
        assert y_true.dtype == np.bool_ and y_pred.dtype == np.bool_, "y_true and y_pred must be boolean arrays."
        true_positives = np.sum(y_true & y_pred)
        actual_positives = np.sum(y_true)
        if actual_positives == 0:
            return 0.0
        
        return true_positives / actual_positives
    
def f1_score(y_true: Union[ndarray, Tensor], y_pred: Union[ndarray, Tensor]) -> float:
    """
    Calculate the classification F1 score between the true and predicted values.
    :param y_true: The true values.
    :param y_pred: The predicted values.
    :return: The classification F1 score.
    """
    if not isinstance(y_true, (ndarray, Tensor)) or not isinstance(y_pred, (ndarray, Tensor)):
        raise ValueError("y_true and y_pred must be numpy arrays or torch tensors.")
    
    if not len(y_true) == len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    
    if isinstance(y_true, Tensor) or isinstance(y_pred, Tensor):
        assert y_true.dtype == torch.bool and y_pred.dtype == torch.bool, "y_true and y_pred must be boolean tensors."
        precision_score = precision(y_true, y_pred)
        recall_score = recall(y_true, y_pred)
        
        # Check for the case where both precision and recall are zero
        if precision_score + recall_score == 0:
            return 0.0  # Return zero or an indicative value for undefined F1
    
        return 2 * (precision_score * recall_score) / (precision_score + recall_score)
    else:
        assert y_true.dtype == np.bool_ and y_pred.dtype == np.bool_, "y_true and y_pred must be boolean arrays."
        precision_score = precision(y_true, y_pred)
        recall_score = recall(y_true, y_pred)
        
        # Check for the case where both precision and recall are zero
        if precision_score + recall_score == 0:
            return 0.0  # Return zero or an indicative value for undefined F1
        return 2 * (precision_score * recall_score) / (precision_score + recall_score)
    
def confusion_matrix(y_true: Union[ndarray, Tensor], y_pred: Union[ndarray, Tensor]) -> ndarray:
    """
    Calculate the confusion matrix between the true and predicted values.
    :param y_true: The true values.
    :param y_pred: The predicted values.
    :return: The confusion matrix.
    """
    if not isinstance(y_true, (ndarray, Tensor)) or not isinstance(y_pred, (ndarray, Tensor)):
        raise ValueError("y_true and y_pred must be numpy arrays or torch tensors.")
    
    if not len(y_true) == len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    
    if isinstance(y_true, Tensor) or isinstance(y_pred, Tensor):
        assert y_true.dtype == torch.bool and y_pred.dtype == torch.bool, "y_true and y_pred must be boolean tensors."
        true_positives = torch.sum(y_true & y_pred)
        false_positives = torch.sum(~y_true & y_pred)
        true_negatives = torch.sum(~y_true & ~y_pred)
        false_negatives = torch.sum(y_true & ~y_pred)
        return np.array([[true_positives, false_positives], [false_negatives, true_negatives]])
    else:
        assert y_true.dtype == np.bool_ and y_pred.dtype == np.bool_, "y_true and y_pred must be boolean arrays."
        true_positives = np.sum(y_true & y_pred)
        false_positives = np.sum(~y_true & y_pred)
        true_negatives = np.sum(~y_true & ~y_pred)
        false_negatives = np.sum(y_true & ~y_pred)
        return np.array([[true_positives, false_positives], [false_negatives, true_negatives]])