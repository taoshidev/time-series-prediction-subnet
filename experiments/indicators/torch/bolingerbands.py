# developer: taoshi-tdougherty
# Copyright © 2024 Taoshi Inc

import torch
import numpy as np
import pandas as pd

from torch import nn
import pandas as pd

def bolingerbands(close: torch.FloatTensor, window=20, num_std=2):
    """
    Calculate the bolinger bands for a given time series using torch tensors
    """
    rolling_mean = close.rolling(window=window).mean()
    rolling_std = close.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def bolingerbands(close: pd.Series, window=20, num_std=2):
    """
    Calculate the bolinger bands for a given time series using pandas series
    """
    rolling_mean = close.rolling(window=window).mean()
    rolling_std = close.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def volatility_bbp(close: torch.FloatTensor, window=20, num_std=2):
    """
    Calculate the bolinger bands percentage for a given time series
    """
    upper_band, lower_band = bolingerbands(close, window, num_std)
    return (close - lower_band) / (upper_band - lower_band)

