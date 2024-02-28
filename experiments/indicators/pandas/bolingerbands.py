# developer: taoshi-tdougherty
# Copyright © 2024 Taoshi Inc

import torch
import numpy as np
import pandas as pd

from torch import nn
import pandas as pd

def bolingerbands(close: pd.Series, window=20, num_std=2):
    """
    Calculate the Bollinger Bands for a given time series using pandas series.
    """
    rolling_mean = close.rolling(window=window).mean()
    rolling_std = close.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def volatility_bbp(close: pd.Series, window=20, num_std=2):
    """
    Calculate the Bollinger Band Percentage (BBP) for a given time series using pandas series.
    """
    upper_band, lower_band = bolingerbands(close, window, num_std)
    bbp = (close - lower_band) / (upper_band - lower_band)
    return bbp

