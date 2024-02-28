import pandas as pd

def rsi(close: pd.Series, window=14):
    """
    Calculate the relative strength index for a given time series using pandas Series
    """
    delta = close.diff()
    gain = (delta[1:] * (delta[1:] > 0)).rolling(window=window).mean()
    loss = (-delta[1:] * (delta[1:] < 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def rsi_signal(close: pd.Series, window=14, signal_low = 30, signal_high = 70):
    """
    Calculate the relative strength index signal for a given time series using pandas Series
    """
    rsi_values = rsi(close, window)
    return (rsi_values > signal_high).astype(float) - (signal_low < 30).astype(float)
