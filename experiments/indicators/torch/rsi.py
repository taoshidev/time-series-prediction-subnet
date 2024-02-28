import torch

def rsi(close: torch.FloatTensor, window=14):
    """
    Calculate the relative strength index for a given time series using torch tensors
    """
    delta = close.diff()
    gain = (delta[1:] * (delta[1:] > 0)).rolling(window=window).mean()
    loss = (-delta[1:] * (delta[1:] < 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def rsi_signal(close: torch.FloatTensor, window=14):
    """
    Calculate the relative strength index signal for a given time series using torch tensors
    """
    rsi_values = rsi(close, window)
    return (rsi_values > 70).float() - (rsi_values < 30).float()