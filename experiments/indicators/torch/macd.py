import torch

def macd(close: torch.FloatTensor, window_short=12, window_long=26, window_signal=9):
    """
    Calculate the moving average convergence divergence for a given time series using torch tensors
    """
    short_ema = close.ewm(span=window_short, adjust=False).mean()
    long_ema = close.ewm(span=window_long, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=window_signal, adjust=False).mean()
    return macd, signal