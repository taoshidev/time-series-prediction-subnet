import torch

def ewma(data, alpha=0.3):
    """
    Compute exponential weighted moving average (EWMA) of a 1D tensor.
    """
    n = data.size(0)
    weights = torch.pow(alpha, torch.arange(n - 1, -1, -1))
    weights /= weights.sum()
    return torch.conv1d(data.view(1, 1, -1), weights.view(1, 1, -1)).view(-1)