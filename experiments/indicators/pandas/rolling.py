import torch

def rolling_window(tensor, window_size):
    """
    Generate rolling windows over a 2D tensor.
    Each window is a view of the original tensor, so no extra memory is allocated.
    
    Args:
    tensor (torch.Tensor): The 1D tensor over which to roll the window.
    window_size (int): The size of the rolling window.
    
    Returns:
    torch.Tensor: A 2D tensor where each row is a window.
    """
    if window_size < 1:
        raise ValueError("`window_size` must be at least 1.")
    if window_size > tensor.size(-1):
        raise ValueError("`window_size` is too large.")
    
    # The size of the rolling window is the last dimension of the shape
    # of the input tensor minus the window size + 1.
    shape = (tensor.size(-1) - window_size + 1, window_size)
    
    # The stride of the rolling window is the last dimension of the input
    # tensor.
    stride = (tensor.stride(-1), tensor.stride(-1))
    
    # Return a new tensor that indexes the input tensor with the rolling
    # window parameters.
    return torch.as_strided(tensor, size=shape, stride=stride)

def ma(tensor, window_size):
    """
    Calculate the moving average over a 2D tensor using a rolling window approach.
    """
    # Use the rolling_window function to create a 2D tensor where each row
    # is a rolling window.
    windows = rolling_window(tensor, window_size)
    
    # Calculate the moving average for each window.
    return windows.mean(dim=-1)