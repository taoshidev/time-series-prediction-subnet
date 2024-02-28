# developer: taoshi-tdougherty
# Copyright © 2024 Taoshi Inc

import torch

def ewma(data, alpha=0.3, adjust=False):
    """
    Compute exponential weighted moving average (EWMA) of a 1D tensor.
    """
    return data.emwa(alpha=alpha, adjust=adjust).mean()