# developer: taoshi-tdougherty
# Copyright © 2024 Taoshi Inc

import pandas as pd

def ewma(series: pd.Series, alpha=0.3, adjust=False):
    """
    Compute exponential weighted moving average (EWMA) of a 1D pandas Series.
    """
    return series.ewm(alpha=alpha, adjust=adjust).mean()
