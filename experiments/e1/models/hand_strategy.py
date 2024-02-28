# developer: taoshi-tdougherty
# Copyright © 2024 Taoshi Inc

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin

class Model(BaseEstimator, RegressorMixin):
    """Hand tuned strategy model. This model is a simple strategy that uses a combination of indicators to make a decision, so it isn't fit to the data."""
    def __init__(self, output_size):
        self.output_size = output_size

    def fit(self, x: pd.DataFrame, y: pd.Series):
        pass

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Objective of this is to make a decision based on the given data"""
        # 1. Trend following - want to first determine a macro trend
        # 2. MACD - moving average convergence divergence
        # 3. RSI - relative strength index
        # 4. Final Augmentation - augment the trend prediction with other indicators
        return np.ones(self.output_size) * x['Close'].values[-1]

        