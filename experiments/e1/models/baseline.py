# developer: taoshi-tdougherty
# Copyright © 2024 Taoshi Inc

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin

class Model(BaseEstimator, RegressorMixin):
    """Baseline model, just predicts the last value."""
    def __init__(self, output_size):
        self.output_size = output_size

    def fit(self, x: pd.DataFrame, y: pd.Series):
        pass

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.ones(self.output_size) * x['Close_lag_1'].values[-1]