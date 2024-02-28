# developer: taoshi-tdougherty
# Copyright © 2024 Taoshi Inc

import pandas as pd
import numpy as np

from typing import Union

from sklearn.model_selection import TimeSeriesSplit

def cv(data_length, train_size:int=500, test_size:int=100, gap:int=0):
    """
    Generate custom time series cross-validation splits with fixed train and test sizes,
    including an optional gap between train and test sets to prevent data leakage.

    Parameters:
    - data_length: int, the total length of the dataset.
    - train_size: int, the fixed size of the training set.
    - test_size: int, the fixed size of the test set.
    - gap: int, the number of observations to leave out between training and testing sets.

    Yields:
    - tuples of (train_index, test_index) for each split.
    """
    # Adjust maximum possible splits calculation to account for the gap
    max_splits = (data_length - train_size - gap) // test_size
    
    for split in range(max_splits):
        start_train = split * test_size
        end_train = start_train + train_size
        start_test = end_train + gap  # Start the test set after the gap
        end_test = start_test + test_size
        
        train_index = np.arange(start_train, end_train)
        test_index = np.arange(start_test, end_test)
        
        yield train_index, test_index

def rolling(data, n_splits = 3, test_size = 2, random_state = 42):
    """
    Rolling cross validation function
    """
    return

def expanding(data, n_splits = 3, test_size = 2, random_state = 42):
    """
    Expanding cross validation function
    """
    return