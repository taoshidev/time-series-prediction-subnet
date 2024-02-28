# developer: taoshi-tdougherty
# Copyright © 2024 Taoshi Inc

import pandas as pd
import numpy as np

from typing import Union

from sklearn.model_selection import TimeSeriesSplit

def cv(data_length, train_size=500, test_size=100):
    """
    Generate custom time series cross-validation splits with fixed train and test sizes.

    Parameters:
    - data_length: int, the total length of the dataset.
    - train_size: int, the fixed size of the training set.
    - test_size: int, the fixed size of the test set.

    Yields:
    - tuples of (train_index, test_index) for each split.
    """
    # Maximum possible splits given the data length, train size, and test size
    max_splits = (data_length - train_size) // test_size
    
    # Generate indices for each split
    for split in range(max_splits):
        start_train = split * test_size
        end_train = start_train + train_size
        start_test = end_train
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