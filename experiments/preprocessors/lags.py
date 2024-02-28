# developer: taoshi-tdougherty
# Copyright © 2024 Taoshi Inc

import pandas as pd
from sklearn.preprocessing import FunctionTransformer

def lags(data: pd.DataFrame, lags: int, column: str = "Close") -> pd.DataFrame:
    """
    Add lags to the dataframe.
    :param data: The dataframe.
    :param lags: The number of lags to add.
    :param column: The column name to add lags for.
    :return: The dataframe with lags added.
    """
    # Add lags here
    for i in range(1, lags + 1):
        data[f"{column}_lag_{i}"] = data[column].shift(i)
    return data

lags_transformer = FunctionTransformer(lags, kw_args={"lags": 1})