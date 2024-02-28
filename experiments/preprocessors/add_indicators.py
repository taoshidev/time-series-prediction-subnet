# developer: taoshi-tdougherty
# Copyright © 2024 Taoshi Inc

from sklearn.preprocessing import FunctionTransformer
import pandas as pd

def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add indicators to the dataframe.
    :param df: The dataframe.
    :return: The dataframe with indicators added.
    """
    # Add indicators here

    return data

indicators_transformer = FunctionTransformer(add_indicators)