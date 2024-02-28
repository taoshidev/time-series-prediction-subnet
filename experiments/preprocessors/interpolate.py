# developer: taoshi-tdougherty
# Copyright © 2024 Taoshi Inc

from sklearn.preprocessing import FunctionTransformer
import pandas as pd

def interpolate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate the dataframe.
    :param df: The dataframe.
    :return: The interpolated dataframe.
    """
    # Interpolate the dataframe here
    return df

interpolate_transformer = FunctionTransformer(interpolate)