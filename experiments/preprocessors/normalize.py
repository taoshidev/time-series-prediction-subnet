# developer: taoshi-tdougherty
# Copyright © 2024 Taoshi Inc

from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
import pandas as pd

def normalize(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the dataframe.
    :param df: The dataframe.
    :return: The normalized dataframe.
    """
    # Normalize the dataframe here
    return MinMaxScaler().fit_transform(data)
    # return (data - data.mean()) / data.std() # this is another way we might want to normalize the data

normalize_transformer = FunctionTransformer(normalize)