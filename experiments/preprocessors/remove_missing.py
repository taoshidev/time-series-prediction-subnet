# developer: taoshi-tdougherty
# Copyright © 2024 Taoshi Inc

from sklearn.preprocessing import FunctionTransformer

def remove_missing(data):
    # 2. Remove missing values
    data = data.dropna()
    return data

missing_values_transformer = FunctionTransformer(remove_missing)