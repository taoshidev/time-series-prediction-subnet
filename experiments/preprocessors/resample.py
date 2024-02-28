# developer: taoshi-tdougherty
# Copyright © 2024 Taoshi Inc

from sklearn.preprocessing import FunctionTransformer

def resample(data):
    # 1. Upsample to 5m from 1m
    data = data.resample("5min").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    })
    return data

resample_transformer = FunctionTransformer(resample)