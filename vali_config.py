# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Taoshi
# Copyright © 2023 Taoshi, LLC

import os


class ValiConfig:
    RMSE_WEIGHT = 0.001
    SCALE_FACTOR_EXP = 0.0001
    SCALE_FACTOR = 100
    SCALE_SHIFT = (1 - 1/SCALE_FACTOR) / 2
    HISTORICAL_DATA_LOOKBACK_DAYS_MIN = 25
    HISTORICAL_DATA_LOOKBACK_DAYS_MAX = 30
    PREDICTIONS_MIN = 100
    PREDICTIONS_MAX = 101
    DELETE_STALE_DATA = 180

    STANDARD_TF = 5

    BASE_DIR = base_directory = os.path.dirname(os.path.abspath(__file__))
