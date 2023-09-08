import os


class ValiConfig:
    SCALE_FACTOR = 0.0001
    HISTORICAL_DATA_LOOKBACK_DAYS_MIN = 5
    HISTORICAL_DATA_LOOKBACK_DAYS_MAX = 10
    PREDICTIONS_MIN = 10
    PREDICTIONS_MAX = 50
    DELETE_STALE_DATA = 180
    BASE_DIR = base_directory = os.path.dirname(os.path.abspath(__file__))


