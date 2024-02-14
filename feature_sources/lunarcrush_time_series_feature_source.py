# developer: taoshi-mbrown
# Copyright © 2024 Taoshi, LLC
import os
import time
from http import HTTPStatus
from json import JSONDecodeError
from logging import getLogger

from enum import Enum
from typing import Dict

import requests

from features import FeatureID, FeatureSource
import numpy as np
from numpy import ndarray


class LunarCrush(str, Enum):
    """
    add in any reusable static values
    """
    TIME = "time"
    HOUR = "hour"
    DAY = "day"

    BUCKET = "bucket"
    INTERVAL = "interval"
    START = "start"
    END = "end"

    ONE_DAY = "1d"
    ONE_WEEK = "1w"
    ONE_MONTH = "1m"
    THREE_MONTHS = "3m"
    SIX_MONTHS = "6m"
    ONE_YEAR = "1y"
    FIVE_YEARS = "5y"
    ALL = "all"

    POSTS_CREATED = "posts_created"
    POSTS_ACTIVE = "posts_active"
    INTERACTIONS = "interactions"
    CONTRIBUTORS_CREATED = "contributors_created"
    CONTRIBUTORS_ACTIVE = "contributors_active"
    SENTIMENT = "sentiment"

    OPEN = "open"
    CLOSE = "close"
    HIGH = "high"
    LOW = "low"
    VOLUME = "volume"
    MARKET_CAP = "market_cap"

    CIRCULATING_SUPPLY = "circulating_supply"
    GALAXY_SCORE = "galaxy_score"
    VOLATILITY = "volatility"
    ALT_RANK = "alt_rank"
    SOCIAL_DOMINANCE = "social_dominance"
    SPAM = "spam"


class LunarCrushTimeSeriesFeatureSource(FeatureSource):
    DEFAULT_RETRIES = 3
    RETRY_DELAY = 1.0

    _BUCKET = [
        LunarCrush.HOUR,
        LunarCrush.DAY
    ]

    _INTERVAL = [
        LunarCrush.ONE_DAY,
        LunarCrush.ONE_WEEK,
        LunarCrush.ONE_MONTH,
        LunarCrush.THREE_MONTHS,
        LunarCrush.SIX_MONTHS,
        LunarCrush.ONE_YEAR,
        LunarCrush.FIVE_YEARS,
        LunarCrush.ALL
    ]

    def __init__(
            self,
            feature_mappings: dict[LunarCrush],
            feature_dtypes: list[np.dtype] = None,
            default_dtype: np.dtype = np.dtype(np.float32),
            api_key: str = None,
            retries: int = DEFAULT_RETRIES,
            **kwargs,
        ):

        feature_ids = list(feature_mappings.keys())
        self.VALID_FEATURE_IDS = feature_ids
        super().__init__(feature_ids, feature_dtypes, default_dtype)

        if api_key is None:
            api_key = os.environ.get("LC_API_KEY")

        self._api_key = api_key
        self._retries = retries
        self._metrics = list(feature_mappings.values())
        self._logger = getLogger(self.__class__.__name__)

    def get_feature_samples(
            self,
            bucket: str,
            endpoint: str,
            start_ms: int = None,
            end_ms: int = None,
            interval: str = None,
        ) -> dict[FeatureID, ndarray]:

        params = {}

        # has to provide a bucket value
        if bucket not in self._BUCKET:
            raise ValueError(f"bucket value [{bucket}] is not supported.")
        else:
            params[LunarCrush.BUCKET] = bucket

        # user can choose to send start/end timestamp or standardized
        # lookback periods (interval)
        if start_ms is None and end_ms is None:
            if interval not in self._INTERVAL:
                raise ValueError(f"start and end ms are not provided"
                            f" and invalid interval provided [{interval}]")
            else:
                params[LunarCrush.INTERVAL] = interval
        elif start_ms is None and end_ms is not None:
            raise ValueError(f"start_ms provided but end_ms not provided.")
        elif end_ms is None and start_ms is not None:
            raise ValueError(f"end_ms provided but start_ms not provided.")
        else:
            params[LunarCrush.START] = start_ms
            params[LunarCrush.END] = end_ms

        data_rows = []
        retries = self._retries

        success = False
        # Loop for retries
        while True:
            try:
                response = requests.get(endpoint, params=params)

                if response.status_code >= HTTPStatus.BAD_REQUEST:
                    try:
                        error_response = response.json()
                        error_message = error_response.get("message")
                        coinbase_error = f", LunarCrush error: {error_message}"
                    except JSONDecodeError:
                        coinbase_error = ""
                        self._logger.error(
                            f"HTTP error {response.status_code}: {response.reason}"
                            f"{coinbase_error}",
                        )
                else:
                    response_rows = response.json()
                    data_rows.extend(response_rows["data"])
                    success = True

            except Exception as e:
                self._logger.warning(
                    "Exception occurred requesting feature samples using "
                    f"{endpoint}: {e}"
                )

            if success or (retries == 0):
                break

            retries -= 1
            time.sleep(self.RETRY_DELAY)

        row_count = len(data_rows)
        if row_count == 0:
            raise RuntimeError("No samples received.")

        feature_samples = self._create_feature_samples(row_count)

        for sample_index in range(row_count):
            row = data_rows[sample_index]
            for feature_index, metric in enumerate(self._metrics):
                feature_samples[feature_index][sample_index] = row[metric]

        results = {
            self.feature_ids[feature_index]: feature_samples[feature_index]
            for feature_index in range(self.feature_count)
        }

        return results


class LunarCrushTimeSeriesTopic(LunarCrushTimeSeriesFeatureSource):
    SOURCE_NAME = "LunarCrushTimeSeriesTopic"

    # have it automatically request from get feature samples
    # and have it not be private
    def query(self, topic: str, bucket: str, start_ms: int, end_ms: int) -> Dict:
        endpoint = f"https://lunarcrush.com/api4/public/{topic}/bitcoin/time-series/v1"
        return self.get_feature_samples(
            bucket=bucket,
            endpoint=endpoint,
            start_ms=start_ms,
            end_ms=end_ms
        )


class LunarCrushTimeSeriesCategory(LunarCrushTimeSeriesFeatureSource):
    SOURCE_NAME = "LunarCrushTimeSeriesCategory"

    # have it automatically request from get feature samples
    # and have it not be private
    def query(self, category: str, bucket: str, start_ms: int, end_ms: int) -> Dict:
        endpoint = f"https://lunarcrush.com/api4/public/category/{category}/time-series/v1"
        return self.get_feature_samples(
            bucket=bucket,
            endpoint=endpoint,
            start_ms=start_ms,
            end_ms=end_ms
        )


class LunarCrushTimeSeriesCoinV2(LunarCrushTimeSeriesFeatureSource):
    SOURCE_NAME = "LunarCrushTimeSeriesCoinV2"

    # have it automatically request from get feature samples
    # and have it not be private
    def query(self, coin_id: int, bucket: str, start_ms: int, end_ms: int) -> Dict:
        endpoint = f"https://lunarcrush.com/api4/public/coins/{coin_id}/time-series/v2"
        return self.get_feature_samples(
            bucket=bucket,
            endpoint=endpoint,
            start_ms=start_ms,
            end_ms=end_ms
        )
