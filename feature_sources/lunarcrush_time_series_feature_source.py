# developer: taoshi-mbrown
# Copyright © 2024 Taoshi Inc
from enum import Enum
from features import FeatureCompaction, FeatureID, FeatureSource
from http import HTTPStatus
from json import JSONDecodeError
from logging import getLogger
import math
import numpy as np
from numpy import ndarray
import os
import requests
import statistics
import time
from time_util import current_interval_ms, time_span_ms
import urllib.parse


class LunarCrushMetric(str, Enum):
    OPEN_TIME = "time"

    POSTS_CREATED = "posts_created"
    POSTS_ACTIVE = "posts_active"
    INTERACTIONS = "interactions"
    CONTRIBUTORS_CREATED = "contributors_created"
    CONTRIBUTORS_ACTIVE = "contributors_active"
    SENTIMENT = "sentiment"
    SPAM = "spam"

    PRICE_OPEN = "open"
    PRICE_CLOSE = "close"
    PRICE_HIGH = "high"
    PRICE_LOW = "low"
    PRICE_FLOOR = "floor_price"
    VOLUME = "volume"
    VOLUME_24H = "volume_24h"
    MARKET_CAP = "market_cap"
    CIRCULATING_SUPPLY = "circulating_supply"
    GALAXY_SCORE = "galaxy_score"

    VOLATILITY = "volatility"
    ALT_RANK = "alt_rank"
    SOCIAL_DOMINANCE = "social_dominance"


class LunarCrushTimeSeriesFeatureSource(FeatureSource):
    DEFAULT_RETRIES = 3
    RETRY_DELAY = 1.0

    _BASE_URL = "https://lunarcrush.com/api4/public"
    _METRIC = "time-series"
    _VERSION = "v1"

    _BUCKETS = {
        time_span_ms(hours=1): "hour",
        time_span_ms(days=1): "day",
    }

    # Default is using last sample, so on only include other types
    _METRIC_COMPACTIONS = {
        LunarCrushMetric.POSTS_CREATED: FeatureCompaction.SUM,
        LunarCrushMetric.POSTS_ACTIVE: FeatureCompaction.SUM,
        LunarCrushMetric.INTERACTIONS: FeatureCompaction.SUM,
        LunarCrushMetric.CONTRIBUTORS_CREATED: FeatureCompaction.SUM,
        LunarCrushMetric.CONTRIBUTORS_ACTIVE: FeatureCompaction.SUM,
        LunarCrushMetric.SPAM: FeatureCompaction.SUM,
        LunarCrushMetric.PRICE_OPEN: FeatureCompaction.FIRST,
        LunarCrushMetric.PRICE_HIGH: FeatureCompaction.MAX,
        LunarCrushMetric.PRICE_LOW: FeatureCompaction.MIN,
        LunarCrushMetric.VOLUME: FeatureCompaction.SUM,
        LunarCrushMetric.VOLUME_24H: FeatureCompaction.SUM,
    }

    def __init__(
        self,
        kind: str,
        selector: str,
        source_interval_ms: int,
        feature_mappings: dict[FeatureID, LunarCrushMetric],
        feature_dtypes: list[np.dtype] = None,
        default_dtype: np.dtype = np.dtype(np.float32),
        api_key: str = None,
        retries: int = DEFAULT_RETRIES,
    ):
        bucket = self._BUCKETS.get(source_interval_ms)
        if bucket is None:
            raise ValueError(f"interval_ms {source_interval_ms} is not supported.")

        feature_ids = list(feature_mappings.keys())
        self.VALID_FEATURE_IDS = feature_ids
        super().__init__(feature_ids, feature_dtypes, default_dtype)

        if api_key is None:
            api_key = os.environ.get("LC_API_KEY")

        headers = {"Authorization": f"Bearer {api_key}"}

        kind = urllib.parse.quote(kind, safe="")
        selector = urllib.parse.quote(selector, safe="")

        self._source_interval_ms = source_interval_ms
        self._bucket = bucket
        self._retries = retries
        self._metrics = list(feature_mappings.values())
        self._convert_metrics = [LunarCrushMetric.OPEN_TIME, *self._metrics]
        self._url = f"{self._BASE_URL}/{kind}/{selector}/{self._METRIC}/{self._VERSION}"
        self._headers = headers
        self._logger = getLogger(self.__class__.__name__)

    # noinspection PyMethodMayBeStatic
    def _convert_metric(self, metric: str, value):
        match metric:
            case LunarCrushMetric.OPEN_TIME:
                value *= time_span_ms(seconds=1)
            case _:
                if value is None:
                    value = 0
                else:
                    value = float(value)
        return value

    def _convert_sample(self, sample: dict) -> None:
        for metric in self._convert_metrics:
            sample_value = sample.get(metric, 0)
            sample[metric.value] = self._convert_metric(metric, sample_value)

    def _convert_samples(self, data_rows: list[dict]) -> None:
        for row in data_rows:
            self._convert_sample(row)

    def _compact_samples(self, samples: list[dict]) -> dict:
        result = samples[-1].copy()
        for metric in self._metrics:
            compaction = self._METRIC_COMPACTIONS.get(metric, FeatureCompaction.LAST)
            if compaction == FeatureCompaction.LAST:
                continue
            elif compaction == FeatureCompaction.FIRST:
                result[metric] = samples[0][metric]
            else:
                values = [sample[metric] for sample in samples]
                match compaction:
                    case FeatureCompaction.MIN:
                        metric_result = min(values)
                    case FeatureCompaction.MAX:
                        metric_result = max(values)
                    case FeatureCompaction.MEAN:
                        metric_result = statistics.mean(values)
                    case FeatureCompaction.MEDIAN:
                        metric_result = statistics.median(values)
                    case FeatureCompaction.MODE:
                        metric_result = statistics.mode(values)
                    case _:
                        metric_result = math.fsum(values)
                result[metric] = metric_result
        return result

    def get_feature_samples(
        self,
        start_time_ms: int,
        interval_ms: int,
        sample_count: int,
    ) -> dict[FeatureID, ndarray]:
        _OPEN_TIME = LunarCrushMetric.OPEN_TIME

        # LunarCrush uses open time for queries
        open_start_time_ms = start_time_ms - interval_ms

        # Align on interval so queries for 1 sample include at least 1 sample
        open_start_time_ms = current_interval_ms(
            open_start_time_ms, self._source_interval_ms
        )

        open_start_time = int(open_start_time_ms / time_span_ms(seconds=1))
        open_end_time_ms = start_time_ms + (interval_ms * (sample_count - 2))
        open_end_time = int(open_end_time_ms / time_span_ms(seconds=1))

        # Ensure that the end time is not the same as the start time
        if open_end_time == open_start_time:
            open_end_time += 1

        query_parameters = {
            "start": open_start_time,
            "end": open_end_time,
            "bucket": self._bucket,
        }
        url = self._url + "?" + urllib.parse.urlencode(query_parameters)

        data_rows = []
        retries = self._retries

        success = False
        # Loop for retries
        while True:
            try:
                response = requests.get(url, headers=self._headers)

                if response.status_code >= HTTPStatus.BAD_REQUEST:
                    try:
                        error_response = response.json()
                        error_message = error_response.get("error")
                        lunarcrush_error = f", LunarCrush error: {error_message}"
                    except JSONDecodeError:
                        lunarcrush_error = ""
                    self._logger.error(
                        f"HTTP error {response.status_code}: {response.reason}"
                        f"{lunarcrush_error}",
                    )
                else:
                    response_rows = response.json()
                    data_rows = response_rows["data"]
                    success = True

            except Exception as e:
                self._logger.warning(
                    "Exception occurred requesting feature samples using " f"{url}: {e}"
                )

            if success or (retries == 0):
                break

            retries -= 1
            time.sleep(self.RETRY_DELAY)

        row_count = len(data_rows)
        if row_count == 0:
            raise RuntimeError("No samples received.")

        self._convert_samples(data_rows)
        feature_samples = self._create_feature_samples(sample_count)

        sample_time_ms = start_time_ms
        interval_rows = []
        row_index = 0
        last_row_index = row_count - 1
        compact_samples = self._compact_samples
        for sample_index in range(sample_count):
            while True:
                row = data_rows[row_index]
                row_time_ms = row[_OPEN_TIME] + self._source_interval_ms
                if row_time_ms > sample_time_ms:
                    break
                interval_rows.append(row)
                if row_index == last_row_index:
                    break
                row_index += 1

            interval_row_count = len(interval_rows)
            if interval_row_count == 1:
                row = interval_rows[0]
            elif interval_row_count > 1:
                row = compact_samples(interval_rows)

            for feature_index, metric in enumerate(self._metrics):
                feature_samples[feature_index][sample_index] = row[metric]

            interval_rows.clear()
            sample_time_ms += interval_ms

        results = {
            self.feature_ids[feature_index]: feature_samples[feature_index]
            for feature_index in range(self.feature_count)
        }

        self._check_feature_samples(results, start_time_ms, interval_ms)

        return results


class LunarCrushTimeSeriesTopic(LunarCrushTimeSeriesFeatureSource):
    SOURCE_NAME = "LunarCrushTimeSeriesTopic"

    def __init__(
        self,
        topic: str,
        source_interval_ms: int,
        feature_mappings: dict[FeatureID, LunarCrushMetric],
        feature_dtypes: list[np.dtype] = None,
        default_dtype: np.dtype = np.dtype(np.float32),
        api_key: str = None,
        retries: int = LunarCrushTimeSeriesFeatureSource.DEFAULT_RETRIES,
    ):
        super().__init__(
            "topic",
            topic,
            source_interval_ms,
            feature_mappings,
            feature_dtypes,
            default_dtype,
            api_key,
            retries,
        )


class LunarCrushTimeSeriesCategory(LunarCrushTimeSeriesFeatureSource):
    SOURCE_NAME = "LunarCrushTimeSeriesCategory"

    def __init__(
        self,
        category: str,
        source_interval_ms: int,
        feature_mappings: dict[FeatureID, LunarCrushMetric],
        feature_dtypes: list[np.dtype] = None,
        default_dtype: np.dtype = np.dtype(np.float32),
        api_key: str = None,
        retries: int = LunarCrushTimeSeriesFeatureSource.DEFAULT_RETRIES,
    ):
        super().__init__(
            "category",
            category,
            source_interval_ms,
            feature_mappings,
            feature_dtypes,
            default_dtype,
            api_key,
            retries,
        )


class LunarCrushTimeSeriesCoin(LunarCrushTimeSeriesFeatureSource):
    SOURCE_NAME = "LunarCrushTimeSeriesCoin"

    _VERSION = "v2"

    def __init__(
        self,
        coin: int | str,
        source_interval_ms: int,
        feature_mappings: dict[FeatureID, LunarCrushMetric],
        feature_dtypes: list[np.dtype] = None,
        default_dtype: np.dtype = np.dtype(np.float32),
        api_key: str = None,
        retries: int = LunarCrushTimeSeriesFeatureSource.DEFAULT_RETRIES,
    ):
        super().__init__(
            "coins",
            str(coin),
            source_interval_ms,
            feature_mappings,
            feature_dtypes,
            default_dtype,
            api_key,
            retries,
        )


class LunarCrushTimeSeriesStock(LunarCrushTimeSeriesFeatureSource):
    SOURCE_NAME = "LunarCrushTimeSeriesStock"

    _VERSION = "v2"

    def __init__(
        self,
        stock: int | str,
        source_interval_ms: int,
        feature_mappings: dict[FeatureID, LunarCrushMetric],
        feature_dtypes: list[np.dtype] = None,
        default_dtype: np.dtype = np.dtype(np.float32),
        api_key: str = None,
        retries: int = LunarCrushTimeSeriesFeatureSource.DEFAULT_RETRIES,
    ):
        super().__init__(
            "stocks",
            str(stock),
            source_interval_ms,
            feature_mappings,
            feature_dtypes,
            default_dtype,
            api_key,
            retries,
        )
