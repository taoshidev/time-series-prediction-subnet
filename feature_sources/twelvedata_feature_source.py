# developer: taoshi-mbrown
# Copyright © 2024 Taoshi Inc
from enum import Enum
from features import FeatureCompaction, FeatureID, FeatureSource
from http import HTTPStatus
from logging import getLogger
import math
import numpy as np
from numpy import ndarray
import os
import requests
from requests import JSONDecodeError
import statistics
import time
from time_util import (
    current_interval_ms,
    datetime,
    time_span_ms,
)
from urllib.parse import urlencode


class TwelveDataField(str, Enum):
    OPEN_TIME = "datetime"

    PRICE_OPEN = "open"
    PRICE_CLOSE = "close"
    PRICE_HIGH = "high"
    PRICE_LOW = "low"
    VOLUME = "volume"


class TwelveDataTimeSeriesFeatureSource(FeatureSource):
    SOURCE_NAME = "TwelveDataTimeSeries"

    DEFAULT_RETRIES = 3
    RETRY_DELAY = 1.0

    _QUERY_LIMIT = 5000

    _URL = "https://api.twelvedata.com/time_series"

    _INTERVALS = {
        time_span_ms(minutes=1): "1min",
        time_span_ms(minutes=5): "5min",
        time_span_ms(minutes=15): "15min",
        time_span_ms(minutes=30): "30min",
        time_span_ms(minutes=45): "45min",
        time_span_ms(hours=1): "1h",
        time_span_ms(hours=2): "2h",
        time_span_ms(hours=4): "4h",
        time_span_ms(days=1): "1day",
        time_span_ms(weeks=1): "1week",
        time_span_ms(days=30): "1month",
    }

    # Default is using last sample, so on only include other types
    _METRIC_COMPACTIONS = {
        TwelveDataField.PRICE_OPEN: FeatureCompaction.FIRST,
        TwelveDataField.PRICE_HIGH: FeatureCompaction.MAX,
        TwelveDataField.PRICE_LOW: FeatureCompaction.MIN,
        TwelveDataField.VOLUME: FeatureCompaction.SUM,
    }

    def __init__(
        self,
        symbol: str,
        source_interval_ms: int,
        feature_mappings: dict[FeatureID, TwelveDataField],
        feature_dtypes: list[np.dtype] = None,
        default_dtype: np.dtype = np.dtype(np.float32),
        api_key: str = None,
        exchange: str = None,
        country: str = None,
        asset_class: str = None,
        retries: int = DEFAULT_RETRIES,
    ):
        query_interval = self._INTERVALS.get(source_interval_ms)
        if query_interval is None:
            raise ValueError(f"interval_ms {source_interval_ms} is not supported.")

        feature_ids = list(feature_mappings.keys())
        self.VALID_FEATURE_IDS = feature_ids
        super().__init__(feature_ids, feature_dtypes, default_dtype)

        if api_key is None:
            api_key = os.environ.get("TD_API_KEY")

        headers = {"Authorization": f"apikey {api_key}"}

        query_parameters = {
            "order": "ASC",
            "interval": query_interval,
            "outputsize": self._QUERY_LIMIT,
            "timezone": "UTC",
            "symbol": symbol,
        }

        if exchange:
            query_parameters["exchange"] = exchange
        if country:
            query_parameters["country"] = country
        if asset_class:
            query_parameters["type"] = asset_class

        self._source_interval_ms = source_interval_ms
        self._feature_mappings = feature_mappings
        self._metrics = list(feature_mappings.values())
        self._convert_metrics = [TwelveDataField.OPEN_TIME, *self._metrics]
        self._headers = headers
        self._query_parameters = query_parameters
        self._retries = retries
        self._logger = getLogger(self.__class__.__name__)

    # noinspection PyMethodMayBeStatic
    def _convert_metric(self, metric: str, value):
        match metric:
            case TwelveDataField.OPEN_TIME:
                value = datetime.parse(value).timestamp_ms()
            case _:
                if value is None:
                    value = 0
                else:
                    value = float(value)
        return value

    def _convert_sample(self, sample: dict) -> None:
        for metric in self._convert_metrics:
            sample[metric.value] = self._convert_metric(metric, sample[metric])

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
        _OPEN_TIME = TwelveDataField.OPEN_TIME

        # LunarCrush uses open time for queries
        query_start_time_ms = start_time_ms - interval_ms

        # Align on interval so queries for 1 sample include at least 1 sample
        query_start_time_ms = current_interval_ms(
            query_start_time_ms, self._source_interval_ms
        )

        end_time_ms = start_time_ms + (interval_ms * (sample_count - 1))

        query_parameters = self._query_parameters.copy()
        data_rows = []
        retries = self._retries
        # Loop for pagination
        while query_start_time_ms < end_time_ms:
            samples_left = (
                end_time_ms - query_start_time_ms
            ) / self._source_interval_ms
            samples_left = int(min(samples_left, self._QUERY_LIMIT))

            query_end_time_ms = query_start_time_ms + (
                self._source_interval_ms * (samples_left - 1)
            )

            start_date = datetime.fromtimestamp_ms(query_start_time_ms)
            end_date = datetime.fromtimestamp_ms(query_end_time_ms)

            query_parameters["start_date"] = str(start_date)
            query_parameters["end_date"] = str(end_date)
            url = self._URL + "?" + urlencode(query_parameters)

            success = False
            # Loop for retries
            while True:
                try:
                    response = requests.get(url, headers=self._headers)

                    if response.status_code >= HTTPStatus.BAD_REQUEST:
                        try:
                            error_response = response.json()
                            error_message = error_response.get("message")
                            server_error = f", TwelveData error: {error_message}"
                        except JSONDecodeError:
                            server_error = ""
                        self._logger.error(
                            f"HTTP error {response.status_code}: {response.reason}"
                            f"{server_error}",
                        )
                    else:
                        response_data = response.json()
                        response_status = response_data["status"]

                        if response_status != "ok":
                            error_message = response_data.get("message")
                            self._logger.error(
                                f"TwelveData error: {response_status}, {error_message}",
                            )

                        response_rows = response_data.get("values", [])
                        data_rows.extend(response_rows)
                        success = True

                except Exception as e:
                    self._logger.warning(
                        "Exception occurred requesting feature samples using "
                        f"{url}: {e}"
                    )

                if success:
                    break

                if retries == 0:
                    raise RuntimeError("Retries exceeded.")

                retries -= 1
                time.sleep(self.RETRY_DELAY)

            query_start_time_ms = query_end_time_ms + self._source_interval_ms

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
