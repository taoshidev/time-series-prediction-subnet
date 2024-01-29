# developer: Taoshidev
# Copyright © 2024 Taoshi, LLC
from enum import IntEnum
from features import FeatureCompaction, FeatureSource, FeatureID
from http import HTTPStatus
from logging import getLogger
import math
import numpy as np
from numpy import ndarray
import requests
import statistics
import time
from time_util import time_span_ms


class BybitKlineField(IntEnum):
    OPEN_TIME = 0
    PRICE_OPEN = 1
    PRICE_HIGH = 2
    PRICE_LOW = 3
    PRICE_CLOSE = 4
    VOLUME = 5
    TURNOVER = 6


class BybitKlineFeatureSource(FeatureSource):
    SOURCE_NAME = "BybitKline"

    DEFAULT_RETRIES = 3
    RETRY_DELAY = 1.0

    _QUERY_LIMIT = 1000

    # Must be in ascending order
    _INTERVALS = {
        time_span_ms(minutes=1): "1",
        time_span_ms(minutes=3): "3",
        time_span_ms(minutes=5): "5",
        time_span_ms(minutes=15): "15",
        time_span_ms(minutes=30): "30",
        time_span_ms(hours=1): "60",
        time_span_ms(hours=2): "120",
        time_span_ms(hours=4): "240",
        time_span_ms(hours=6): "360",
        time_span_ms(hours=12): "720",
        time_span_ms(days=1): "D",
        time_span_ms(weeks=1): "W",
        time_span_ms(days=30): "M",
    }

    # Default is using last sample, so on only include other types
    _FIELD_COMPACTIONS = {
        BybitKlineField.PRICE_OPEN: FeatureCompaction.FIRST,
        BybitKlineField.PRICE_HIGH: FeatureCompaction.MAX,
        BybitKlineField.PRICE_LOW: FeatureCompaction.MIN,
        BybitKlineField.VOLUME: FeatureCompaction.SUM,
        BybitKlineField.TURNOVER: FeatureCompaction.SUM,
    }

    _UNCONVERTED_FIELDS = {
        BybitKlineField.PRICE_OPEN,
        BybitKlineField.PRICE_HIGH,
        BybitKlineField.PRICE_LOW,
        BybitKlineField.PRICE_CLOSE,
        BybitKlineField.VOLUME,
        BybitKlineField.TURNOVER,
    }

    def __init__(
        self,
        category: str,
        symbol: str,
        interval_ms: int,
        feature_mappings: dict[FeatureID, BybitKlineField],
        feature_dtypes: list[np.dtype] = None,
        default_dtype: np.dtype = np.dtype(np.float32),
        retries: int = DEFAULT_RETRIES,
    ):
        query_interval = self._INTERVALS.get(interval_ms)
        if query_interval is None:
            raise ValueError(f"interval_ms {interval_ms} is not supported.")

        feature_ids = list(feature_mappings.keys())
        self.VALID_FEATURE_IDS = feature_ids
        super().__init__(feature_ids, feature_dtypes, default_dtype)

        self._category = category
        self._symbol = symbol
        self._interval_ms = interval_ms
        self._query_interval = query_interval
        self._feature_mappings = feature_mappings
        self._fields = list(feature_mappings.values())
        self._retries = retries
        self._logger = getLogger(self.__class__.__name__)

        self._convert_field_indexes = set(self._fields) & self._UNCONVERTED_FIELDS

    def _convert_sample(self, sample: list) -> None:
        for i in self._convert_field_indexes:
            sample[i] = float(sample[i])

    def _convert_samples(self, data_rows: list[list]) -> None:
        _OPEN_TIME = BybitKlineField.OPEN_TIME
        for row in data_rows:
            row[_OPEN_TIME] = int(row[_OPEN_TIME])
            self._convert_sample(row)

    def _compact_samples(self, samples: list[list]) -> list:
        result = samples[-1].copy()
        for field in BybitKlineField:
            compaction = self._FIELD_COMPACTIONS.get(field, FeatureCompaction.LAST)
            if compaction == FeatureCompaction.LAST:
                continue
            elif compaction == FeatureCompaction.FIRST:
                result[field] = samples[0][field]
            else:
                values = [sample[field] for sample in samples]
                match compaction:
                    case FeatureCompaction.MIN:
                        field_result = min(values)
                    case FeatureCompaction.MAX:
                        field_result = max(values)
                    case FeatureCompaction.MEAN:
                        field_result = statistics.mean(values)
                    case FeatureCompaction.MEDIAN:
                        field_result = statistics.median(values)
                    case FeatureCompaction.MODE:
                        field_result = statistics.mode(values)
                    case _:
                        field_result = math.fsum(values)
                result[field] = field_result
        return result

    def get_feature_samples(
        self,
        start_time_ms: int,
        interval_ms: int,
        sample_count: int,
    ) -> dict[FeatureID, ndarray]:
        _OPEN_TIME = BybitKlineField.OPEN_TIME

        # Bybit uses open time for queries
        open_start_time_ms = start_time_ms - interval_ms
        if interval_ms < self._interval_ms:
            open_start_time_ms -= self._interval_ms

        data_rows = []
        retries = self._retries
        samples_left = sample_count
        # Loop for pagination
        while True:
            page_sample_count = min(samples_left, self._QUERY_LIMIT)
            open_end_time_ms = open_start_time_ms + (
                interval_ms * (page_sample_count - 1)
            )

            url = (
                "https://api.bybit.com/v5/market/kline"
                f"?category={self._category}&symbol={self._symbol}"
                f"&start={open_start_time_ms}&end={open_end_time_ms}"
                f"&interval={self._query_interval}&limit={self._QUERY_LIMIT}"
            )

            success = False
            # Loop for retries
            while True:
                try:
                    response = requests.get(url)

                    if response.status_code >= HTTPStatus.BAD_REQUEST:
                        self._logger.error(
                            f"HTTP error {response.status_code}: {response.reason}",
                        )
                    else:
                        response_data = response.json()
                        error_code = response_data.get("retCode")
                        error_message = response_data.get("retMsg")
                        response_result = response_data.get("result")
                        response_rows = response_result.get("list")
                        if response_rows is not None:
                            response_rows.reverse()
                            data_rows.extend(response_rows)
                            success = True
                        else:
                            self._logger.error(
                                f"Bybit error {error_code}: {error_message}",
                            )
                except Exception as e:
                    self._logger.warning(
                        "Exception occurred requesting feature samples using "
                        f"{url}: {e}"
                    )

                if success or (retries == 0):
                    break

                retries -= 1
                time.sleep(self.RETRY_DELAY)

            samples_left -= page_sample_count
            if samples_left <= 0:
                break

            open_start_time_ms = int(data_rows[-1][_OPEN_TIME]) + self._interval_ms

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
                row_time_ms = row[_OPEN_TIME] + self._interval_ms
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

            for feature_index, field in enumerate(self._fields):
                feature_samples[feature_index][sample_index] = row[field]

            interval_rows.clear()
            sample_time_ms += interval_ms

        results = {
            self.feature_ids[feature_index]: feature_samples[feature_index]
            for feature_index in range(self.feature_count)
        }

        return results
