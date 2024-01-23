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
from requests import JSONDecodeError
import statistics
import time
from time_util import time_span_ms


class BinanceKlineField(IntEnum):
    OPEN_TIME = 0
    PRICE_OPEN = 1
    PRICE_HIGH = 2
    PRICE_LOW = 3
    PRICE_CLOSE = 4
    VOLUME = 5
    CLOSE_TIME = 6
    QUOTE_ASSET_VOLUME = 7
    TRADES_COUNT = 8
    TAKER_BUY_VOLUME_BASE = 9
    TAKER_BUY_VOLUME_QUOTE = 10


class BinanceKlineFeatureSource(FeatureSource):
    SOURCE_NAME = "BinanceKline"

    DEFAULT_RETRIES = 3
    RETRY_DELAY = 1.0

    _QUERY_LIMIT = 1000

    # Must be in ascending order
    _INTERVALS = {
        time_span_ms(minutes=1): "1m",
        time_span_ms(minutes=3): "3m",
        time_span_ms(minutes=5): "5m",
        time_span_ms(minutes=15): "15m",
        time_span_ms(minutes=30): "30m",
        time_span_ms(hours=1): "1h",
        time_span_ms(hours=2): "2h",
        time_span_ms(hours=4): "4h",
        time_span_ms(hours=6): "6h",
        time_span_ms(hours=8): "8h",
        time_span_ms(hours=12): "12h",
        time_span_ms(days=1): "1d",
        time_span_ms(days=3): "3d",
        time_span_ms(weeks=1): "1w",
    }

    # Default is using last sample, so on only include other types
    _FIELD_COMPACTIONS = {
        BinanceKlineField.PRICE_OPEN: FeatureCompaction.FIRST,
        BinanceKlineField.PRICE_HIGH: FeatureCompaction.MAX,
        BinanceKlineField.PRICE_LOW: FeatureCompaction.MIN,
        BinanceKlineField.VOLUME: FeatureCompaction.SUM,
        BinanceKlineField.QUOTE_ASSET_VOLUME: FeatureCompaction.SUM,
        BinanceKlineField.TRADES_COUNT: FeatureCompaction.SUM,
        BinanceKlineField.TAKER_BUY_VOLUME_BASE: FeatureCompaction.SUM,
        BinanceKlineField.TAKER_BUY_VOLUME_QUOTE: FeatureCompaction.SUM,
    }

    _UNCONVERTED_FIELDS = {
        BinanceKlineField.PRICE_OPEN,
        BinanceKlineField.PRICE_HIGH,
        BinanceKlineField.PRICE_LOW,
        BinanceKlineField.PRICE_CLOSE,
        BinanceKlineField.VOLUME,
        BinanceKlineField.QUOTE_ASSET_VOLUME,
        BinanceKlineField.TAKER_BUY_VOLUME_BASE,
        BinanceKlineField.TAKER_BUY_VOLUME_QUOTE,
    }

    def __init__(
        self,
        symbol: str,
        feature_mappings: dict[FeatureID, BinanceKlineField],
        feature_dtypes: list[np.dtype] = None,
        default_dtype: np.dtype = np.dtype(np.float32),
        allow_empty_response=True,
        retries: int = DEFAULT_RETRIES,
    ):
        feature_ids = list(feature_mappings.keys())
        self.VALID_FEATURE_IDS = feature_ids
        super().__init__(feature_ids, feature_dtypes, default_dtype)

        self._symbol = symbol
        self._feature_mappings = feature_mappings
        self._fields = list(feature_mappings.values())
        self._allow_empty_response = allow_empty_response
        self._retries = retries
        self._logger = getLogger("BinanceKlineFeatureSource")

        convert_fields = set(self._fields) & self._UNCONVERTED_FIELDS
        self._convert_field_indexes = [field for field in convert_fields]

    # noinspection PyMethodMayBeStatic
    def _convert_sample(self, sample: list) -> list:
        for i in self._convert_field_indexes:
            sample[i] = float(sample[i])
        return sample

    def _convert_samples(self, data_rows: list[list]) -> list[list]:
        return [self._convert_sample(row) for row in data_rows]

    # noinspection PyMethodMayBeStatic
    def _get_empty_converted_samples(self, open_time_ms: int, interval_ms: int) -> list:
        close_time_ms = open_time_ms + interval_ms - 1
        return [open_time_ms, 0, 0, 0, 0, 0, close_time_ms, 0, 0, 0, 0]

    def _compact_samples(self, samples: list[list]) -> list:
        result = samples[-1].copy()
        for field in BinanceKlineField:
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
        query_interval_ms = 0
        for supported_interval_ms in self._INTERVALS.keys():
            if (query_interval_ms == 0) or (supported_interval_ms <= interval_ms):
                query_interval_ms = supported_interval_ms
            else:
                break

        query_interval = self._INTERVALS[query_interval_ms]

        # Binance uses open time for queries
        open_start_time_ms = start_time_ms - interval_ms
        if interval_ms < query_interval_ms:
            open_start_time_ms -= query_interval_ms

        open_end_time_ms = start_time_ms + (interval_ms * (sample_count - 2))

        data_rows = []
        retries = self._retries
        # Loop for pagination
        while True:
            url = (
                "https://api.binance.com/api/v3/klines"
                f"?symbol={self._symbol}&interval={query_interval}&startTime={open_start_time_ms}"
                f"&endTime={open_end_time_ms}&limit={self._QUERY_LIMIT}"
            )

            success = False
            response_row_count = 0
            # Loop for retries
            while True:
                try:
                    response = requests.get(url)

                    if response.status_code >= HTTPStatus.BAD_REQUEST:
                        try:
                            error_response = response.json()
                            error_code = error_response.get("code")
                            error_msg = error_response.get("msg")
                            binance_error = f", Binance error {error_code}: {error_msg}"
                        except JSONDecodeError:
                            binance_error = ""
                        self._logger.error(
                            f"HTTP error {response.status_code}: {response.reason}"
                            f"{binance_error}",
                        )
                    else:
                        response_rows = response.json()
                        response_row_count = len(response_rows)
                        data_rows.extend(response_rows)
                        success = True

                except:
                    # TODO: Logging
                    pass

                if success or (retries == 0):
                    break

                retries -= 1
                time.sleep(self.RETRY_DELAY)

            if response_row_count != self._QUERY_LIMIT:
                break

            open_start_time_ms = data_rows[-1][BinanceKlineField.CLOSE_TIME] + 1

        row_count = len(data_rows)
        if row_count == 0:
            if self._allow_empty_response:
                converted_samples = self._get_empty_converted_samples(
                    open_start_time_ms, interval_ms
                )
                row_count = len(converted_samples)
            else:
                raise Exception()  # TODO: Implement
        else:
            converted_samples = self._convert_samples(data_rows)
        feature_samples = self._create_feature_samples(sample_count)

        sample_time_ms = start_time_ms
        interval_rows = []
        row_index = 0
        last_row_index = row_count - 1
        compact_samples = self._compact_samples
        for sample_index in range(sample_count):
            while True:
                row = converted_samples[row_index]
                row_time_ms = row[BinanceKlineField.OPEN_TIME] + query_interval_ms
                if row_time_ms > sample_time_ms:
                    break
                interval_rows.append(row)
                if row_index == last_row_index:
                    break
                row_index += 1

            if len(interval_rows) > 1:
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
