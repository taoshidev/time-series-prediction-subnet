# developer: taoshi-mbrown
# Copyright © 2024 Taoshi Inc
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
from time_util import current_interval_ms, time_span_ms
from urllib.parse import urlencode


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

    _URL = "https://api.binance.com/api/v3/klines"

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
        source_interval_ms: int,
        feature_mappings: dict[FeatureID, BinanceKlineField],
        feature_dtypes: list[np.dtype] = None,
        default_dtype: np.dtype = np.dtype(np.float32),
        retries: int = DEFAULT_RETRIES,
    ):
        query_interval = self._INTERVALS.get(source_interval_ms)
        if query_interval is None:
            raise ValueError(f"interval_ms {source_interval_ms} is not supported.")

        feature_ids = list(feature_mappings.keys())
        self.VALID_FEATURE_IDS = feature_ids
        super().__init__(feature_ids, feature_dtypes, default_dtype)

        query_parameters = {
            "symbol": symbol,
            "interval": query_interval,
            "limit": self._QUERY_LIMIT,
        }

        self._source_interval_ms = source_interval_ms
        self._feature_mappings = feature_mappings
        self._fields = list(feature_mappings.values())
        self._query_parameters = query_parameters
        self._retries = retries
        self._logger = getLogger(self.__class__.__name__)

        self._convert_field_indexes = set(self._fields) & self._UNCONVERTED_FIELDS

    def _convert_sample(self, sample: list) -> None:
        for i in self._convert_field_indexes:
            sample[i] = float(sample[i])

    def _convert_samples(self, data_rows: list[list]) -> None:
        for row in data_rows:
            self._convert_sample(row)

    # TODO: Examine moving this into FeatureSource base class
    def _compact_samples(self, samples: list[list]) -> list:
        result = samples[-1].copy()
        for field in self._fields:
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
        _OPEN_TIME = BinanceKlineField.OPEN_TIME

        # Binance uses open time for queries
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
            page_sample_count = (
                end_time_ms - query_start_time_ms
            ) / self._source_interval_ms
            page_sample_count = int(min(page_sample_count, self._QUERY_LIMIT))

            query_end_time_ms = query_start_time_ms + (
                self._source_interval_ms * (page_sample_count - 1)
            )

            query_parameters["startTime"] = str(query_start_time_ms)
            query_parameters["endTime"] = str(query_end_time_ms)
            url = self._URL + "?" + urlencode(query_parameters)

            success = False
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

        # TODO: Examine moving the rest of this function into FeatureSource base class
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

            for feature_index, field in enumerate(self._fields):
                feature_samples[feature_index][sample_index] = row[field]

            interval_rows.clear()
            sample_time_ms += interval_ms

        results = {
            self.feature_ids[feature_index]: feature_samples[feature_index]
            for feature_index in range(self.feature_count)
        }

        self._check_feature_samples(results, start_time_ms, interval_ms)

        return results
