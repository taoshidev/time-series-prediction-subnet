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


class KrakenKlineField(IntEnum):
    OPEN_TIME = 0
    PRICE_OPEN = 1
    PRICE_HIGH = 2
    PRICE_LOW = 3
    PRICE_CLOSE = 4
    VWAP = 5
    VOLUME = 6
    TRADES_COUNT = 7


# Kraken is only useful for recent historical samples
# (within the last 720 samples of the interval specified)
class KrakenKlineFeatureSource(FeatureSource):
    SOURCE_NAME = "KrakenKline"

    DEFAULT_RETRIES = 3
    RETRY_DELAY = 1.0

    _QUERY_LIMIT = 720

    # Must be in ascending order
    _INTERVALS = [
        time_span_ms(minutes=1),
        time_span_ms(minutes=5),
        time_span_ms(minutes=15),
        time_span_ms(minutes=30),
        time_span_ms(hours=1),
        time_span_ms(hours=4),
        time_span_ms(days=1),
        time_span_ms(weeks=1),
        time_span_ms(days=15),
    ]

    # Default is using last sample, so on only include other types
    _FIELD_COMPACTIONS = {
        KrakenKlineField.PRICE_OPEN: FeatureCompaction.FIRST,
        KrakenKlineField.PRICE_HIGH: FeatureCompaction.MAX,
        KrakenKlineField.PRICE_LOW: FeatureCompaction.MIN,
        KrakenKlineField.VWAP: FeatureCompaction.MEAN,
        KrakenKlineField.VOLUME: FeatureCompaction.SUM,
        KrakenKlineField.TRADES_COUNT: FeatureCompaction.SUM,
    }

    _UNCONVERTED_FIELDS = {
        KrakenKlineField.PRICE_OPEN,
        KrakenKlineField.PRICE_HIGH,
        KrakenKlineField.PRICE_LOW,
        KrakenKlineField.PRICE_CLOSE,
        KrakenKlineField.VWAP,
        KrakenKlineField.VOLUME,
    }

    def __init__(
        self,
        symbol: str,
        interval_ms: int,
        feature_mappings: dict[FeatureID, KrakenKlineField],
        feature_dtypes: list[np.dtype] = None,
        default_dtype: np.dtype = np.dtype(np.float32),
        retries: int = DEFAULT_RETRIES,
    ):
        if interval_ms not in self._INTERVALS:
            raise ValueError(f"interval_ms {interval_ms} is not supported.")
        query_interval = int(interval_ms / time_span_ms(minutes=1))

        feature_ids = list(feature_mappings.keys())
        self.VALID_FEATURE_IDS = feature_ids
        super().__init__(feature_ids, feature_dtypes, default_dtype)

        self._symbol = symbol
        self._interval_ms = interval_ms
        self._query_interval = query_interval
        self._feature_mappings = feature_mappings
        self._fields = list(feature_mappings.values())
        self._retries = retries
        self._logger = getLogger(self.__class__.__name__)

        convert_fields = set(self._fields) & self._UNCONVERTED_FIELDS
        self._convert_field_indexes = [field for field in convert_fields]

    # noinspection PyMethodMayBeStatic
    def _convert_sample(self, sample: list) -> None:
        for i in self._convert_field_indexes:
            sample[i] = float(sample[i])

    def _convert_samples(self, data_rows: list[list]) -> None:
        _OPEN_TIME = KrakenKlineField.OPEN_TIME
        _SEC_TO_MS = time_span_ms(seconds=1)
        for row in data_rows:
            row[_OPEN_TIME] *= _SEC_TO_MS
            self._convert_sample(row)

    def _compact_samples(self, samples: list[list]) -> list:
        result = samples[-1].copy()
        for field in KrakenKlineField:
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
        _OPEN_TIME = KrakenKlineField.OPEN_TIME

        if sample_count > self._QUERY_LIMIT:
            raise ValueError(
                f"sample_count {interval_ms} is greater than the "
                f"maximum of {self._QUERY_LIMIT}."
            )

        # Kraken uses open time for queries
        open_start_time_ms = start_time_ms - interval_ms
        if interval_ms < self._interval_ms:
            open_start_time_ms -= self._interval_ms

        open_end_time_ms = open_start_time_ms + (interval_ms * (sample_count - 2))

        query_since = int(open_start_time_ms / time_span_ms(seconds=1))

        url = (
            "https://api.kraken.com/0/public/OHLC"
            f"?pair={self._symbol}&interval={self._query_interval}&since={query_since}"
        )

        data_rows = []
        retries = self._retries
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
                    response_error = response_data.get("error")
                    response_result = response_data.get("result")
                    if response_result is not None:
                        data_rows = response_result.get(self._symbol)
                        success = True
                    else:
                        self._logger.error(
                            f"Kraken error: {response_error}",
                        )
            except Exception as e:
                self._logger.warning(
                    f"Exception occurred requesting feature samples using {url}: {e}"
                )

            if success or (retries == 0):
                break

            retries -= 1
            time.sleep(self.RETRY_DELAY)

        row_count = len(data_rows)
        if row_count == 0:
            raise RuntimeError("No samples received.")

        first_row = data_rows[0]
        first_open_time_ms = first_row[_OPEN_TIME] * time_span_ms(seconds=1)
        if first_open_time_ms > open_end_time_ms:
            raise RuntimeError("Requested timeframe is too far in the past.")

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
