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


class CoinbaseKlineField(IntEnum):
    OPEN_TIME = 0
    PRICE_LOW = 1
    PRICE_HIGH = 2
    PRICE_OPEN = 3
    PRICE_CLOSE = 4
    VOLUME = 5


class CoinbaseKlineFeatureSource(FeatureSource):
    SOURCE_NAME = "CoinbaseKline"

    DEFAULT_RETRIES = 3
    RETRY_DELAY = 1.0

    _QUERY_LIMIT = 300

    # Must be in ascending order
    _INTERVALS = [
        time_span_ms(minutes=1),
        time_span_ms(minutes=5),
        time_span_ms(minutes=15),
        time_span_ms(hours=1),
        time_span_ms(hours=6),
        time_span_ms(days=1),
    ]

    # Default is using last sample, so on only include other types
    _FIELD_COMPACTIONS = {
        CoinbaseKlineField.PRICE_OPEN: FeatureCompaction.FIRST,
        CoinbaseKlineField.PRICE_HIGH: FeatureCompaction.MAX,
        CoinbaseKlineField.PRICE_LOW: FeatureCompaction.MIN,
        CoinbaseKlineField.VOLUME: FeatureCompaction.SUM,
    }

    def __init__(
        self,
        symbol: str,
        source_interval_ms: int,
        feature_mappings: dict[FeatureID, CoinbaseKlineField],
        feature_dtypes: list[np.dtype] = None,
        default_dtype: np.dtype = np.dtype(np.float32),
        retries: int = DEFAULT_RETRIES,
    ):
        if source_interval_ms not in self._INTERVALS:
            raise ValueError(f"interval_ms {source_interval_ms} is not supported.")
        query_interval = int(source_interval_ms / time_span_ms(seconds=1))

        feature_ids = list(feature_mappings.keys())
        self.VALID_FEATURE_IDS = feature_ids
        super().__init__(feature_ids, feature_dtypes, default_dtype)

        self._symbol = symbol
        self._source_interval_ms = source_interval_ms
        self._query_interval = query_interval
        self._feature_mappings = feature_mappings
        self._fields = list(feature_mappings.values())
        self._retries = retries
        self._logger = getLogger(self.__class__.__name__)

    # noinspection PyMethodMayBeStatic
    def _convert_samples(self, data_rows: list[list]) -> None:
        _OPEN_TIME = CoinbaseKlineField.OPEN_TIME
        _SEC_TO_MS = time_span_ms(seconds=1)
        for row in data_rows:
            row[_OPEN_TIME] *= _SEC_TO_MS

    def _compact_samples(self, samples: list[list]) -> list:
        result = samples[-1].copy()
        for field in CoinbaseKlineField:
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
        _OPEN_TIME = CoinbaseKlineField.OPEN_TIME

        # Coinbase uses open time for queries
        open_start_time_ms = start_time_ms - interval_ms

        # Align on interval so queries for 1 sample include at least 1 sample
        open_start_time_ms = current_interval_ms(
            open_start_time_ms, self._source_interval_ms
        )

        data_rows = []
        retries = self._retries
        samples_left = sample_count
        # Loop for pagination
        while True:
            page_sample_count = min(samples_left, self._QUERY_LIMIT)
            open_end_time_ms = open_start_time_ms + (
                interval_ms * (page_sample_count - 1)
            )

            query_start_time = int(open_start_time_ms / time_span_ms(seconds=1))
            query_end_time = int(open_end_time_ms / time_span_ms(seconds=1))

            url = (
                f"https://api.exchange.coinbase.com/products/{self._symbol}/candles"
                f"?granularity={self._query_interval}&start={query_start_time}"
                f"&end={query_end_time}"
            )

            success = False
            # Loop for retries
            while True:
                try:
                    response = requests.get(url)

                    if response.status_code >= HTTPStatus.BAD_REQUEST:
                        try:
                            error_response = response.json()
                            error_message = error_response.get("message")
                            coinbase_error = f", Coinbase error: {error_message}"
                        except JSONDecodeError:
                            coinbase_error = ""
                        self._logger.error(
                            f"HTTP error {response.status_code}: {response.reason}"
                            f"{coinbase_error}",
                        )
                    else:
                        response_rows = response.json()
                        response_rows.reverse()
                        data_rows.extend(response_rows)
                        success = True

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

            open_start_time_ms = (
                data_rows[-1][_OPEN_TIME] * time_span_ms(seconds=1)
            ) + self._source_interval_ms

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

        return results
