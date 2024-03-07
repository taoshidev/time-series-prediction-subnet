# developer: taoshi-mbrown
# Copyright © 2024 Taoshi Inc
from enum import Enum
from features import FeatureCompaction, FeatureID, FeatureSource
import math
import numpy as np
from numpy import ndarray
import statistics
from time_util import current_interval_ms, datetime, time_span_ms
from yfinance import Ticker


class YahooFinanceKlineField(str, Enum):
    OPEN_TIME = "Date"

    PRICE_OPEN = "Open"
    PRICE_CLOSE = "Close"
    PRICE_HIGH = "High"
    PRICE_LOW = "Low"
    VOLUME = "Volume"
    DIVIDENDS = "Dividends"
    SPLITS = "Stock Splits"


class YahooFinanceKlineFeatureSource(FeatureSource):
    SOURCE_NAME = "YahooFinanceKline"

    _INTERVALS = {
        time_span_ms(minutes=1): "1m",
        time_span_ms(minutes=2): "2m",
        time_span_ms(minutes=5): "5m",
        time_span_ms(minutes=15): "15m",
        time_span_ms(minutes=30): "30m",
        time_span_ms(hours=1): "60m",
        time_span_ms(hours=3): "90m",
        time_span_ms(days=1): "1d",
        time_span_ms(days=3): "5d",
        time_span_ms(weeks=1): "1wk",
        time_span_ms(days=30): "1mo",
        time_span_ms(days=90): "3mo",
    }

    # Default is using last sample, so on only include other types
    _METRIC_COMPACTIONS = {
        YahooFinanceKlineField.PRICE_OPEN: FeatureCompaction.FIRST,
        YahooFinanceKlineField.PRICE_HIGH: FeatureCompaction.MAX,
        YahooFinanceKlineField.PRICE_LOW: FeatureCompaction.MIN,
        YahooFinanceKlineField.VOLUME: FeatureCompaction.SUM,
        YahooFinanceKlineField.DIVIDENDS: FeatureCompaction.SUM,
        YahooFinanceKlineField.SPLITS: FeatureCompaction.SUM,
    }

    def __init__(
        self,
        ticker: str,
        source_interval_ms: int,
        feature_mappings: dict[FeatureID, YahooFinanceKlineField],
        feature_dtypes: list[np.dtype] = None,
        default_dtype: np.dtype = np.dtype(np.float32),
    ):
        query_interval = self._INTERVALS.get(source_interval_ms)
        if query_interval is None:
            raise ValueError(f"interval_ms {source_interval_ms} is not supported.")

        feature_ids = list(feature_mappings.keys())
        self.VALID_FEATURE_IDS = feature_ids
        super().__init__(feature_ids, feature_dtypes, default_dtype)

        self._source_interval_ms = source_interval_ms
        self._query_interval = query_interval
        self._metrics = list(feature_mappings.values())
        self._convert_metrics = [YahooFinanceKlineField.OPEN_TIME, *self._metrics]
        self._client = Ticker(ticker)

    # noinspection PyMethodMayBeStatic
    def _convert_samples(self, data_rows: list[dict]) -> None:
        _OPEN_TIME = YahooFinanceKlineField.OPEN_TIME
        _SEC_TO_MS = time_span_ms(seconds=1)
        for row in data_rows:
            open_time = row[_OPEN_TIME]
            row[_OPEN_TIME] = int(open_time.timestamp() * _SEC_TO_MS)

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
        _OPEN_TIME = YahooFinanceKlineField.OPEN_TIME

        # Yahoo Finance uses open time for queries
        query_start_time_ms = start_time_ms - interval_ms

        # Align on interval so queries for 1 sample include at least 1 sample
        query_start_time_ms = current_interval_ms(
            query_start_time_ms, self._source_interval_ms
        )

        query_end_time_ms = start_time_ms + (interval_ms * (sample_count - 2))

        start_time = datetime.fromtimestamp_ms(query_start_time_ms)
        end_time = datetime.fromtimestamp_ms(query_end_time_ms)

        data_frame = self._client.history(
            interval=self._query_interval,
            start=start_time,
            end=end_time,
            timeout=None,
            raise_errors=True,
        )
        # Ensure the index is accessible as a time field
        data_frame[_OPEN_TIME.value] = data_frame.index
        data_rows = data_frame.to_dict(orient="records")

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
