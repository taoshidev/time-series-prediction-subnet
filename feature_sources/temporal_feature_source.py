# developer: Taoshidev
# Copyright © 2024 Taoshi, LLC
from datetime import datetime, timezone
from features import FeatureID, FeatureSource
import numpy as np
from numpy import ndarray
from time_util.time_util import TimeUtil


class TemporalFeatureSource(FeatureSource):
    SOURCE_NAME = "Temporal"

    VALID_FEATURE_IDS = [
        FeatureID.TIME_OF_DAY,
        FeatureID.TIME_OF_WEEK,
        FeatureID.TIME_OF_MONTH,
        FeatureID.TIME_OF_YEAR,
    ]

    _DAY_MS = 24 * 60 * 60 * 1000
    _WEEK_MS = 7 * _DAY_MS

    # Unix epoch starts on a Thursday
    _OFFSET_THURSDAY_TO_SUNDAY_MS = 3 * _DAY_MS

    @staticmethod
    def _get_month_term(time_ms) -> tuple[int, int]:
        term_datetime = TimeUtil.millis_to_timestamp(time_ms)
        year = term_datetime.year
        month = term_datetime.month
        if month == 12:
            end_year = year + 1
            end_month = 1
        else:
            end_year = year
            end_month = month + 1
        start = datetime(year=year, month=month, day=1, tzinfo=timezone.utc)
        end = datetime(year=end_year, month=end_month, day=1, tzinfo=timezone.utc)
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        return start_ms, end_ms

    @staticmethod
    def _get_year_term(time_ms) -> tuple[int, int]:
        term_datetime = TimeUtil.millis_to_timestamp(time_ms)
        year = term_datetime.year
        start = datetime(year=year, month=1, day=1, tzinfo=timezone.utc)
        end = datetime(year=year + 1, month=1, day=1, tzinfo=timezone.utc)
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        return start_ms, end_ms

    def get_feature_samples(
        self,
        start_time_ms: int,
        interval_ms: int,
        sample_count: int,
    ) -> dict[FeatureID, ndarray]:
        offset = 0
        divisor = 0
        results = {}
        for feature_index, feature_id in enumerate(self.feature_ids):
            term_function = None
            match feature_id:
                case FeatureID.TIME_OF_DAY:
                    offset = 0
                    divisor = self._DAY_MS
                case FeatureID.TIME_OF_WEEK:
                    offset = self._OFFSET_THURSDAY_TO_SUNDAY_MS
                    divisor = self._WEEK_MS
                case FeatureID.TIME_OF_MONTH:
                    term_function = self._get_month_term
                case FeatureID.TIME_OF_YEAR:
                    term_function = self._get_year_term

            dtype = self.feature_dtypes[feature_index]
            samples = np.empty(shape=sample_count, dtype=dtype)

            if term_function is None:
                # -1 to 1 normalization
                divisor /= 2
                current_time_ms = start_time_ms + offset
                for i in range(sample_count):
                    samples[i] = ((current_time_ms / divisor) % 2) - 1
                    current_time_ms += interval_ms
            else:
                term_start_ms = 0
                term_end_ms = 0
                current_time_ms = start_time_ms
                for i in range(sample_count):
                    if current_time_ms >= term_end_ms:
                        term_start_ms, term_end_ms = term_function(current_time_ms)
                        divisor = (term_end_ms - term_start_ms) / 2
                    samples[i] = (((current_time_ms - term_start_ms) / divisor) % 2) - 1
                    current_time_ms += interval_ms

            results[feature_id] = samples

        return results
