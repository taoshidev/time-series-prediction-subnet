# developer: taoshi-mbrown
# Copyright © 2024 Taoshi Inc
from features import FeatureID, FeatureSource
import numpy as np
from numpy import ndarray
from time_util import datetime, time_span_ms


class TemporalFeatureSource(FeatureSource):
    SOURCE_NAME = "Temporal"

    VALID_FEATURE_IDS = [
        FeatureID.EPOCH_TIMESTAMP_MS,
        FeatureID.TIME_OF_DAY,
        FeatureID.TIME_OF_WEEK,
        FeatureID.TIME_OF_MONTH,
        FeatureID.TIME_OF_YEAR,
    ]

    @staticmethod
    def _get_month_term(time_ms) -> tuple[int, int]:
        term_datetime = datetime.fromtimestamp_ms(time_ms)
        year = term_datetime.year
        month = term_datetime.month
        if month == 12:
            end_year = year + 1
            end_month = 1
        else:
            end_year = year
            end_month = month + 1
        start_ms = datetime(year=year, month=month, day=1).timestamp_ms()
        end_ms = datetime(year=end_year, month=end_month, day=1).timestamp_ms()
        return start_ms, end_ms

    @staticmethod
    def _get_year_term(time_ms) -> tuple[int, int]:
        term_datetime = datetime.fromtimestamp_ms(time_ms)
        year = term_datetime.year
        start_ms = datetime(year=year, month=1, day=1).timestamp_ms()
        end_ms = datetime(year=year + 1, month=1, day=1).timestamp_ms()
        return start_ms, end_ms

    def get_feature_samples(
        self,
        start_time_ms: int,
        interval_ms: int,
        sample_count: int,
    ) -> dict[FeatureID, ndarray]:
        results = {}
        for feature_index, feature_id in enumerate(self.feature_ids):
            dtype = self.feature_dtypes[feature_index]
            samples = np.empty(shape=sample_count, dtype=dtype)

            if feature_id == FeatureID.EPOCH_TIMESTAMP_MS:
                current_time_ms = start_time_ms
                for i in range(sample_count):
                    samples[i] = current_time_ms
                    current_time_ms += interval_ms
            else:
                offset = 0
                divisor = 0
                term_function = None
                match feature_id:
                    case FeatureID.TIME_OF_DAY:
                        offset = 0
                        divisor = time_span_ms(days=1)
                    case FeatureID.TIME_OF_WEEK:
                        # Unix epoch starts on a Thursday
                        offset = time_span_ms(days=4)
                        divisor = time_span_ms(weeks=1)
                    case FeatureID.TIME_OF_MONTH:
                        term_function = self._get_month_term
                    case FeatureID.TIME_OF_YEAR:
                        term_function = self._get_year_term

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
                        samples[i] = (
                            ((current_time_ms - term_start_ms) / divisor) % 2
                        ) - 1
                        current_time_ms += interval_ms

            results[feature_id] = samples

        self._check_feature_samples(results, start_time_ms, interval_ms)

        return results
