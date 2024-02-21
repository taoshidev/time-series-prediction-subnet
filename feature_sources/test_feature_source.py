# developer: taoshi-mbrown
# Copyright © 2024 Taoshi Inc
from features import FeatureID, FeatureSource
import numpy as np
from numpy import ndarray
from time_util import time_span_ms


class TestFeatureSource(FeatureSource):
    SOURCE_NAME = "Test"

    def __init__(
        self,
        feature_ids: list[FeatureID],
        feature_dtypes: list[np.dtype] = None,
        default_dtype: np.dtype = np.dtype(np.float32),
    ):
        self.VALID_FEATURE_IDS = feature_ids
        super().__init__(feature_ids, feature_dtypes, default_dtype)

    def get_feature_samples(
        self,
        start_time_ms: int,
        interval_ms: int,
        sample_count: int,
    ) -> dict[FeatureID, ndarray]:
        divisor = time_span_ms(minutes=15)
        results = {}
        for feature_index, feature_id in enumerate(self.feature_ids):
            dtype = self.feature_dtypes[feature_index]
            samples = np.empty(shape=sample_count, dtype=dtype)
            current_time_ms = start_time_ms + feature_index
            for i in range(sample_count):
                samples[i] = current_time_ms % divisor
                current_time_ms += interval_ms
            results[feature_id] = samples

        self._check_feature_samples(results, start_time_ms, interval_ms)

        return results
