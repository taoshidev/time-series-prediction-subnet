# developer: Taoshidev
# Copyright © 2024 Taoshi, LLC
from abc import ABC, abstractmethod
from enum import IntEnum
from features import FeatureID
import numpy as np
from numpy import ndarray


class FeatureCompaction(IntEnum):
    LAST = 0
    FIRST = 1
    SUM = 2
    MIN = 3
    MAX = 4
    MEAN = 5
    MEDIAN = 6
    MODE = 7


class FeatureSource(ABC):
    SOURCE_NAME = None

    VALID_FEATURE_IDS = []

    # Features are defined and checked ahead of time so that the source can be optimized
    # for retrieving and caching the features that it is responsible for providing.
    #
    # Additional parameters can be added in child classes for things like server name,
    # database file, cache size, etc...
    def __init__(
        self,
        feature_ids: list[FeatureID],
        feature_dtypes: list[np.dtype] = None,
        default_dtype: np.dtype = np.dtype(np.float32),
    ):
        if not feature_ids:
            raise ValueError("No feature_ids.")

        for feature_id in feature_ids:
            if feature_id not in self.VALID_FEATURE_IDS:
                raise ValueError(f"Feature {feature_id} not recognized.")

        feature_count = len(feature_ids)

        if len(set(feature_ids)) != feature_count:
            raise ValueError(f"Duplicate feature_ids.")

        self.feature_ids = feature_ids
        self.feature_count = feature_count

        if feature_dtypes is None:
            self.feature_dtypes = [default_dtype] * feature_count
        elif len(feature_dtypes) != feature_count:
            raise ValueError("Length of feature_ids and feature_dtypes do not match.")
        else:
            self.feature_dtypes = [np.dtype(d) for d in feature_dtypes]

    def _create_feature_samples(self, sample_count: int):
        feature_samples = []
        for feature_index in range(self.feature_count):
            dtype = self.feature_dtypes[feature_index]
            feature_samples.append(np.empty(sample_count, dtype))
        return feature_samples

    # Returns: dict with FeatureID keys and 1-dimensional dtype sample array values
    #
    # start_time_ms should always be a multiple of interval_ms
    #
    # Gaps in the samples are automatically filled in with previous values or
    # extrapolations as appropriate for the feature.
    @abstractmethod
    def get_feature_samples(
        self,
        start_time_ms: int,
        interval_ms: int,
        sample_count: int,
    ) -> dict[FeatureID, ndarray]:
        pass

    def feature_samples_to_array(
        self, feature_samples: dict[FeatureID, ndarray], dtype: np.dtype = np.float32
    ) -> ndarray:
        sample_count = None
        for feature_id, samples in feature_samples.items():
            if sample_count is None:
                sample_count = len(samples)
            elif len(samples) != sample_count:
                raise Exception()  # TODO: Implement

        results = np.empty(shape=(self.feature_count, sample_count), dtype=dtype)
        for i, feature_id in enumerate(self.feature_ids):
            samples = feature_samples[feature_id]
            results[i] = samples
        return results.T
