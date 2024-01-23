# developer: Taoshidev
# Copyright © 2024 Taoshi, LLC
from abc import abstractmethod
from numpy import ndarray
from features import FeatureID, FeatureSource


class FeatureStorage(FeatureSource):
    @abstractmethod
    def set_feature_samples(
        self,
        start_time_ms: int,
        interval_ms: int,
        samples: dict[FeatureID, ndarray],
    ):
        pass

    @abstractmethod
    def get_start_time_ms(self) -> int:
        pass

    @abstractmethod
    def get_sample_count(self) -> int:
        pass
