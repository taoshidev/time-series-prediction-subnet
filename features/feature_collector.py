# developer: Taoshidev
# Copyright © 2024 Taoshi, LLC
from concurrent.futures import as_completed, ThreadPoolExecutor
from features import FeatureID, FeatureSource
import numpy as np
from numpy import ndarray


class FeatureCollector(FeatureSource):
    SOURCE_NAME = "Collector"

    class FeatureSamplesCache:
        def __init__(
            self,
            feature_samples: dict[FeatureID, ndarray],
            start_time_ms: int,
            interval_ms: int,
            sample_count: int,
        ):
            cache_copy = {}
            for key, value in feature_samples.items():
                cache_copy[key] = value.copy()
            self.samples = cache_copy
            self.start_time_ms = start_time_ms
            self.interval_ms = interval_ms
            self.sample_count = sample_count
            self.end_time_ms = start_time_ms + (interval_ms * sample_count)

    # feature_ids parameter allows overriding the order/exclusion of features
    def __init__(
        self,
        sources: list[FeatureSource],
        timeout: float = None,
        feature_ids: list[FeatureID] = None,
        cache_results=True,
    ):
        if not sources:
            raise ValueError("No sources.")

        self._sources = sources
        self._timeout = timeout

        if feature_ids is None:
            feature_ids = []
            for source in sources:
                feature_overlap = set(feature_ids) & set(source.feature_ids)
                if feature_overlap:
                    raise RuntimeError(
                        f"Overlap of features {feature_overlap} with "
                        f"{source.SOURCE_NAME}."
                    )
                feature_ids.extend(source.feature_ids)

        self.VALID_FEATURE_IDS = feature_ids
        super().__init__(feature_ids)

        self._cache_results = cache_results
        self._cache = None

    def get_feature_samples(
        self,
        start_time_ms: int,
        interval_ms: int,
        sample_count: int,
    ) -> dict[FeatureID, ndarray]:
        uncached_samples = {}
        cache_start_index = 0
        cache_end_index = 0
        cached_sample_count = 0
        save_cache_results = self._cache_results
        uncached_start_time_ms = start_time_ms
        uncached_sample_count = sample_count

        if (self._cache is not None) and (interval_ms == self._cache.interval_ms):
            # For simplicity and speed, the cache will only hit if the last portion of
            # the last collection overlaps with the previous collection
            if self._cache.start_time_ms <= start_time_ms < self._cache.end_time_ms:
                cache_start_index = int(
                    (start_time_ms - self._cache.start_time_ms) / interval_ms
                )
                cached_sample_count = self._cache.sample_count - cache_start_index
                cache_end_index = cache_start_index + cached_sample_count

                if cached_sample_count == sample_count:
                    save_cache_results = False
                else:
                    uncached_start_time_ms = self._cache.end_time_ms
                    uncached_sample_count = sample_count - cached_sample_count

        collect_uncached = cached_sample_count != sample_count

        if collect_uncached:
            with ThreadPoolExecutor(max_workers=len(self._sources)) as executor:
                futures = []
                future_sources = {}
                for source in self._sources:
                    future = executor.submit(
                        source.get_feature_samples,
                        uncached_start_time_ms,
                        interval_ms,
                        uncached_sample_count,
                    )
                    future_sources[future] = source
                    futures.append(future)

                for future in as_completed(futures, timeout=self._timeout):
                    future_result = future.result()
                    for feature_id, future_result_samples in future_result.items():
                        future_result_sample_count = len(future_result_samples)
                        if future_result_sample_count != uncached_sample_count:
                            future_source = future_sources[future]
                            raise RuntimeError(
                                f"Expected {uncached_sample_count} samples from "
                                f"{future_source.SOURCE_NAME} for feature {feature_id}, "
                                f"but {future_result_sample_count} samples returned."
                            )

                    feature_overlap = uncached_samples.keys() & future_result.keys()
                    if feature_overlap:
                        future_source = future_sources[future]
                        raise RuntimeError(
                            f"Collection overlap of features {feature_overlap} with "
                            f"{future_source.SOURCE_NAME}."
                        )

                    uncached_samples.update(future_result)

            for feature_id in self.feature_ids:
                if feature_id not in uncached_samples:
                    raise RuntimeError(f"Feature {feature_id} missing from collection.")

        if cached_sample_count == 0:
            feature_samples = uncached_samples
        else:
            feature_samples = {}
            for feature_id in self.feature_ids:
                feature_samples = np.empty(sample_count)
                cached_feature_samples = self._cache[feature_id]
                feature_samples[:cached_sample_count] = cached_feature_samples[
                    cache_start_index:cache_end_index
                ]
                if collect_uncached:
                    feature_samples[cached_sample_count:] = uncached_samples[feature_id]
                feature_samples[feature_id] = feature_samples

        if save_cache_results:
            self._cache = self.FeatureSamplesCache(
                feature_samples, start_time_ms, interval_ms, sample_count
            )

        return feature_samples
