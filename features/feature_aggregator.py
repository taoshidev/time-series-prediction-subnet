# developer: Taoshidev
# Copyright © 2024 Taoshi, LLC
from concurrent.futures import as_completed, ThreadPoolExecutor
from features import FeatureID, FeatureSource
import numpy as np
from numpy import ndarray
from typing_extensions import Protocol
from typing import Callable
from statistics import mean


IndividualAggregator = Callable[[list], int | float]


class GroupAggregator(Protocol):
    def aggregate(self, values: dict[FeatureID, int | float]) -> int | float:
        pass


class FeatureAggregator(FeatureSource):
    SOURCE_NAME = "Aggregator"

    # feature_dtypes and default_dtype are not enforced when features have one source
    def __init__(
        self,
        sources: list[FeatureSource],
        timeout: float = None,
        feature_ids: list[FeatureID] = None,
        feature_dtypes: list[np.dtype] = None,
        default_dtype: np.dtype = np.dtype(np.float32),
        default_aggregator: IndividualAggregator = mean,
        aggregation_map: dict[FeatureID, IndividualAggregator] = None,
    ):
        if not sources:
            raise ValueError("No sources.")

        self._sources = sources
        self._timeout = timeout
        self._default_aggregator = default_aggregator
        self._aggregation_map = aggregation_map

        # TODO: Verify feature_ids are the same between sources

        if feature_ids is None:
            feature_ids = []
            for source in sources:
                feature_ids.extend(source.feature_ids)
            feature_ids = list(set(feature_ids))

        self.VALID_FEATURE_IDS = feature_ids
        super().__init__(feature_ids, feature_dtypes, default_dtype)

    def aggregate(
        self, sample_count: int, sources_feature_samples: list[dict[FeatureID, ndarray]]
    ) -> dict[FeatureID, ndarray]:
        results = {}

        for feature_index, feature_id in self.feature_ids:
            aggregator = self._default_aggregator
            if self._aggregation_map is not None:
                aggregator = self._aggregation_map.get(feature_id, aggregator)

            feature_samples = []
            for source_feature_samples in sources_feature_samples:
                samples = source_feature_samples.get(feature_id)
                if samples is not None:
                    feature_samples.append(samples)

            feature_samples_count = len(feature_samples)
            if feature_samples_count == 0:
                raise Exception()  # TODO: Implement

            elif feature_samples_count == 1:
                aggregated_samples = feature_samples[0]

            else:
                feature_dtype = self.feature_dtypes[feature_index]
                aggregated_samples = np.empty(sample_count, feature_dtype)
                for i in range(sample_count):
                    values = [feature_sample[i] for feature_sample in feature_samples]
                    aggregated_samples[i] = aggregator(values)

            results[feature_id] = aggregated_samples

        return results

    def get_feature_samples(
        self,
        start_time_ms: int,
        interval_ms: int,
        sample_count: int,
    ) -> dict[FeatureID, ndarray]:
        sources_feature_samples = []

        with ThreadPoolExecutor(max_workers=len(self._sources)) as executor:
            futures = []
            future_sources = {}
            for source in self._sources:
                future = executor.submit(
                    source.get_feature_samples,
                    start_time_ms,
                    interval_ms,
                    sample_count,
                )
                future_sources[future] = source
                futures.append(future)

            try:
                for future in as_completed(futures, timeout=self._timeout):
                    try:
                        future_result = future.result()
                    except:
                        future_result = None
                        # TODO: Logging

                    if future_result is not None:
                        for feature_id, future_result_samples in future_result.values():
                            future_result_sample_count = len(future_result_samples)
                            if future_result_sample_count != sample_count:
                                future_source = future_sources[future]
                                raise Exception(
                                    f"Expected {sample_count} samples from "
                                    f"{future_source.SOURCE_NAME} for feature {feature_id}, "
                                    f"but {future_result_sample_count} samples returned."
                                )

                        sources_feature_samples.append(future_result)

            except TimeoutError:
                pass  # TODO: Logging

        source_count = len(sources_feature_samples)

        if source_count == 0:
            raise Exception()  # TODO: Implement
        elif source_count == 1:
            feature_samples = sources_feature_samples[0]
        else:
            feature_samples = self.aggregate(sample_count, sources_feature_samples)

        return feature_samples
