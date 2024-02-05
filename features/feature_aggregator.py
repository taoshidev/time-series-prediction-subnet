# developer: taoshi-mbrown
# Copyright © 2024 Taoshi, LLC
from abc import ABC, abstractmethod
from collections.abc import Iterable
from concurrent.futures import as_completed, ThreadPoolExecutor
from features import FeatureID, FeatureSource
import logging
import numpy as np
from numpy import ndarray
from statistics import fmean
from typing import Callable

SimpleAggregator = Callable[[Iterable], int | float]


class IndividualAggregator(ABC):
    @abstractmethod
    def aggregate(self, feature_samples: Iterable) -> int | float:
        pass


class GroupAggregator(ABC):
    def __init__(self, feature_ids: list[FeatureID]):
        self.feature_ids = feature_ids

    @abstractmethod
    def aggregate(
        self, sources_feature_samples: list[dict[FeatureID, int | float]]
    ) -> dict[FeatureID, int | float]:
        pass


class WeightedMeanGroupAggregator(GroupAggregator):
    def __init__(self, feature_ids: list[FeatureID], weight_feature_id: FeatureID):
        self._value_feature_ids = feature_ids
        self._weight_feature_id = weight_feature_id
        super().__init__([*feature_ids, weight_feature_id])

    def aggregate(
        self, sources_features_sample: list[dict[FeatureID, int | float]]
    ) -> dict[FeatureID, int | float]:
        results = {}

        for value_feature_id in self._value_feature_ids:
            values = []
            weights = []

            for sample in sources_features_sample:
                value = sample[value_feature_id]
                weight = sample[self._weight_feature_id]
                values.append(value)
                weights.append(weight)

            results[value_feature_id] = fmean(values, weights=weights)

        return results


class FeatureAggregator(FeatureSource):
    SOURCE_NAME = "Aggregator"

    # feature_dtypes and default_dtype are not enforced when aggregation uses
    # a single source
    def __init__(
        self,
        sources: list[FeatureSource],
        timeout: float = None,
        feature_ids: list[FeatureID] = None,
        feature_dtypes: list[np.dtype] = None,
        default_dtype: np.dtype = np.dtype(np.float32),
        default_aggregator: SimpleAggregator | IndividualAggregator = fmean,
        aggregation_map: dict[
            FeatureID, SimpleAggregator | IndividualAggregator
        ] = None,
        group_aggregators: list[GroupAggregator] = None,
    ):
        if not sources:
            raise ValueError("No sources.")

        self._sources = sources
        self._timeout = timeout
        self._default_aggregator = default_aggregator
        self._aggregation_map = aggregation_map
        self._logger = logging.getLogger(self.__class__.__name__)

        if feature_ids is None:
            feature_ids = []
            for source in sources:
                feature_ids.extend(source.feature_ids)
            feature_ids = list(set(feature_ids))

        if group_aggregators is not None:
            for group_aggregator in group_aggregators:
                for feature_id in group_aggregator.feature_ids:
                    if feature_id in aggregation_map:
                        raise ValueError(
                            f"Feature {feature_id} in group_aggregators has existing "
                            "mapping."
                        )
        self._group_aggregators = group_aggregators

        self.VALID_FEATURE_IDS = feature_ids
        super().__init__(feature_ids, feature_dtypes, default_dtype)

        self._feature_id_dtype_map = {
            self.feature_ids[i]: self.feature_dtypes[i]
            for i in range(self.feature_count)
        }

    def aggregate_sources_feature_samples(
        self, sample_count: int, sources_feature_samples: list[dict[FeatureID, ndarray]]
    ) -> dict[FeatureID, ndarray]:
        results = {}

        for feature_index, feature_id in enumerate(self.feature_ids):
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
                raise RuntimeError(f"Feature {feature_id} missing from aggregation.")
            elif feature_samples_count == 1:
                aggregated_samples = feature_samples[0]

            else:
                feature_dtype = self.feature_dtypes[feature_index]
                aggregated_samples = np.empty(sample_count, feature_dtype)
                for i in range(sample_count):
                    values = [feature_sample[i] for feature_sample in feature_samples]
                    if isinstance(aggregator, IndividualAggregator):
                        aggregated_samples[i] = aggregator.aggregate(values)
                    else:
                        aggregated_samples[i] = aggregator(values)

            results[feature_id] = aggregated_samples

        if self._group_aggregators is not None:
            for group_aggregator in self._group_aggregators:
                group_feature_ids = group_aggregator.feature_ids

                # Get all the sources that supply all the feature IDs necessary
                group_sources = []
                for source in sources_feature_samples:
                    if all(feature_id in source for feature_id in group_feature_ids):
                        group_sources.append(source)

                aggregated_feature_samples = {}
                for i in range(sample_count):
                    sources_features_sample = []
                    for feature_id in group_feature_ids:
                        features_sample = {}
                        for group_source in group_sources:
                            features_sample[feature_id] = group_source[feature_id][i]
                        sources_features_sample.append(features_sample)

                    sample_aggregation_result = group_aggregator.aggregate(
                        sources_features_sample  # type:ignore
                    )

                    # Extract the results from the aggregation of each sample across the
                    # feature sources and store the aggregated values as feature
                    # samples to be returned
                    for feature_id, value in sample_aggregation_result:
                        feature_samples = aggregated_feature_samples.get(feature_id)
                        if feature_samples is None:
                            feature_dtype = self._feature_id_dtype_map[feature_id]
                            feature_samples = np.empty(sample_count, feature_dtype)
                            aggregated_feature_samples[feature_id] = feature_samples
                        feature_samples[i] = value

                results.update(aggregated_feature_samples)

        return results

    def get_feature_samples(
        self,
        start_time_ms: int,
        interval_ms: int,
        sample_count: int,
    ) -> dict[FeatureID, ndarray]:
        sources_feature_samples = []

        with ThreadPoolExecutor(max_workers=len(self._sources)) as executor:
            future_sources = {}
            futures = []
            timed_out_sources = []
            for source in self._sources:
                future = executor.submit(
                    source.get_feature_samples,
                    start_time_ms,
                    interval_ms,
                    sample_count,
                )
                future_sources[future] = source
                futures.append(future)
                timed_out_sources.append(source.SOURCE_NAME)

            try:
                for future in as_completed(futures, timeout=self._timeout):
                    future_source = future_sources[future]
                    timed_out_sources.remove(future_source.SOURCE_NAME)
                    try:
                        future_result = future.result()
                    except Exception as e:
                        future_result = None
                        self._logger.warning(
                            "Exception occurred requesting feature samples from "
                            f"{future_source.SOURCE_NAME}: {e}"
                        )

                    if future_result is not None:
                        for feature_id, future_result_samples in future_result.items():
                            result_sample_count = len(future_result_samples)
                            if result_sample_count != sample_count:
                                raise RuntimeError(
                                    f"Expected {sample_count} samples from "
                                    f"{future_source.SOURCE_NAME} for feature "
                                    f"{feature_id}, but received {result_sample_count}."
                                )

                        sources_feature_samples.append(future_result)

            except TimeoutError:
                self._logger.warning(
                    "Timeout occurred requesting feature samples "
                    f"from: {timed_out_sources}."
                )

        source_count = len(sources_feature_samples)

        if source_count == 0:
            raise RuntimeError("No sources returned samples to aggregate.")
        elif source_count == 1:
            feature_samples = sources_feature_samples[0]

            for feature_id in self.feature_ids:
                if feature_id not in feature_samples:
                    raise RuntimeError(
                        f"Feature {feature_id} missing from aggregation."
                    )
        else:
            feature_samples = self.aggregate_sources_feature_samples(
                sample_count, sources_feature_samples
            )

        return feature_samples
