# developer: Taoshidev
# Copyright © 2024 Taoshi, LLC
from abc import ABC, abstractmethod
from collections.abc import Iterable
from concurrent.futures import as_completed, ThreadPoolExecutor
from features import FeatureID, FeatureSource
import logging
import numpy as np
from numpy import ndarray
from statistics import mean
from typing import Callable

SimpleAggregator = Callable[[Iterable], int | float]


class IndividualAggregator(ABC):
    @abstractmethod
    def aggregate(self, feature_samples: Iterable) -> int | float:
        pass


class GroupAggregator(ABC):
    @abstractmethod
    def aggregate(
        self, sources_feature_samples: list[dict[FeatureID, int | float]]
    ) -> dict[FeatureID, int | float]:
        pass


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
        default_aggregator: SimpleAggregator | IndividualAggregator = mean,
        aggregation_map: dict[
            FeatureID, SimpleAggregator | IndividualAggregator
        ] = None,
        group_aggregation_map: list[tuple[list[FeatureID], GroupAggregator]] = None,
    ):
        if not sources:
            raise ValueError("No sources.")

        self._sources = sources
        self._timeout = timeout
        self._default_aggregator = default_aggregator
        self._aggregation_map = aggregation_map
        self._logger = logging.getLogger(self.__class__.__name__)

        # TODO: Implement group aggregation

        if feature_ids is None:
            feature_ids = []
            for source in sources:
                feature_ids.extend(source.feature_ids)
            feature_ids = list(set(feature_ids))

        self.VALID_FEATURE_IDS = feature_ids
        super().__init__(feature_ids, feature_dtypes, default_dtype)

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
