# developer: taoshi-mbrown
# Copyright © 2024 Taoshi, LLC
import copy

from features import FeatureID
from numpy import ndarray
from typing_extensions import Protocol


class IndividualScaler(Protocol):
    copy: bool

    def fit_transform(self, x) -> any:
        pass

    def inverse_transform(self, x) -> any:
        pass


class GroupScaler(Protocol):
    copy: bool

    def fit(self, x):
        pass

    def partial_fit(self, x):
        pass

    def transform(self, x) -> any:
        pass

    def inverse_transform(self, x) -> any:
        pass


# FeatureScaler is not thread safe. Create a new instance or deep copy for each thread.
class FeatureScaler:
    def __init__(
        self,
        default_scaler: IndividualScaler,
        exclude_feature_ids: list[FeatureID] = None,
        scaling_map: dict[FeatureID, IndividualScaler] = None,
        group_scaling_map: dict[tuple[FeatureID, ...], GroupScaler] = None,
    ):
        _SCALER_COPY_ERROR = "Scaling requires copy property of all scalers to be True."

        if not default_scaler.copy:
            raise ValueError(_SCALER_COPY_ERROR)

        if exclude_feature_ids is None:
            exclude_feature_ids = []

        if scaling_map is None:
            mapped_feature_ids = []
        else:
            scalers = scaling_map.values()
            if any(not scaler.copy for scaler in scalers):
                raise ValueError(_SCALER_COPY_ERROR)

            mapped_feature_ids = list(scaling_map.keys())
            for feature_id in mapped_feature_ids:
                if feature_id in exclude_feature_ids:
                    raise ValueError(
                        f"Feature {feature_id} in scaling_map is also in "
                        "exclude_feature_ids."
                    )

        all_grouped_feature_ids = []
        deep_copied_group_scaling_map = {}
        assigned_scalers = {}

        if group_scaling_map is not None:
            for group_feature_ids, scaler in group_scaling_map.items():
                if not scaler.copy:
                    raise ValueError(_SCALER_COPY_ERROR)

                scaler = copy.deepcopy(scaler)
                deep_copied_group_scaling_map[group_feature_ids] = scaler

                for feature_id in group_feature_ids:
                    if feature_id in exclude_feature_ids:
                        raise ValueError(
                            f"Feature {feature_id} in group_scaling_map is also in "
                            "exclude_feature_ids."
                        )
                    if feature_id in mapped_feature_ids:
                        raise ValueError(
                            f"Feature {feature_id} in group_scaling_map has more "
                            "than one mapping."
                        )
                    assigned_scalers[feature_id] = scaler

                mapped_feature_ids.extend(group_feature_ids)
                all_grouped_feature_ids.extend(group_feature_ids)

        self._default_scaler = default_scaler
        # Exclude grouped scaling from default and individual scaling
        self._exclude_feature_ids = exclude_feature_ids + all_grouped_feature_ids
        self._scaling_map = scaling_map
        self._group_scaling_map = deep_copied_group_scaling_map
        self._assigned_scalers = assigned_scalers

    def scale_feature_samples(
        self, feature_samples: dict[FeatureID, ndarray]
    ) -> dict[FeatureID, ndarray]:
        results = {}

        for feature_id, samples in feature_samples.items():
            if feature_id in self._exclude_feature_ids:
                results[feature_id] = samples
            else:
                scaler = self._assigned_scalers.get(feature_id)
                if scaler is None:
                    if self._scaling_map is None:
                        scaler = self._default_scaler
                    else:
                        scaler = self._scaling_map.get(feature_id, self._default_scaler)
                    scaler = copy.deepcopy(scaler)
                    self._assigned_scalers[feature_id] = scaler
                reshaped_samples = samples.reshape(-1, 1)
                results[feature_id] = scaler.fit_transform(reshaped_samples).flatten()

        if self._group_scaling_map is not None:
            for group_feature_ids, scaler in self._group_scaling_map.items():
                fit = scaler.fit
                for feature_id in group_feature_ids:
                    samples = feature_samples[feature_id]
                    reshaped_samples = samples.reshape(-1, 1)
                    fit(reshaped_samples)
                    # Subsequent fitting must not clear the existing fitting
                    fit = scaler.partial_fit
                for feature_id in group_feature_ids:
                    samples = feature_samples[feature_id]
                    reshaped_samples = samples.reshape(-1, 1)
                    results[feature_id] = scaler.transform(reshaped_samples).flatten()

        return results

    def unscale_feature_samples(
        self, feature_samples: dict[FeatureID, ndarray], ignore_unknown: bool = False
    ) -> dict[FeatureID, ndarray]:
        results = {}

        for feature_id, samples in feature_samples.items():
            scaler = self._assigned_scalers.get(feature_id)
            if scaler is None:
                if ignore_unknown:
                    results[feature_id] = samples
                else:
                    raise RuntimeError(
                        f"Feature {feature_id} has not been scaled, "
                        "so it cannot be unscaled."
                    )
            else:
                reshaped_samples = samples.reshape(-1, 1)
                results[feature_id] = scaler.inverse_transform(
                    reshaped_samples
                ).flatten()

        return results
