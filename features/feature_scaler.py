# developer: Taoshidev
# Copyright © 2024 Taoshi, LLC
from features import FeatureID
from numpy import ndarray
from typing_extensions import Protocol


class IndividualScaler(Protocol):
    copy: bool

    def fit_transform(self, x):
        pass


class GroupScaler(Protocol):
    copy: bool

    def fit(self, x):
        pass

    def partial_fit(self, x):
        pass

    def transform(self, x):
        pass


class FeatureScaler:
    def __init__(
        self,
        default_scaler: IndividualScaler,
        exclude_feature_ids: list[FeatureID] = None,
        scaling_map: dict[FeatureID, IndividualScaler] = None,
        group_scaling_map: list[tuple[list[FeatureID], GroupScaler]] = None,
    ):
        _SCALER_COPY_ERROR = (
            "Inplace scaling requires copy property of all scalers to be False."
        )

        if not default_scaler.copy:
            raise ValueError(_SCALER_COPY_ERROR)

        if exclude_feature_ids is None:
            exclude_feature_ids = []

        if scaling_map is None:
            mapped_feature_ids = []
        else:
            scalers = scaling_map.values()
            if any(scaler.copy for scaler in scalers):
                raise ValueError(_SCALER_COPY_ERROR)

            mapped_feature_ids = list(scaling_map.keys())
            for feature_id in mapped_feature_ids:
                if feature_id in exclude_feature_ids:
                    raise ValueError(
                        f"Feature {feature_id} in scaling_map is also in exclude_feature_ids."
                    )

        all_grouped_feature_ids = []
        for group_feature_ids, scaler in group_scaling_map:
            if scaler.copy:
                raise ValueError(_SCALER_COPY_ERROR)

            for feature_id in group_feature_ids:
                if feature_id in exclude_feature_ids:
                    raise Exception(
                        f"Feature {feature_id} in group_scaling_map is also in exclude_feature_ids."
                    )
                if feature_id in mapped_feature_ids:
                    raise Exception(
                        f"Feature {feature_id} in group_scaling_map has more than one mapping."
                    )
            mapped_feature_ids.extend(group_feature_ids)
            all_grouped_feature_ids.extend(group_feature_ids)

        self._default_scalar = default_scaler
        # Exclude grouped scaling from default and individual scaling
        self._exclude_feature_ids = exclude_feature_ids + all_grouped_feature_ids
        self._scaling_map = scaling_map
        self._group_scaling_map = group_scaling_map

    def scale_samples(self, feature_samples: dict[FeatureID, ndarray]) -> None:
        for feature_id, samples in feature_samples.items():
            if feature_id not in self._exclude_feature_ids:
                if self._scaling_map is None:
                    scaler = self._default_scalar
                else:
                    scaler = self._scaling_map.get(feature_id, self._default_scalar)
                scaler.fit_transform(samples)

        if self._group_scaling_map is not None:
            for group_feature_ids, scaler in self._group_scaling_map:
                fit = scaler.fit
                for feature_id in group_feature_ids:
                    fit(feature_samples[feature_id])
                    # Subsequent fitting must not clear the existing fitting
                    fit = scaler.partial_fit
                for feature_id in group_feature_ids:
                    scaler.transform(feature_samples[feature_id])
