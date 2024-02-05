# developer: taoshi-mbrown
# Copyright © 2024 Taoshi, LLC
from feature_sources import BinaryFileFeatureStorage, TemporalFeatureSource
from time_util import time_span_ms
import unittest


class TestBinaryFileFeatureStorage(unittest.TestCase):
    def test_populate_feature_storage(self):
        _FILE_NAME = "temporal-test.taosfs"
        _START_TIME_MS = 0
        _INTERVAL_MS = time_span_ms(minutes=5)
        _SAMPLE_COUNT = 1000
        _ITERATIONS = 5

        temporal_feature_ids = TemporalFeatureSource.VALID_FEATURE_IDS
        temporal_feature_source = TemporalFeatureSource(temporal_feature_ids)

        test_storage_output = BinaryFileFeatureStorage(
            filename=_FILE_NAME,
            mode="w",
            feature_ids=temporal_feature_ids,
        )

        start_time_ms = _START_TIME_MS
        for i in range(_ITERATIONS):
            feature_samples = temporal_feature_source.get_feature_samples(
                start_time_ms, _INTERVAL_MS, _SAMPLE_COUNT
            )
            test_storage_output.set_feature_samples(
                start_time_ms, _INTERVAL_MS, feature_samples
            )
            start_time_ms += _INTERVAL_MS * _SAMPLE_COUNT

        test_storage_output.close()

        test_storage_input = BinaryFileFeatureStorage(
            filename=_FILE_NAME,
            mode="r",
            feature_ids=temporal_feature_ids,
        )

        start_time_ms = _START_TIME_MS
        for i in range(_ITERATIONS):
            expected_feature_samples = temporal_feature_source.get_feature_samples(
                start_time_ms, _INTERVAL_MS, _SAMPLE_COUNT
            )
            stored_feature_samples = test_storage_input.get_feature_samples(
                start_time_ms, _INTERVAL_MS, _SAMPLE_COUNT
            )

            for feature_id in temporal_feature_ids:
                # noinspection PyUnresolvedReferences
                assert (
                    stored_feature_samples[feature_id]
                    == expected_feature_samples[feature_id]
                ).all()

            start_time_ms += _INTERVAL_MS * _SAMPLE_COUNT

        # Backwards
        for i in range(_ITERATIONS):
            start_time_ms -= _INTERVAL_MS * _SAMPLE_COUNT

            expected_feature_samples = temporal_feature_source.get_feature_samples(
                start_time_ms, _INTERVAL_MS, _SAMPLE_COUNT
            )
            stored_feature_samples = test_storage_input.get_feature_samples(
                start_time_ms, _INTERVAL_MS, _SAMPLE_COUNT
            )

            for feature_id in temporal_feature_ids:
                # noinspection PyUnresolvedReferences
                assert (
                    stored_feature_samples[feature_id]
                    == expected_feature_samples[feature_id]
                ).all()

        test_storage_input.close()


if __name__ == "__main__":
    unittest.main()
