# developer: Taoshidev
# Copyright © 2024 Taoshi, LLC
from feature_sources import TemporalFeatureSource
from features import FeatureID
from numpy import ndarray
from time_util import datetime, time_span_ms
import unittest


class TestTemporalFeatureSource(unittest.TestCase):
    def assert_feature(
        self,
        feature_samples: dict[FeatureID, ndarray],
        feature_id: FeatureID,
        expected_value,
    ):
        self.assertAlmostEqual(feature_samples[feature_id][0], expected_value, places=2)

    def test_populate_feature_storage(self):
        _INTERVAL_MS = time_span_ms(minutes=5)

        test_feature_ids = TemporalFeatureSource.VALID_FEATURE_IDS
        test_source = TemporalFeatureSource(test_feature_ids)

        start_time_ms = datetime.parse("2023-01-01 00:00:00").timestamp_ms()
        test_samples = test_source.get_feature_samples(start_time_ms, _INTERVAL_MS, 1)

        self.assert_feature(test_samples, FeatureID.TIME_OF_DAY, -1)
        self.assert_feature(test_samples, FeatureID.TIME_OF_WEEK, -1)
        self.assert_feature(test_samples, FeatureID.TIME_OF_MONTH, -1)
        self.assert_feature(test_samples, FeatureID.TIME_OF_YEAR, -1)

        start_time_ms = datetime.parse("2023-01-01 12:00:00").timestamp_ms()
        test_samples = test_source.get_feature_samples(start_time_ms, _INTERVAL_MS, 1)

        self.assert_feature(test_samples, FeatureID.TIME_OF_DAY, 0)

        start_time_ms = datetime.parse("2023-01-04 12:00:00").timestamp_ms()
        test_samples = test_source.get_feature_samples(start_time_ms, _INTERVAL_MS, 1)

        self.assert_feature(test_samples, FeatureID.TIME_OF_WEEK, 0)

        start_time_ms = datetime.parse("2023-07-16 12:00:00").timestamp_ms()
        test_samples = test_source.get_feature_samples(start_time_ms, _INTERVAL_MS, 1)

        self.assert_feature(test_samples, FeatureID.TIME_OF_MONTH, 0)

        start_time_ms = datetime.parse("2023-07-02 12:00:00").timestamp_ms()
        test_samples = test_source.get_feature_samples(start_time_ms, _INTERVAL_MS, 1)

        self.assert_feature(test_samples, FeatureID.TIME_OF_YEAR, 0)

        start_time_ms = datetime.parse("2023-12-31 23:59:59").timestamp_ms()
        test_samples = test_source.get_feature_samples(start_time_ms, _INTERVAL_MS, 1)

        self.assert_feature(test_samples, FeatureID.TIME_OF_DAY, 1)
        self.assert_feature(test_samples, FeatureID.TIME_OF_MONTH, 1)
        self.assert_feature(test_samples, FeatureID.TIME_OF_YEAR, 1)

        start_time_ms = datetime.parse("2023-01-07 23:59:59").timestamp_ms()
        test_samples = test_source.get_feature_samples(start_time_ms, _INTERVAL_MS, 1)

        self.assert_feature(test_samples, FeatureID.TIME_OF_WEEK, 1)


if __name__ == "__main__":
    unittest.main()
