# developer: taoshi-mbrown
# Copyright © 2024 Taoshi Inc
from features import FeatureID
from feature_sources import TestFeatureSource
import numpy as np
from time_util import datetime, time_span_ms
import unittest


class TestFeatureSourceInternals(unittest.TestCase):
    def test_check_feature_samples(self):
        test_feature_ids = [
            FeatureID.BTC_USD_CLOSE,
        ]

        test_source = TestFeatureSource(test_feature_ids)

        now_time_ms = datetime.now().timestamp_ms()
        interval_ms = time_span_ms(minutes=5)

        test_samples = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        test_feature_samples = {FeatureID.BTC_USD_CLOSE: test_samples}

        test_source._check_feature_samples(
            test_feature_samples, now_time_ms, interval_ms
        )

        test_samples[3] = np.nan
        with self.assertRaises(RuntimeError):
            test_source._check_feature_samples(
                test_feature_samples, now_time_ms, interval_ms
            )

        test_samples[3] = np.inf
        with self.assertRaises(RuntimeError):
            test_source._check_feature_samples(
                test_feature_samples, now_time_ms, interval_ms
            )

        test_samples[3] = -np.inf
        with self.assertRaises(RuntimeError):
            test_source._check_feature_samples(
                test_feature_samples, now_time_ms, interval_ms
            )


if __name__ == "__main__":
    unittest.main()
