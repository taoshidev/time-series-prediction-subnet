# developer: Taoshidev
# Copyright © 2024 Taoshi, LLC
from feature_sources.binance_kline_feature_source import (
    BinanceKlineFeatureSource,
    BinanceKlineField,
)
from features import FeatureID
from time_util import datetime, time_span_ms
import unittest


class TestOHLCVFeatureSource(unittest.TestCase):
    def test_binance_ohlcv_feature_storage(self):
        _START_TIME_MS = datetime.parse("2023-01-01 00:00:00").timestamp_ms()
        _INTERVAL_MS = time_span_ms(minutes=5)
        _SAMPLE_COUNT = 2500
        _ITERATIONS = 3

        test_source = BinanceKlineFeatureSource(
            symbol="BTCUSDT",
            feature_mappings={
                FeatureID.BTC_USD_CLOSE: BinanceKlineField.PRICE_CLOSE,
                FeatureID.BTC_USD_HIGH: BinanceKlineField.PRICE_HIGH,
                FeatureID.BTC_USD_LOW: BinanceKlineField.PRICE_LOW,
                FeatureID.BTC_USD_VOLUME: BinanceKlineField.VOLUME,
            },
        )

        all_feature_samples = {feature_id: [] for feature_id in test_source.feature_ids}
        start_time_ms = _START_TIME_MS
        for i in range(_ITERATIONS):
            feature_samples = test_source.get_feature_samples(
                start_time_ms, _INTERVAL_MS, _SAMPLE_COUNT
            )
            for feature_id, samples in feature_samples.items():
                all_feature_samples[feature_id].extend(samples)
            start_time_ms += _INTERVAL_MS * _SAMPLE_COUNT

        expected_values = {
            # Open time: 1672530900000
            0: {
                FeatureID.BTC_USD_CLOSE: 16542.40000000,
                FeatureID.BTC_USD_HIGH: 16544.47000000,
                FeatureID.BTC_USD_LOW: 16535.05000000,
                FeatureID.BTC_USD_VOLUME: 227.06684000,
            },
            # Open time: 1674405600000
            6249: {
                FeatureID.BTC_USD_CLOSE: 22831.88000000,
                FeatureID.BTC_USD_HIGH: 22831.88000000,
                FeatureID.BTC_USD_LOW: 22797.00000000,
                FeatureID.BTC_USD_VOLUME: 665.50900000,
            },
            # Open time: 1674780600000
            -1: {
                FeatureID.BTC_USD_CLOSE: 22913.75000000,
                FeatureID.BTC_USD_HIGH: 22982.91000000,
                FeatureID.BTC_USD_LOW: 22897.02000000,
                FeatureID.BTC_USD_VOLUME: 1445.37762000,
            },
        }

        for index, samples in expected_values.items():
            for feature_id, expected_value in samples.items():
                self.assertAlmostEqual(
                    all_feature_samples[feature_id][index], expected_value, delta=6
                )


if __name__ == "__main__":
    unittest.main()
