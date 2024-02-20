# developer: taoshi-mbrown
# Copyright © 2024 Taoshi Inc
from features import FeatureID, FeatureAggregator
from feature_sources import (
    BinanceKlineFeatureSource,
    BinanceKlineField,
    BybitKlineFeatureSource,
    BybitKlineField,
    CoinbaseKlineFeatureSource,
    CoinbaseKlineField,
)
from statistics import fmean
from time_util import datetime, time_span_ms
import unittest


class TestFeatureAggregator(unittest.TestCase):
    def test_binance_bybit_coinbase_kline_feature_aggregator(self):
        _START_TIME_MS = datetime.parse("2023-01-01 00:00:00").timestamp_ms()
        _INTERVAL_MS = time_span_ms(minutes=5)
        _SAMPLE_COUNT = 2500
        _ITERATIONS = 3

        binance_source = BinanceKlineFeatureSource(
            symbol="BTCUSDT",
            source_interval_ms=time_span_ms(minutes=5),
            feature_mappings={
                FeatureID.BTC_USD_CLOSE: BinanceKlineField.PRICE_CLOSE,
                FeatureID.BTC_USD_HIGH: BinanceKlineField.PRICE_HIGH,
                FeatureID.BTC_USD_LOW: BinanceKlineField.PRICE_LOW,
                FeatureID.BTC_USD_VOLUME: BinanceKlineField.VOLUME,
            },
        )

        bybit_source = BybitKlineFeatureSource(
            category="spot",
            symbol="BTCUSDT",
            source_interval_ms=time_span_ms(minutes=5),
            feature_mappings={
                FeatureID.BTC_USD_CLOSE: BybitKlineField.PRICE_CLOSE,
                FeatureID.BTC_USD_HIGH: BybitKlineField.PRICE_HIGH,
                FeatureID.BTC_USD_LOW: BybitKlineField.PRICE_LOW,
                FeatureID.BTC_USD_VOLUME: BybitKlineField.VOLUME,
            },
        )

        coinbase_source = CoinbaseKlineFeatureSource(
            symbol="BTC-USD",
            source_interval_ms=time_span_ms(minutes=5),
            feature_mappings={
                FeatureID.BTC_USD_CLOSE: CoinbaseKlineField.PRICE_CLOSE,
                FeatureID.BTC_USD_HIGH: CoinbaseKlineField.PRICE_HIGH,
                FeatureID.BTC_USD_LOW: CoinbaseKlineField.PRICE_LOW,
                FeatureID.BTC_USD_VOLUME: CoinbaseKlineField.VOLUME,
            },
        )

        test_aggregator = FeatureAggregator(
            sources=[binance_source, bybit_source, coinbase_source],
            aggregation_map={
                FeatureID.BTC_USD_CLOSE: fmean,
                FeatureID.BTC_USD_HIGH: max,
                FeatureID.BTC_USD_LOW: min,
                FeatureID.BTC_USD_VOLUME: sum,
            },
        )

        test_feature_samples = {
            feature_id: [] for feature_id in test_aggregator.feature_ids
        }
        start_time_ms = _START_TIME_MS
        for i in range(_ITERATIONS):
            feature_samples = test_aggregator.get_feature_samples(
                start_time_ms, _INTERVAL_MS, _SAMPLE_COUNT
            )
            for feature_id, samples in feature_samples.items():
                test_feature_samples[feature_id].extend(samples)
            start_time_ms += _INTERVAL_MS * _SAMPLE_COUNT

        expected_values = {
            # Open time: 1672530900000
            0: {
                FeatureID.BTC_USD_CLOSE: fmean((16542.40000000, 16541.8, 16530.35)),
                FeatureID.BTC_USD_HIGH: max(16544.47000000, 16542.97, 16532),
                FeatureID.BTC_USD_LOW: min(16535.05000000, 16534.59, 16525.5),
                FeatureID.BTC_USD_VOLUME: sum((227.06684000, 8.600131, 65.73377818)),
            },
            # Open time: 1674405600000
            6249: {
                FeatureID.BTC_USD_CLOSE: fmean((22831.88000000, 22828.44, 22835.7)),
                FeatureID.BTC_USD_HIGH: max(22831.88000000, 22829.99, 22835.7),
                FeatureID.BTC_USD_LOW: min(22797.00000000, 22802.25, 22801.94),
                FeatureID.BTC_USD_VOLUME: sum((665.50900000, 13.348506, 44.66172957)),
            },
            # Open time: 1674780600000
            -1: {
                FeatureID.BTC_USD_CLOSE: fmean((22913.75000000, 22907.61, 22917.36)),
                FeatureID.BTC_USD_HIGH: max(22982.91000000, 22979.14, 22985.1),
                FeatureID.BTC_USD_LOW: min(22897.02000000, 22900.01, 22900.4),
                FeatureID.BTC_USD_VOLUME: sum((1445.37762000, 27.59141, 146.46861788)),
            },
        }

        for index, samples in expected_values.items():
            for feature_id, expected_value in samples.items():
                test_value = test_feature_samples[feature_id][index]
                self.assertAlmostEqual(
                    test_value,
                    expected_value,
                    places=2,
                    msg=f"index: {index} feature_id: {feature_id}",
                )


if __name__ == "__main__":
    unittest.main()
