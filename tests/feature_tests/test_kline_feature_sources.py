# developer: taoshi-mbrown
# Copyright © 2024 Taoshi, LLC
from features import FeatureID
from feature_sources import (
    BinanceKlineFeatureSource,
    BinanceKlineField,
    BybitKlineFeatureSource,
    BybitKlineField,
    CoinbaseKlineFeatureSource,
    CoinbaseKlineField,
    KrakenKlineFeatureSource,
    KrakenKlineField,
)
from time_util import datetime, time_span_ms, previous_interval_ms
import unittest


class TestKlineFeatureSource(unittest.TestCase):
    def test_binance_kline_feature_source(self):
        _START_TIME_MS = datetime.parse("2023-01-01 00:00:00").timestamp_ms()
        _INTERVAL_MS = time_span_ms(minutes=5)
        _SAMPLE_COUNT = 2500
        _ITERATIONS = 3

        test_source = BinanceKlineFeatureSource(
            symbol="BTCUSDT",
            interval_ms=time_span_ms(minutes=5),
            feature_mappings={
                FeatureID.BTC_USD_CLOSE: BinanceKlineField.PRICE_CLOSE,
                FeatureID.BTC_USD_HIGH: BinanceKlineField.PRICE_HIGH,
                FeatureID.BTC_USD_LOW: BinanceKlineField.PRICE_LOW,
                FeatureID.BTC_USD_VOLUME: BinanceKlineField.VOLUME,
            },
        )

        test_feature_samples = {
            feature_id: [] for feature_id in test_source.feature_ids
        }
        start_time_ms = _START_TIME_MS
        for i in range(_ITERATIONS):
            feature_samples = test_source.get_feature_samples(
                start_time_ms, _INTERVAL_MS, _SAMPLE_COUNT
            )
            for feature_id, samples in feature_samples.items():
                test_feature_samples[feature_id].extend(samples)
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
                    test_feature_samples[feature_id][index],
                    expected_value,
                    places=2,
                    msg=f"index: {index} feature_id: {feature_id}",
                )

    def test_bybit_kline_feature_source(self):
        _START_TIME_MS = datetime.parse("2023-01-01 00:00:00").timestamp_ms()
        _INTERVAL_MS = time_span_ms(minutes=5)
        _SAMPLE_COUNT = 2500
        _ITERATIONS = 3

        test_source = BybitKlineFeatureSource(
            category="spot",
            symbol="BTCUSDT",
            interval_ms=time_span_ms(minutes=5),
            feature_mappings={
                FeatureID.BTC_USD_CLOSE: BybitKlineField.PRICE_CLOSE,
                FeatureID.BTC_USD_HIGH: BybitKlineField.PRICE_HIGH,
                FeatureID.BTC_USD_LOW: BybitKlineField.PRICE_LOW,
                FeatureID.BTC_USD_VOLUME: BybitKlineField.VOLUME,
            },
        )

        test_feature_samples = {
            feature_id: [] for feature_id in test_source.feature_ids
        }
        start_time_ms = _START_TIME_MS
        for i in range(_ITERATIONS):
            feature_samples = test_source.get_feature_samples(
                start_time_ms, _INTERVAL_MS, _SAMPLE_COUNT
            )
            for feature_id, samples in feature_samples.items():
                test_feature_samples[feature_id].extend(samples)
            start_time_ms += _INTERVAL_MS * _SAMPLE_COUNT

        expected_values = {
            # Open time: 1672530900000
            0: {
                FeatureID.BTC_USD_CLOSE: 16541.8,
                FeatureID.BTC_USD_HIGH: 16542.97,
                FeatureID.BTC_USD_LOW: 16534.59,
                FeatureID.BTC_USD_VOLUME: 8.600131,
            },
            # Open time: 1674405600000
            6249: {
                FeatureID.BTC_USD_CLOSE: 22828.44,
                FeatureID.BTC_USD_HIGH: 22829.99,
                FeatureID.BTC_USD_LOW: 22802.25,
                FeatureID.BTC_USD_VOLUME: 13.348506,
            },
            # Open time: 1674780600000
            -1: {
                FeatureID.BTC_USD_CLOSE: 22907.61,
                FeatureID.BTC_USD_HIGH: 22979.14,
                FeatureID.BTC_USD_LOW: 22900.01,
                FeatureID.BTC_USD_VOLUME: 27.59141,
            },
        }

        for index, samples in expected_values.items():
            for feature_id, expected_value in samples.items():
                self.assertAlmostEqual(
                    test_feature_samples[feature_id][index],
                    expected_value,
                    places=2,
                    msg=f"index: {index} feature_id: {feature_id}",
                )

    def test_coinbase_kline_feature_source(self):
        _START_TIME_MS = datetime.parse("2023-01-01 00:00:00").timestamp_ms()
        _INTERVAL_MS = time_span_ms(minutes=5)
        _SAMPLE_COUNT = 2500
        _ITERATIONS = 3

        test_source = CoinbaseKlineFeatureSource(
            symbol="BTC-USD",
            interval_ms=time_span_ms(minutes=5),
            feature_mappings={
                FeatureID.BTC_USD_CLOSE: CoinbaseKlineField.PRICE_CLOSE,
                FeatureID.BTC_USD_HIGH: CoinbaseKlineField.PRICE_HIGH,
                FeatureID.BTC_USD_LOW: CoinbaseKlineField.PRICE_LOW,
                FeatureID.BTC_USD_VOLUME: CoinbaseKlineField.VOLUME,
            },
        )

        test_feature_samples = {
            feature_id: [] for feature_id in test_source.feature_ids
        }
        start_time_ms = _START_TIME_MS
        for i in range(_ITERATIONS):
            feature_samples = test_source.get_feature_samples(
                start_time_ms, _INTERVAL_MS, _SAMPLE_COUNT
            )
            for feature_id, samples in feature_samples.items():
                test_feature_samples[feature_id].extend(samples)
            start_time_ms += _INTERVAL_MS * _SAMPLE_COUNT

        expected_values = {
            # Open time: 1672530900 (seconds)
            0: {
                FeatureID.BTC_USD_CLOSE: 16530.35,
                FeatureID.BTC_USD_HIGH: 16532,
                FeatureID.BTC_USD_LOW: 16525.5,
                FeatureID.BTC_USD_VOLUME: 65.73377818,
            },
            # Open time: 1674405600 (seconds)
            6249: {
                FeatureID.BTC_USD_CLOSE: 22835.7,
                FeatureID.BTC_USD_HIGH: 22835.7,
                FeatureID.BTC_USD_LOW: 22801.94,
                FeatureID.BTC_USD_VOLUME: 44.66172957,
            },
            # Open time: 1674780600 (seconds)
            -1: {
                FeatureID.BTC_USD_CLOSE: 22917.36,
                FeatureID.BTC_USD_HIGH: 22985.1,
                FeatureID.BTC_USD_LOW: 22900.4,
                FeatureID.BTC_USD_VOLUME: 146.46861788,
            },
        }

        for index, samples in expected_values.items():
            for feature_id, expected_value in samples.items():
                self.assertAlmostEqual(
                    test_feature_samples[feature_id][index],
                    expected_value,
                    places=2,
                    msg=f"index: {index} feature_id: {feature_id}",
                )

    def test_kraken_kline_feature_source(self):
        _INTERVAL_MS = time_span_ms(minutes=5)
        _SKIP_COUNT = 10
        _SAMPLE_COUNT = 100
        _START_TIME_MS = previous_interval_ms(
            datetime.now().timestamp_ms(), _INTERVAL_MS
        ) - ((_SAMPLE_COUNT + _SKIP_COUNT) * _INTERVAL_MS)
        _BTC_USD_LOW_MIN = 30000
        _BTC_USD_HIGH_MAX = 70000

        test_source = KrakenKlineFeatureSource(
            symbol="XXBTZUSD",
            interval_ms=time_span_ms(minutes=5),
            feature_mappings={
                FeatureID.BTC_USD_CLOSE: KrakenKlineField.PRICE_CLOSE,
                FeatureID.BTC_USD_HIGH: KrakenKlineField.PRICE_HIGH,
                FeatureID.BTC_USD_LOW: KrakenKlineField.PRICE_LOW,
                FeatureID.BTC_USD_VOLUME: KrakenKlineField.VOLUME,
            },
        )

        test_feature_samples = test_source.get_feature_samples(
            _START_TIME_MS, _INTERVAL_MS, _SAMPLE_COUNT
        )

        last_close = None
        last_high = None
        last_low = None
        last_volume = None
        for i in range(_SAMPLE_COUNT):
            close = test_feature_samples[FeatureID.BTC_USD_CLOSE][i]
            high = test_feature_samples[FeatureID.BTC_USD_HIGH][i]
            low = test_feature_samples[FeatureID.BTC_USD_LOW][i]
            volume = test_feature_samples[FeatureID.BTC_USD_VOLUME][i]
            assert close != last_close
            assert high != last_high
            assert high >= close
            assert high < _BTC_USD_HIGH_MAX
            assert low != last_low
            assert low <= close
            assert low > _BTC_USD_LOW_MIN
            assert volume != last_volume


if __name__ == "__main__":
    unittest.main()
