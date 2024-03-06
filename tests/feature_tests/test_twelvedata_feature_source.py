# developer: taoshi-mbrown
# Copyright © 2024 Taoshi Inc
from features import FeatureID
from feature_sources import (
    TwelveDataField,
    TwelveDataTimeSeriesFeatureSource,
)
from time_util import datetime, time_span_ms
import unittest


class TestTwelveDataFeatureSource(unittest.TestCase):
    def disabled_test_twelvedata_feature_source_btc_usd_5m(self):
        _START_TIME_MS = datetime.parse("2023-01-01 00:05:00").timestamp_ms()
        _INTERVAL_MS = time_span_ms(minutes=5)
        _SAMPLE_COUNT = 7500

        test_source = TwelveDataTimeSeriesFeatureSource(
            symbol="BTC/USD",
            source_interval_ms=_INTERVAL_MS,
            feature_mappings={
                FeatureID.BTC_USD_CLOSE: TwelveDataField.PRICE_CLOSE,
                FeatureID.BTC_USD_HIGH: TwelveDataField.PRICE_HIGH,
                FeatureID.BTC_USD_LOW: TwelveDataField.PRICE_LOW,
            },
        )

        test_feature_samples = test_source.get_feature_samples(
            _START_TIME_MS, _INTERVAL_MS, _SAMPLE_COUNT
        )

        expected_values = {
            # Open time: 2023-01-01T00:00:00Z
            0: {
                FeatureID.BTC_USD_CLOSE: 16522.65039,
                FeatureID.BTC_USD_HIGH: 16531.40039,
                FeatureID.BTC_USD_LOW: 16520.28906,
            },
            # Open time: 2023-01-22T16:45:00Z
            6249: {
                FeatureID.BTC_USD_CLOSE: 22852.57031,
                FeatureID.BTC_USD_HIGH: 22852.57031,
                FeatureID.BTC_USD_LOW: 22824.25000,
            },
            # Open time: 2023-01-27T00:55:00Z
            -1: {
                FeatureID.BTC_USD_CLOSE: 22926.849619,
                FeatureID.BTC_USD_HIGH: 22938.86914,
                FeatureID.BTC_USD_LOW: 22908.21094,
            },
        }

        for index, samples in expected_values.items():
            for feature_id, expected_value in samples.items():
                test_value = float(test_feature_samples[feature_id][index])
                self.assertAlmostEqual(
                    test_value,
                    expected_value,
                    places=2,
                    msg=f"index: {index} feature_id: {feature_id}",
                )

    def disabled_test_twelvedata_feature_source_spx_usd_5m(self):
        _START_TIME_MS = datetime.parse("2024-01-04 15:05:00").timestamp_ms()
        _INTERVAL_MS = time_span_ms(minutes=5)
        _SAMPLE_COUNT = 60

        test_source = TwelveDataTimeSeriesFeatureSource(
            symbol="SPX",
            exchange="NYSE",
            source_interval_ms=_INTERVAL_MS,
            feature_mappings={
                FeatureID.SPX_USD_CLOSE: TwelveDataField.PRICE_CLOSE,
                FeatureID.SPX_USD_HIGH: TwelveDataField.PRICE_HIGH,
                FeatureID.SPX_USD_LOW: TwelveDataField.PRICE_LOW,
                FeatureID.SPX_USD_VOLUME: TwelveDataField.VOLUME,
            },
        )

        test_feature_samples = test_source.get_feature_samples(
            _START_TIME_MS, _INTERVAL_MS, _SAMPLE_COUNT
        )

        expected_values = {
            # Open time: 2024-01-04T15:00:00Z
            0: {
                FeatureID.SPX_USD_CLOSE: 4714.18994,
                FeatureID.SPX_USD_HIGH: 4714.37988,
                FeatureID.SPX_USD_LOW: 4704.56982,
                FeatureID.SPX_USD_VOLUME: 33504468,
            },
            # Open time: 2024-01-04T17:25:00Z
            29: {
                FeatureID.SPX_USD_CLOSE: 4712.93018,
                FeatureID.SPX_USD_HIGH: 4713.97021,
                FeatureID.SPX_USD_LOW: 4711.64014,
                FeatureID.SPX_USD_VOLUME: 15525027,
            },
            # Open time: 2024-01-04T19:55:00Z
            -1: {
                FeatureID.SPX_USD_CLOSE: 4703.91992,
                FeatureID.SPX_USD_HIGH: 4706.43994,
                FeatureID.SPX_USD_LOW: 4703.14014,
                FeatureID.SPX_USD_VOLUME: 15028636,
            },
        }

        for index, samples in expected_values.items():
            for feature_id, expected_value in samples.items():
                test_value = float(test_feature_samples[feature_id][index])
                self.assertAlmostEqual(
                    test_value,
                    expected_value,
                    places=2,
                    msg=f"index: {index} feature_id: {feature_id}",
                )

    def test_twelvedata_feature_source_eur_usd_5m(self):
        _START_TIME_MS = datetime.parse("2024-01-04 15:05:00").timestamp_ms()
        _INTERVAL_MS = time_span_ms(minutes=5)
        _SAMPLE_COUNT = 60

        test_source = TwelveDataTimeSeriesFeatureSource(
            symbol="EUR/USD",
            source_interval_ms=_INTERVAL_MS,
            feature_mappings={
                FeatureID.EUR_USD_CLOSE: TwelveDataField.PRICE_CLOSE,
                FeatureID.EUR_USD_HIGH: TwelveDataField.PRICE_HIGH,
                FeatureID.EUR_USD_LOW: TwelveDataField.PRICE_LOW,
            },
        )

        test_feature_samples = test_source.get_feature_samples(
            _START_TIME_MS, _INTERVAL_MS, _SAMPLE_COUNT
        )

        expected_values = {
            # Open time: 2024-01-04T15:00:00Z
            0: {
                FeatureID.EUR_USD_CLOSE: 1.09570,
                FeatureID.EUR_USD_HIGH: 1.09600,
                FeatureID.EUR_USD_LOW: 1.09455,
            },
            # Open time: 2024-01-04T17:25:00Z
            29: {
                FeatureID.EUR_USD_CLOSE: 1.09460,
                FeatureID.EUR_USD_HIGH: 1.09500,
                FeatureID.EUR_USD_LOW: 1.09450,
            },
            # Open time: 2024-01-04T19:55:00Z
            -1: {
                FeatureID.EUR_USD_CLOSE: 1.09480,
                FeatureID.EUR_USD_HIGH: 1.09500,
                FeatureID.EUR_USD_LOW: 1.09470,
            },
        }

        for index, samples in expected_values.items():
            for feature_id, expected_value in samples.items():
                test_value = float(test_feature_samples[feature_id][index])
                self.assertAlmostEqual(
                    test_value,
                    expected_value,
                    places=2,
                    msg=f"index: {index} feature_id: {feature_id}",
                )


if __name__ == "__main__":
    unittest.main()
