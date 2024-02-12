# developer: taoshi-mbrown
# Copyright © 2024 Taoshi, LLC
from features import FeatureID, FeatureCollector
from feature_sources import TemporalFeatureSource, TestFeatureSource
from time_util import datetime, time_span_ms
import unittest


class TestFeatureAggregator(unittest.TestCase):
    @staticmethod
    def compare_samples(
        test_source: TestFeatureSource,
        temporal_source: TemporalFeatureSource,
        feature_collector: FeatureCollector,
        start_time_ms: int,
        interval_ms: int,
        sample_count: int,
    ):
        test_samples = test_source.get_feature_samples(
            start_time_ms, interval_ms, sample_count
        )

        temporal_samples = temporal_source.get_feature_samples(
            start_time_ms, interval_ms, sample_count
        )

        collector_samples = feature_collector.get_feature_samples(
            start_time_ms, interval_ms, sample_count
        )

        for feature_id in test_source.feature_ids:
            # noinspection PyUnresolvedReferences
            assert (collector_samples[feature_id] == test_samples[feature_id]).all()

        for feature_id in temporal_source.feature_ids:
            # noinspection PyUnresolvedReferences
            assert (collector_samples[feature_id] == temporal_samples[feature_id]).all()

    def test_binance_bybit_coinbase_kline_feature_aggregator(self):
        _START_TIME_MS = datetime.parse("2023-01-01 00:00:00").timestamp_ms()
        _INTERVAL_MS = time_span_ms(minutes=5)
        _SAMPLE_COUNT = 2500
        _LOOK_BACK_SIZE = 1000
        _JUMP_AHEAD_ITERATIONS = 4

        test_feature_ids = [
            FeatureID.BTC_USD_CLOSE,
            FeatureID.BTC_USD_HIGH,
            FeatureID.BTC_USD_LOW,
            FeatureID.BTC_USD_VOLUME,
        ]
        test_source = TestFeatureSource(test_feature_ids)

        temporal_feature_ids = TemporalFeatureSource.VALID_FEATURE_IDS
        temporal_source = TemporalFeatureSource(temporal_feature_ids)

        feature_collector = FeatureCollector(
            sources=[test_source, temporal_source],
            cache_results=True,
        )

        self.compare_samples(
            test_source,
            temporal_source,
            feature_collector,
            _START_TIME_MS,
            _INTERVAL_MS,
            _SAMPLE_COUNT,
        )

        start_time_ms = (
            _START_TIME_MS
            + (_SAMPLE_COUNT * _INTERVAL_MS)
            - (_LOOK_BACK_SIZE * _INTERVAL_MS)
        )

        self.compare_samples(
            test_source,
            temporal_source,
            feature_collector,
            start_time_ms,
            _INTERVAL_MS,
            _SAMPLE_COUNT,
        )

        start_time_ms = _START_TIME_MS + (
            _JUMP_AHEAD_ITERATIONS * _SAMPLE_COUNT * _INTERVAL_MS
        )

        self.compare_samples(
            test_source,
            temporal_source,
            feature_collector,
            start_time_ms,
            _INTERVAL_MS,
            _SAMPLE_COUNT,
        )


if __name__ == "__main__":
    unittest.main()
