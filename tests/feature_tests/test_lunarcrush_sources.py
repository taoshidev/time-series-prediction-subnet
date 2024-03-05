# developer: taoshi-mbrown
# Copyright © 2024 Taoshi Inc
from features import FeatureID
from feature_sources import (
    LunarCrushMetric,
    LunarCrushTimeSeriesCategory,
    LunarCrushTimeSeriesCoin,
    LunarCrushTimeSeriesStock,
    LunarCrushTimeSeriesTopic,
)
import math
from time_util import datetime, time_span_ms
import unittest


class TestLunarCrushFeatureSources(unittest.TestCase):
    def test_lunarcrush_category_feature_source(self):
        _START_TIME_MS = datetime.parse("2024-02-01 01:00:00").timestamp_ms()
        _INTERVAL_MS = time_span_ms(hours=1)
        _SAMPLE_COUNT = 500

        test_source = LunarCrushTimeSeriesCategory(
            category="cryptocurrencies",
            source_interval_ms=time_span_ms(hours=1),
            feature_mappings={
                FeatureID.CRYPTO_SOCIAL_POSTS_CREATED: LunarCrushMetric.POSTS_CREATED,
                FeatureID.CRYPTO_SOCIAL_POSTS_ACTIVE: LunarCrushMetric.POSTS_ACTIVE,
                FeatureID.CRYPTO_SOCIAL_INTERACTIONS: LunarCrushMetric.INTERACTIONS,
                FeatureID.CRYPTO_SOCIAL_CONTRIBUTORS_CREATED: LunarCrushMetric.CONTRIBUTORS_CREATED,
                FeatureID.CRYPTO_SOCIAL_CONTRIBUTORS_ACTIVE: LunarCrushMetric.CONTRIBUTORS_ACTIVE,
                FeatureID.CRYPTO_SOCIAL_SENTIMENT: LunarCrushMetric.SENTIMENT,
                FeatureID.CRYPTO_SOCIAL_SPAM: LunarCrushMetric.SPAM,
            },
        )

        test_feature_samples = test_source.get_feature_samples(
            _START_TIME_MS, _INTERVAL_MS, _SAMPLE_COUNT
        )

        expected_values = {
            # start: 1706745600
            0: {
                FeatureID.CRYPTO_SOCIAL_POSTS_CREATED: 7508,
                FeatureID.CRYPTO_SOCIAL_POSTS_ACTIVE: 7631,
                FeatureID.CRYPTO_SOCIAL_INTERACTIONS: 300300,
                FeatureID.CRYPTO_SOCIAL_CONTRIBUTORS_CREATED: 5339,
                FeatureID.CRYPTO_SOCIAL_CONTRIBUTORS_ACTIVE: 10441,
                FeatureID.CRYPTO_SOCIAL_SENTIMENT: 79,
                FeatureID.CRYPTO_SOCIAL_SPAM: 900,
            },
            # start: 1707656400
            253: {
                FeatureID.CRYPTO_SOCIAL_POSTS_CREATED: 7098,
                FeatureID.CRYPTO_SOCIAL_POSTS_ACTIVE: 150224,
                FeatureID.CRYPTO_SOCIAL_INTERACTIONS: 28660679,
                FeatureID.CRYPTO_SOCIAL_CONTRIBUTORS_CREATED: 4645,
                FeatureID.CRYPTO_SOCIAL_CONTRIBUTORS_ACTIVE: 66932,
                FeatureID.CRYPTO_SOCIAL_SENTIMENT: 83,
                FeatureID.CRYPTO_SOCIAL_SPAM: 0,
            },
            # start: 1708542000
            -1: {
                FeatureID.CRYPTO_SOCIAL_POSTS_CREATED: 6947,
                FeatureID.CRYPTO_SOCIAL_POSTS_ACTIVE: 135999,
                FeatureID.CRYPTO_SOCIAL_INTERACTIONS: 19205146,
                FeatureID.CRYPTO_SOCIAL_CONTRIBUTORS_CREATED: 5023,
                FeatureID.CRYPTO_SOCIAL_CONTRIBUTORS_ACTIVE: 61985,
                FeatureID.CRYPTO_SOCIAL_SENTIMENT: 83,
                FeatureID.CRYPTO_SOCIAL_SPAM: 1389,
            },
        }

        for index, samples in expected_values.items():
            for feature_id, expected_value in samples.items():
                test_value = test_feature_samples[feature_id][index]
                self.assertTrue(
                    math.isclose(
                        test_value,
                        expected_value,
                        rel_tol=1e-07,
                    ),
                    msg=f"index: {index} feature_id: {feature_id.name} "
                    f"test_value:{test_value} expected_value: {expected_value}",
                )

    def test_lunarcrush_coin_feature_source(self):
        _START_TIME_MS = datetime.parse("2024-02-01 01:00:00").timestamp_ms()
        _INTERVAL_MS = time_span_ms(hours=1)
        _SAMPLE_COUNT = 500

        test_source = LunarCrushTimeSeriesCoin(
            coin="bitcoin",
            source_interval_ms=time_span_ms(hours=1),
            feature_mappings={
                FeatureID.BTC_USD_OPEN: LunarCrushMetric.PRICE_OPEN,
                FeatureID.BTC_USD_CLOSE: LunarCrushMetric.PRICE_CLOSE,
                FeatureID.BTC_USD_HIGH: LunarCrushMetric.PRICE_HIGH,
                FeatureID.BTC_USD_LOW: LunarCrushMetric.PRICE_LOW,
                FeatureID.BTC_USD_VOLUME: LunarCrushMetric.VOLUME_24H,
                FeatureID.BTC_USD_MARKET_CAP: LunarCrushMetric.MARKET_CAP,
                FeatureID.BTC_CIRCULATING_SUPPLY: LunarCrushMetric.CIRCULATING_SUPPLY,
                FeatureID.BTC_SOCIAL_SENTIMENT: LunarCrushMetric.SENTIMENT,
                FeatureID.BTC_GALAXY_SCORE: LunarCrushMetric.GALAXY_SCORE,
                FeatureID.BTC_USD_VOLATILITY: LunarCrushMetric.VOLATILITY,
                FeatureID.BTC_ALT_RANK: LunarCrushMetric.ALT_RANK,
                FeatureID.BTC_SOCIAL_POSTS_CREATED: LunarCrushMetric.POSTS_CREATED,
                FeatureID.BTC_SOCIAL_POSTS_ACTIVE: LunarCrushMetric.POSTS_ACTIVE,
                FeatureID.BTC_SOCIAL_CONTRIBUTORS_CREATED: LunarCrushMetric.CONTRIBUTORS_CREATED,
                FeatureID.BTC_SOCIAL_CONTRIBUTORS_ACTIVE: LunarCrushMetric.CONTRIBUTORS_ACTIVE,
                FeatureID.BTC_SOCIAL_INTERACTIONS: LunarCrushMetric.INTERACTIONS,
                FeatureID.BTC_SOCIAL_DOMINANCE: LunarCrushMetric.SOCIAL_DOMINANCE,
            },
        )

        test_feature_samples = test_source.get_feature_samples(
            _START_TIME_MS, _INTERVAL_MS, _SAMPLE_COUNT
        )

        expected_values = {
            # start: 1706745600
            0: {
                FeatureID.BTC_USD_OPEN: 42569.76139843987,
                FeatureID.BTC_USD_CLOSE: 42421.03004299571,
                FeatureID.BTC_USD_HIGH: 42657.611350802246,
                FeatureID.BTC_USD_LOW: 42222.44063408701,
                FeatureID.BTC_USD_VOLUME: 24549499233.53,
                FeatureID.BTC_USD_MARKET_CAP: 832064112201.09,
                FeatureID.BTC_CIRCULATING_SUPPLY: 19614425,
                FeatureID.BTC_SOCIAL_SENTIMENT: 78,
                FeatureID.BTC_GALAXY_SCORE: 75,
                FeatureID.BTC_USD_VOLATILITY: 0.008550716661002312,
                FeatureID.BTC_ALT_RANK: 32,
                FeatureID.BTC_SOCIAL_POSTS_CREATED: 1117,
                FeatureID.BTC_SOCIAL_POSTS_ACTIVE: 32130,
                FeatureID.BTC_SOCIAL_CONTRIBUTORS_CREATED: 757,
                FeatureID.BTC_SOCIAL_CONTRIBUTORS_ACTIVE: 16399,
                FeatureID.BTC_SOCIAL_INTERACTIONS: 4263864,
                FeatureID.BTC_SOCIAL_DOMINANCE: 0,
            },
            # start: 1707656400
            253: {
                FeatureID.BTC_USD_OPEN: 48321.70719797294,
                FeatureID.BTC_USD_CLOSE: 48100.66526716035,
                FeatureID.BTC_USD_HIGH: 48371.15928848688,
                FeatureID.BTC_USD_LOW: 48065.92147415219,
                FeatureID.BTC_USD_VOLUME: 19354861474.99,
                FeatureID.BTC_USD_MARKET_CAP: 943966801546.9432,
                FeatureID.BTC_CIRCULATING_SUPPLY: 19624818,
                FeatureID.BTC_SOCIAL_SENTIMENT: 81,
                FeatureID.BTC_GALAXY_SCORE: 70,
                FeatureID.BTC_USD_VOLATILITY: 0.0089,
                FeatureID.BTC_ALT_RANK: 47,
                FeatureID.BTC_SOCIAL_POSTS_CREATED: 1411,
                FeatureID.BTC_SOCIAL_POSTS_ACTIVE: 33865,
                FeatureID.BTC_SOCIAL_CONTRIBUTORS_CREATED: 935,
                FeatureID.BTC_SOCIAL_CONTRIBUTORS_ACTIVE: 15689,
                FeatureID.BTC_SOCIAL_INTERACTIONS: 8055386,
                FeatureID.BTC_SOCIAL_DOMINANCE: 22.54300244967515,
            },
            # start: 1708542000
            -1: {
                FeatureID.BTC_USD_OPEN: 50864.29735630042,
                FeatureID.BTC_USD_CLOSE: 51048.33614009864,
                FeatureID.BTC_USD_HIGH: 51048.33614009864,
                FeatureID.BTC_USD_LOW: 50664.7403283826,
                FeatureID.BTC_USD_VOLUME: 27995847309.48,
                FeatureID.BTC_USD_MARKET_CAP: 1002280479357.89,
                FeatureID.BTC_CIRCULATING_SUPPLY: 19633950,
                FeatureID.BTC_SOCIAL_SENTIMENT: 82,
                FeatureID.BTC_GALAXY_SCORE: 70,
                FeatureID.BTC_USD_VOLATILITY: 0.0083,
                FeatureID.BTC_ALT_RANK: 87,
                FeatureID.BTC_SOCIAL_POSTS_CREATED: 1834,
                FeatureID.BTC_SOCIAL_POSTS_ACTIVE: 49356,
                FeatureID.BTC_SOCIAL_CONTRIBUTORS_CREATED: 1234,
                FeatureID.BTC_SOCIAL_CONTRIBUTORS_ACTIVE: 23064,
                FeatureID.BTC_SOCIAL_INTERACTIONS: 10123046,
                FeatureID.BTC_SOCIAL_DOMINANCE: 36.291443319436176,
            },
        }

        for index, samples in expected_values.items():
            for feature_id, expected_value in samples.items():
                test_value = test_feature_samples[feature_id][index]
                self.assertTrue(
                    math.isclose(
                        test_value,
                        expected_value,
                        rel_tol=1e-07,
                    ),
                    msg=f"index: {index} feature_id: {feature_id.name} "
                    f"test_value:{test_value} expected_value: {expected_value}",
                )

    def lunarcrush_stock_feature_source(self):
        _START_TIME_MS = datetime.parse("2024-02-01 01:00:00").timestamp_ms()
        _INTERVAL_MS = time_span_ms(hours=1)
        _SAMPLE_COUNT = 500

        test_source = LunarCrushTimeSeriesStock(
            stock="nvda",
            source_interval_ms=time_span_ms(hours=1),
            feature_mappings={
                FeatureID.NVDA_USD_OPEN: LunarCrushMetric.PRICE_OPEN,
                FeatureID.NVDA_USD_CLOSE: LunarCrushMetric.PRICE_CLOSE,
                FeatureID.NVDA_USD_HIGH: LunarCrushMetric.PRICE_HIGH,
                FeatureID.NVDA_USD_LOW: LunarCrushMetric.PRICE_LOW,
                FeatureID.NVDA_USD_VOLUME: LunarCrushMetric.VOLUME,
            },
        )

        test_feature_samples = test_source.get_feature_samples(
            _START_TIME_MS, _INTERVAL_MS, _SAMPLE_COUNT
        )

        expected_values = {
            # start: 1706745600
            0: {
                FeatureID.NVDA_USD_OPEN: 616.93,
                FeatureID.NVDA_USD_CLOSE: 618.23,
                FeatureID.NVDA_USD_HIGH: 618.25,
                FeatureID.NVDA_USD_LOW: 616.65,
                FeatureID.NVDA_USD_VOLUME: 49327112.28,
            },
            # start: 1707656400
            # (markets are closed on Sunday, so no volume)
            253: {
                FeatureID.NVDA_USD_OPEN: 721.31,
                FeatureID.NVDA_USD_CLOSE: 721.31,
                FeatureID.NVDA_USD_HIGH: 721.31,
                FeatureID.NVDA_USD_LOW: 721.31,
                FeatureID.NVDA_USD_VOLUME: 0,
            },
            # start: 1707742800
            277: {
                FeatureID.NVDA_USD_OPEN: 727,
                FeatureID.NVDA_USD_CLOSE: 725.68,
                FeatureID.NVDA_USD_HIGH: 728.4799,
                FeatureID.NVDA_USD_LOW: 722,
                FeatureID.NVDA_USD_VOLUME: 393150789.56,
            },
            # start: 1708542000
            -1: {
                FeatureID.NVDA_USD_OPEN: 670.46,
                FeatureID.NVDA_USD_CLOSE: 667.736,
                FeatureID.NVDA_USD_HIGH: 674.95,
                FeatureID.NVDA_USD_LOW: 666.74,
                FeatureID.NVDA_USD_VOLUME: 4099908587.69,
            },
        }

        for index, samples in expected_values.items():
            for feature_id, expected_value in samples.items():
                test_value = test_feature_samples[feature_id][index]
                self.assertTrue(
                    math.isclose(
                        test_value,
                        expected_value,
                        rel_tol=1e-07,
                    ),
                    msg=f"index: {index} feature_id: {feature_id.name} "
                    f"test_value:{test_value} expected_value: {expected_value}",
                )

    def test_lunarcrush_topic_feature_source(self):
        _START_TIME_MS = datetime.parse("2023-01-01 01:00:00").timestamp_ms()
        _INTERVAL_MS = time_span_ms(hours=1)
        _SAMPLE_COUNT = 7500

        test_source = LunarCrushTimeSeriesTopic(
            topic="bitcoin",
            source_interval_ms=time_span_ms(hours=1),
            feature_mappings={
                FeatureID.BTC_SOCIAL_POSTS_CREATED: LunarCrushMetric.POSTS_CREATED,
                FeatureID.BTC_SOCIAL_POSTS_ACTIVE: LunarCrushMetric.POSTS_ACTIVE,
                FeatureID.BTC_SOCIAL_INTERACTIONS: LunarCrushMetric.INTERACTIONS,
                FeatureID.BTC_SOCIAL_CONTRIBUTORS_CREATED: LunarCrushMetric.CONTRIBUTORS_CREATED,
                FeatureID.BTC_SOCIAL_CONTRIBUTORS_ACTIVE: LunarCrushMetric.CONTRIBUTORS_ACTIVE,
                FeatureID.BTC_SOCIAL_SENTIMENT: LunarCrushMetric.SENTIMENT,
            },
        )

        test_feature_samples = test_source.get_feature_samples(
            _START_TIME_MS, _INTERVAL_MS, _SAMPLE_COUNT
        )

        expected_values = {
            # start: 1672531200
            0: {
                FeatureID.BTC_SOCIAL_POSTS_CREATED: 317,
                FeatureID.BTC_SOCIAL_POSTS_ACTIVE: 3957,
                FeatureID.BTC_SOCIAL_INTERACTIONS: 278276,
                FeatureID.BTC_SOCIAL_CONTRIBUTORS_CREATED: 255,
                FeatureID.BTC_SOCIAL_CONTRIBUTORS_ACTIVE: 2898,
                FeatureID.BTC_SOCIAL_SENTIMENT: 80,
            },
            # start: 1695027600
            6249: {
                FeatureID.BTC_SOCIAL_POSTS_CREATED: 1216,
                FeatureID.BTC_SOCIAL_POSTS_ACTIVE: 18201,
                FeatureID.BTC_SOCIAL_INTERACTIONS: 2200823,
                FeatureID.BTC_SOCIAL_CONTRIBUTORS_CREATED: 1005,
                FeatureID.BTC_SOCIAL_CONTRIBUTORS_ACTIVE: 10481,
                FeatureID.BTC_SOCIAL_SENTIMENT: 79,
            },
            # start: 1699527600
            -1: {
                FeatureID.BTC_SOCIAL_POSTS_CREATED: 676,
                FeatureID.BTC_SOCIAL_POSTS_ACTIVE: 41184,
                FeatureID.BTC_SOCIAL_INTERACTIONS: 4800962,
                FeatureID.BTC_SOCIAL_CONTRIBUTORS_CREATED: 556,
                FeatureID.BTC_SOCIAL_CONTRIBUTORS_ACTIVE: 16607,
                FeatureID.BTC_SOCIAL_SENTIMENT: 80,
            },
        }

        for index, samples in expected_values.items():
            for feature_id, expected_value in samples.items():
                test_value = test_feature_samples[feature_id][index]
                self.assertTrue(
                    math.isclose(
                        test_value,
                        expected_value,
                        rel_tol=1e-07,
                    ),
                    msg=f"index: {index} feature_id: {feature_id.name} "
                    f"test_value:{test_value} expected_value: {expected_value}",
                )


if __name__ == "__main__":
    unittest.main()
