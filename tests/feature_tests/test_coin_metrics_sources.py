# developer: taoshi-mbrown
# Copyright © 2024 Taoshi Inc
from features import FeatureID
from feature_sources import (
    CoinMetric,
    CoinMetricsAssetMetrics,
    CoinMetricsExchangeMetrics,
    CoinMetricsExchangeAssetMetrics,
    CoinMetricsMarketMetrics,
    CoinMetricsPairMetrics,
    CoinMetricsPairCandles,
    CoinMetricsInstitutionMetrics,
    CoinMetricsMarketTrades,
    CoinMetricsMarketOpenInterest,
    CoinMetricsMarketLiquidations,
    CoinMetricsMarketFundingRates,
    CoinMetricsMarketQuotes,
    CoinMetricsMarketCandles,
    CoinMetricsMarketContractPrices,
    CoinMetricsMarketImpliedVolatility,
    CoinMetricsMarketGreeks,
    CoinMetricsIndexCandles,
    CoinMetricsIndexLevels,
)
from time_util import datetime, time_span_ms
import unittest


class TestCoinMetricsFeatureSources(unittest.TestCase):
    def test_asset_metrics_feature_source(self):
        _START_TIME_MS = datetime.parse("2023-01-01 00:00:00").timestamp_ms()
        _INTERVAL_MS = time_span_ms(minutes=10)
        _SAMPLE_COUNT = 2000

        test_source = CoinMetricsAssetMetrics(
            kind="btc",
            source_interval_ms=_INTERVAL_MS,
            feature_mappings={
                FeatureID.BTC_USD_VOLATILITY: CoinMetric.VOLATILITY_REALIZED_USD_ROLLING_24H,
            },
        )

        test_feature_samples = test_source.get_feature_samples(
            _START_TIME_MS, _INTERVAL_MS, _SAMPLE_COUNT
        )

        expected_values = {
            0: {
                FeatureID.BTC_USD_VOLATILITY: 0.1137554,
            },
            1249: {
                FeatureID.BTC_USD_VOLATILITY: 0.2113918,
            },
            -1: {
                FeatureID.BTC_USD_VOLATILITY: 1.001736,
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

    def test_exchange_metrics_feature_source(self):
        _START_TIME_MS = datetime.parse("2023-01-01 00:00:00").timestamp_ms()
        _INTERVAL_MS = time_span_ms(hours=1)
        _SAMPLE_COUNT = 2000

        test_source = CoinMetricsExchangeMetrics(
            kind="binance",
            source_interval_ms=_INTERVAL_MS,
            feature_mappings={
                FeatureID.BTC_USD_FUTURES_LIQUIDATIONS_BUY: CoinMetric.LIQUIDATIONS_BUY_UNITS_5M,
                FeatureID.BTC_USD_FUTURES_LIQUIDATIONS_BUY_USD: CoinMetric.LIQUIDATIONS_BUY_USD_5M,
                FeatureID.BTC_USD_FUTURES_LIQUIDATIONS_SELL: CoinMetric.LIQUIDATIONS_SELL_UNITS_5M,
                FeatureID.BTC_USD_FUTURES_LIQUIDATIONS_SELL_USD: CoinMetric.LIQUIDATIONS_SELL_USD_5M,
            },
        )

        test_feature_samples = test_source.get_feature_samples(
            _START_TIME_MS, _INTERVAL_MS, _SAMPLE_COUNT
        )

        expected_values = {
            0: {
                FeatureID.BTC_USD_FUTURES_LIQUIDATIONS_BUY: 0.0,
                FeatureID.BTC_USD_FUTURES_LIQUIDATIONS_BUY_USD: 0.0,
                FeatureID.BTC_USD_FUTURES_LIQUIDATIONS_SELL: 443.527,
                FeatureID.BTC_USD_FUTURES_LIQUIDATIONS_SELL_USD: 269.94495,
            },
            1249: {
                FeatureID.BTC_USD_FUTURES_LIQUIDATIONS_BUY: 347535.1,
                FeatureID.BTC_USD_FUTURES_LIQUIDATIONS_BUY_USD: 158288.08,
                FeatureID.BTC_USD_FUTURES_LIQUIDATIONS_SELL: 324885.38,
                FeatureID.BTC_USD_FUTURES_LIQUIDATIONS_SELL_USD: 24944.207,
            },
            -1: {
                FeatureID.BTC_USD_FUTURES_LIQUIDATIONS_BUY: 7858.7,
                FeatureID.BTC_USD_FUTURES_LIQUIDATIONS_BUY_USD: 5586.3613,
                FeatureID.BTC_USD_FUTURES_LIQUIDATIONS_SELL: 1034368.94,
                FeatureID.BTC_USD_FUTURES_LIQUIDATIONS_SELL_USD: 7260.515,
            },
        }

        for index, samples in expected_values.items():
            for feature_id, expected_value in samples.items():
                test_value = float(test_feature_samples[feature_id][index])
                self.assertAlmostEqual(
                    test_value,
                    expected_value,
                    places=1,
                    msg=f"index: {index} feature_id: {feature_id}",
                )

    def test_market_metrics_feature_source(self):
        _START_TIME_MS = datetime.parse("2023-01-01 00:00:00").timestamp_ms()
        _INTERVAL_MS = time_span_ms(minutes=1)
        _SAMPLE_COUNT = 2000

        test_source = CoinMetricsMarketMetrics(
            kind="binance-btc-usdt-spot",
            source_interval_ms=_INTERVAL_MS,
            feature_mappings={
                FeatureID.BTC_USD_SPREAD: CoinMetric.LIQUIDITY_BID_ASK_SPREAD_PERCENT_1M,
            },
        )

        test_feature_samples = test_source.get_feature_samples(
            _START_TIME_MS, _INTERVAL_MS, _SAMPLE_COUNT
        )

        expected_values = {
            0: {
                FeatureID.BTC_USD_SPREAD: 0.0034465278,
            },
            1249: {
                FeatureID.BTC_USD_SPREAD: 0.0024788794,
            },
            -1: {
                FeatureID.BTC_USD_SPREAD: 0.001673903,
            },
        }

        for index, samples in expected_values.items():
            for feature_id, expected_value in samples.items():
                test_value = float(test_feature_samples[feature_id][index])
                self.assertAlmostEqual(
                    test_value,
                    expected_value,
                    places=4,
                    msg=f"index: {index} feature_id: {feature_id}",
                )

    def test_market_open_interest_feature_source(self):
        _START_TIME_MS = datetime.parse("2023-01-01 00:00:00").timestamp_ms()
        _INTERVAL_MS = time_span_ms(minutes=1)
        _SAMPLE_COUNT = 2000

        test_source = CoinMetricsMarketOpenInterest(
            kind="binance-BTCUSD_PERP-future",
            source_interval_ms=_INTERVAL_MS,
            feature_mappings={
                FeatureID.BTC_USD_FUTURES_OPEN_CONTRACTS: CoinMetric.CONTRACT_COUNT,
            },
        )

        test_feature_samples = test_source.get_feature_samples(
            _START_TIME_MS, _INTERVAL_MS, _SAMPLE_COUNT
        )

        expected_values = {
            0: {
                FeatureID.BTC_USD_FUTURES_OPEN_CONTRACTS: 3390107.0,
            },
            1249: {
                FeatureID.BTC_USD_FUTURES_OPEN_CONTRACTS: 3425057.0,
            },
            -1: {
                FeatureID.BTC_USD_FUTURES_OPEN_CONTRACTS: 3468191.0,
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

    def test_market_funding_rates_feature_source(self):
        _START_TIME_MS = datetime.parse("2023-01-01 00:00:00").timestamp_ms()
        _INTERVAL_MS = time_span_ms(hours=1)
        _SAMPLE_COUNT = 2000

        test_source = CoinMetricsMarketFundingRates(
            kind="binance-BTCUSD_PERP-future",
            source_interval_ms=_INTERVAL_MS,
            feature_mappings={
                FeatureID.BTC_USD_FUTURES_FUNDING_RATE: CoinMetric.RATE,
            },
        )

        test_feature_samples = test_source.get_feature_samples(
            _START_TIME_MS, _INTERVAL_MS, _SAMPLE_COUNT
        )

        expected_values = {
            0: {
                FeatureID.BTC_USD_FUTURES_FUNDING_RATE: -1.609e-05,
            },
            1249: {
                FeatureID.BTC_USD_FUTURES_FUNDING_RATE: 1e-04,
            },
            -1: {
                FeatureID.BTC_USD_FUTURES_FUNDING_RATE: -7.29e-05,
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

    def test_market_candles_feature_source(self):
        _START_TIME_MS = datetime.parse("2023-01-01 00:00:00").timestamp_ms()
        _INTERVAL_MS = time_span_ms(minutes=5)
        _SAMPLE_COUNT = 7500

        test_source = CoinMetricsMarketCandles(
            kind="binance-btc-usdt-spot",
            source_interval_ms=_INTERVAL_MS,
            feature_mappings={
                FeatureID.BTC_USD_CLOSE: CoinMetric.PRICE_CLOSE,
                FeatureID.BTC_USD_HIGH: CoinMetric.PRICE_HIGH,
                FeatureID.BTC_USD_LOW: CoinMetric.PRICE_LOW,
                FeatureID.BTC_USD_VOLUME: CoinMetric.VOLUME,
            },
        )

        test_feature_samples = test_source.get_feature_samples(
            _START_TIME_MS, _INTERVAL_MS, _SAMPLE_COUNT
        )

        # Expected values are the same as those used in
        # TestKlineFeatureSource.test_binance_kline_feature_source
        expected_values = {
            0: {
                FeatureID.BTC_USD_CLOSE: 16542.40000000,
                FeatureID.BTC_USD_HIGH: 16544.47000000,
                FeatureID.BTC_USD_LOW: 16535.05000000,
                FeatureID.BTC_USD_VOLUME: 227.06684000,
            },
            6249: {
                FeatureID.BTC_USD_CLOSE: 22831.88000000,
                FeatureID.BTC_USD_HIGH: 22831.88000000,
                FeatureID.BTC_USD_LOW: 22797.00000000,
                FeatureID.BTC_USD_VOLUME: 665.50900000,
            },
            -1: {
                FeatureID.BTC_USD_CLOSE: 22913.75000000,
                FeatureID.BTC_USD_HIGH: 22982.91000000,
                FeatureID.BTC_USD_LOW: 22897.02000000,
                FeatureID.BTC_USD_VOLUME: 1445.37762000,
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
