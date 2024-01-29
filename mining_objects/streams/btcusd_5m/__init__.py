# developer: Taoshidev
# Copyright © 2024 Taoshi, LLC
from features import FeatureID, FeatureSource, FeatureScaler
from feature_sources import (
    CoinMetric,
    CoinMetricsAssetMetrics,
    CoinMetricsExchangeMetrics,
    CoinMetricsMarketCandles,
    CoinMetricsMarketFundingRates,
    CoinMetricsMarketMetrics,
    CoinMetricsMarketOpenInterest,
    TemporalFeatureSource,
)
from sklearn.preprocessing import MinMaxScaler
from time_util import time_span_ms

INTERVAL_MS = time_span_ms(minutes=5)
ASSET = "btc"
EXCHANGE = "binance"
MARKET = "binance-btc-usdt-spot"
FUTURE_MARKET = "binance-BTCUSD_PERP-future"

btc_usd_candles_feature_source = CoinMetricsMarketCandles(
    kind=MARKET,
    interval_ms=time_span_ms(minutes=5),
    feature_mappings={
        FeatureID.BTC_USD_CLOSE: CoinMetric.PRICE_CLOSE,
        FeatureID.BTC_USD_HIGH: CoinMetric.PRICE_HIGH,
        FeatureID.BTC_USD_LOW: CoinMetric.PRICE_LOW,
        FeatureID.BTC_USD_VOLUME: CoinMetric.VOLUME,
    },
)

btc_usd_volatility_feature_source = CoinMetricsAssetMetrics(
    kind=ASSET,
    interval_ms=time_span_ms(minutes=10),
    feature_mappings={
        FeatureID.BTC_USD_VOLATILITY: CoinMetric.VOLATILITY_REALIZED_USD_ROLLING_24H,
    },
)

btc_address_count_feature_source = CoinMetricsAssetMetrics(
    kind=ASSET,
    interval_ms=time_span_ms(days=1),
    feature_mappings={
        FeatureID.BTC_HASH_RATE: CoinMetric.HASH_RATE,
        FeatureID.BTC_ADDR_COUNT_100K_USD: CoinMetric.ADDR_COUNT_100K_USD,
        FeatureID.BTC_ADDR_COUNT_1M_USD: CoinMetric.ADDR_COUNT_1M_USD,
        FeatureID.BTC_ADDR_COUNT_10M_USD: CoinMetric.ADDR_COUNT_10M_USD,
        FeatureID.BTC_MCTC: CoinMetric.MCTC,
        FeatureID.BTC_MCRC: CoinMetric.MCRC,
        FeatureID.BTC_MOMR: CoinMetric.MOMR,
        FeatureID.BTC_MARKET_CAP_USD: CoinMetric.MARKET_CAP_USD,
    },
)

btc_usd_spread_feature_source = CoinMetricsMarketMetrics(
    kind=MARKET,
    interval_ms=time_span_ms(minutes=1),
    feature_mappings={
        FeatureID.BTC_USD_SPREAD: CoinMetric.LIQUIDITY_BID_ASK_SPREAD_PERCENT_1M,
    },
)

btc_usd_open_interest_feature_source = CoinMetricsMarketOpenInterest(
    kind=FUTURE_MARKET,
    interval_ms=time_span_ms(minutes=1),
    feature_mappings={
        FeatureID.BTC_USD_FUTURES_OPEN_CONTRACTS: CoinMetric.CONTRACT_COUNT,
    },
)

btc_usd_liquidation_feature_source = CoinMetricsExchangeMetrics(
    kind=EXCHANGE,
    interval_ms=time_span_ms(hours=1),
    feature_mappings={
        FeatureID.BTC_USD_FUTURES_LIQUIDATIONS_BUY: CoinMetric.LIQUIDATIONS_BUY_UNITS_5M,
        FeatureID.BTC_USD_FUTURES_LIQUIDATIONS_BUY_USD: CoinMetric.LIQUIDATIONS_BUY_USD_5M,
        FeatureID.BTC_USD_FUTURES_LIQUIDATIONS_SELL: CoinMetric.LIQUIDATIONS_SELL_UNITS_5M,
        FeatureID.BTC_USD_FUTURES_LIQUIDATIONS_SELL_USD: CoinMetric.LIQUIDATIONS_SELL_USD_5M,
    },
)

btc_usd_funding_rate_feature_source = CoinMetricsMarketFundingRates(
    kind=FUTURE_MARKET,
    interval_ms=time_span_ms(hours=1),
    feature_mappings={
        FeatureID.BTC_USD_FUTURES_FUNDING_RATE: CoinMetric.RATE,
    },
)

temporal_feature_ids = TemporalFeatureSource.VALID_FEATURE_IDS
temporal_feature_source = TemporalFeatureSource(temporal_feature_ids)


def get_feature_ids(feature_sources: list[FeatureSource]) -> list[FeatureID]:
    results = []
    for feature_source in feature_sources:
        for feature_id in feature_source.feature_ids:
            results.append(feature_id)
    return results


historical_sources = [
    btc_usd_candles_feature_source,
    btc_usd_volatility_feature_source,
    btc_address_count_feature_source,
    btc_usd_spread_feature_source,
    btc_usd_open_interest_feature_source,
    btc_usd_liquidation_feature_source,
    btc_usd_funding_rate_feature_source,
]

historical_feature_ids = get_feature_ids(historical_sources)

# Features that can be easily created and do not need to be stored
spontaneous_feature_sources = [temporal_feature_source]
spontaneous_feature_ids = get_feature_ids(spontaneous_feature_sources)

model_feature_sources = historical_sources + spontaneous_feature_sources
model_feature_ids = historical_feature_ids + spontaneous_feature_ids

_default_scaler = MinMaxScaler(feature_range=(-1, 1), copy=False)
model_feature_scaler = FeatureScaler(
    default_scaler=_default_scaler, exclude_feature_ids=temporal_feature_ids
)
