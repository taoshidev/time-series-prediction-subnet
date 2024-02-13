# developer: taoshi-mbrown
# Copyright © 2024 Taoshi, LLC
from features import (
    FeatureAggregator,
    FeatureCollector,
    FeatureID,
    FeatureScaler,
    get_feature_ids,
)
from feature_sources import (
    CoinMetric,
    CoinMetricsAssetMetrics,
    CoinMetricsExchangeMetrics,
    CoinMetricsMarketCandles,
    CoinMetricsMarketFundingRates,
    CoinMetricsMarketMetrics,
    CoinMetricsMarketOpenInterest,
    BinanceKlineFeatureSource,
    BinanceKlineField,
    BybitKlineField,
    BybitKlineFeatureSource,
    CoinbaseKlineFeatureSource,
    CoinbaseKlineField,
    KrakenKlineFeatureSource,
    KrakenKlineField,
    TemporalFeatureSource,
)
from sklearn.preprocessing import MinMaxScaler
from statistics import fmean
from time_util import time_span_ms

SAMPLE_COUNT = 1500
INTERVAL_MS = time_span_ms(minutes=5)
PREDICTION_COUNT = 10
PREDICTION_LENGTH = 100

_VALIDATOR_AGGREGATOR_TIMEOUT = 10.0

_INCLUDE_EXCHANGE_KLINES = True
_INCLUDE_COIN_METRICS = False

historical_sources = []


if _INCLUDE_EXCHANGE_KLINES:
    binance_source = BinanceKlineFeatureSource(
        symbol="BTCUSDT",
        interval_ms=time_span_ms(minutes=5),
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
        interval_ms=time_span_ms(minutes=5),
        feature_mappings={
            FeatureID.BTC_USD_CLOSE: BybitKlineField.PRICE_CLOSE,
            FeatureID.BTC_USD_HIGH: BybitKlineField.PRICE_HIGH,
            FeatureID.BTC_USD_LOW: BybitKlineField.PRICE_LOW,
            FeatureID.BTC_USD_VOLUME: BybitKlineField.VOLUME,
        },
    )

    coinbase_source = CoinbaseKlineFeatureSource(
        symbol="BTC-USD",
        interval_ms=time_span_ms(minutes=5),
        feature_mappings={
            FeatureID.BTC_USD_CLOSE: CoinbaseKlineField.PRICE_CLOSE,
            FeatureID.BTC_USD_HIGH: CoinbaseKlineField.PRICE_HIGH,
            FeatureID.BTC_USD_LOW: CoinbaseKlineField.PRICE_LOW,
            FeatureID.BTC_USD_VOLUME: CoinbaseKlineField.VOLUME,
        },
    )

    kline_aggregator = FeatureAggregator(
        sources=[binance_source, bybit_source, coinbase_source],
        aggregation_map={
            FeatureID.BTC_USD_CLOSE: fmean,
            FeatureID.BTC_USD_HIGH: max,
            FeatureID.BTC_USD_LOW: min,
            FeatureID.BTC_USD_VOLUME: sum,
        },
    )

    historical_sources.append(kline_aggregator)


if _INCLUDE_COIN_METRICS:
    COIN_METRICS_ASSET = "btc"
    COIN_METRICS_EXCHANGE = "binance"
    COIN_METRICS_MARKET = "binance-btc-usdt-spot"
    COIN_METRICS_FUTURE_MARKET = "binance-BTCUSD_PERP-future"

    if not _INCLUDE_EXCHANGE_KLINES:
        btc_usd_candles_feature_source = CoinMetricsMarketCandles(
            kind=COIN_METRICS_MARKET,
            interval_ms=time_span_ms(minutes=5),
            feature_mappings={
                FeatureID.BTC_USD_CLOSE: CoinMetric.PRICE_CLOSE,
                FeatureID.BTC_USD_HIGH: CoinMetric.PRICE_HIGH,
                FeatureID.BTC_USD_LOW: CoinMetric.PRICE_LOW,
                FeatureID.BTC_USD_VOLUME: CoinMetric.VOLUME,
            },
        )
        historical_sources.append(btc_usd_candles_feature_source)

    btc_usd_volatility_feature_source = CoinMetricsAssetMetrics(
        kind=COIN_METRICS_ASSET,
        interval_ms=time_span_ms(minutes=10),
        feature_mappings={
            FeatureID.BTC_USD_VOLATILITY: CoinMetric.VOLATILITY_REALIZED_USD_ROLLING_24H,
        },
    )

    btc_address_count_feature_source = CoinMetricsAssetMetrics(
        kind=COIN_METRICS_ASSET,
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
        kind=COIN_METRICS_MARKET,
        interval_ms=time_span_ms(minutes=1),
        feature_mappings={
            FeatureID.BTC_USD_SPREAD: CoinMetric.LIQUIDITY_BID_ASK_SPREAD_PERCENT_1M,
        },
    )

    btc_usd_open_interest_feature_source = CoinMetricsMarketOpenInterest(
        kind=COIN_METRICS_FUTURE_MARKET,
        interval_ms=time_span_ms(minutes=1),
        feature_mappings={
            FeatureID.BTC_USD_FUTURES_OPEN_CONTRACTS: CoinMetric.CONTRACT_COUNT,
        },
    )

    btc_usd_liquidation_feature_source = CoinMetricsExchangeMetrics(
        kind=COIN_METRICS_EXCHANGE,
        interval_ms=time_span_ms(hours=1),
        feature_mappings={
            FeatureID.BTC_USD_FUTURES_LIQUIDATIONS_BUY: CoinMetric.LIQUIDATIONS_BUY_UNITS_5M,
            FeatureID.BTC_USD_FUTURES_LIQUIDATIONS_BUY_USD: CoinMetric.LIQUIDATIONS_BUY_USD_5M,
            FeatureID.BTC_USD_FUTURES_LIQUIDATIONS_SELL: CoinMetric.LIQUIDATIONS_SELL_UNITS_5M,
            FeatureID.BTC_USD_FUTURES_LIQUIDATIONS_SELL_USD: CoinMetric.LIQUIDATIONS_SELL_USD_5M,
        },
    )

    btc_usd_funding_rate_feature_source = CoinMetricsMarketFundingRates(
        kind=COIN_METRICS_FUTURE_MARKET,
        interval_ms=time_span_ms(hours=1),
        feature_mappings={
            FeatureID.BTC_USD_FUTURES_FUNDING_RATE: CoinMetric.RATE,
        },
    )

    historical_sources += [
        btc_usd_volatility_feature_source,
        btc_address_count_feature_source,
        btc_usd_spread_feature_source,
        btc_usd_open_interest_feature_source,
        btc_usd_liquidation_feature_source,
        btc_usd_funding_rate_feature_source,
    ]


temporal_feature_ids = [
    FeatureID.TIME_OF_DAY,
    FeatureID.TIME_OF_WEEK,
    FeatureID.TIME_OF_MONTH,
    FeatureID.TIME_OF_YEAR,
]
temporal_feature_source = TemporalFeatureSource(temporal_feature_ids)

historical_feature_ids = get_feature_ids(historical_sources)

# Features that can be quickly created at runtime and do not need to be stored
spontaneous_feature_sources = [temporal_feature_source]
spontaneous_feature_ids = get_feature_ids(spontaneous_feature_sources)

model_feature_sources = historical_sources + spontaneous_feature_sources
model_feature_ids = historical_feature_ids + spontaneous_feature_ids

legacy_model_feature_sources = historical_sources
legacy_model_feature_ids = historical_feature_ids

_default_scaler = MinMaxScaler(feature_range=(-1.0, 1.0), copy=False)
model_feature_scaler = FeatureScaler(
    default_scaler=_default_scaler,
    exclude_feature_ids=temporal_feature_ids,
    group_scaling_map={
        (
            FeatureID.BTC_USD_CLOSE,
            FeatureID.BTC_USD_HIGH,
            FeatureID.BTC_USD_LOW,
        ): _default_scaler,
    },
)

_legacy_default_scaler = MinMaxScaler(feature_range=(0.0, 1.0), copy=False)
legacy_model_feature_scaler = FeatureScaler(default_scaler=_default_scaler)

prediction_feature_ids = [FeatureID.BTC_USD_CLOSE]

validator_binance_source = BinanceKlineFeatureSource(
    symbol="BTCUSDT",
    interval_ms=time_span_ms(minutes=5),
    feature_mappings={
        FeatureID.BTC_USD_CLOSE: BinanceKlineField.PRICE_CLOSE,
    },
)

validator_bybit_source = BybitKlineFeatureSource(
    category="spot",
    symbol="BTCUSDT",
    interval_ms=time_span_ms(minutes=5),
    feature_mappings={
        FeatureID.BTC_USD_CLOSE: BybitKlineField.PRICE_CLOSE,
    },
)

validator_coinbase_source = CoinbaseKlineFeatureSource(
    symbol="BTC-USD",
    interval_ms=time_span_ms(minutes=5),
    feature_mappings={
        FeatureID.BTC_USD_CLOSE: CoinbaseKlineField.PRICE_CLOSE,
    },
)

validator_aggregator = FeatureAggregator(
    sources=[
        validator_binance_source,
        validator_bybit_source,
        validator_coinbase_source,
    ],
    aggregation_map={
        FeatureID.BTC_USD_CLOSE: fmean,
    },
    timeout=_VALIDATOR_AGGREGATOR_TIMEOUT,
)

validator_feature_source = FeatureCollector(
    sources=[validator_aggregator],
)
