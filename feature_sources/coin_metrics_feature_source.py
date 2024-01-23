# developer: Taoshidev
# Copyright © 2024 Taoshi, LLC
import os
from abc import abstractmethod
from coinmetrics.api_client import CoinMetricsClient, DataCollection
from coinmetrics.constants import PagingFrom
from enum import Enum
from features import FeatureCompaction, FeatureID, FeatureSource
import math
import numpy as np
from numpy import ndarray
from time_util import datetime, time_span_ms, parse_time_interval_ms
import statistics


class CoinMetric(str, Enum):
    TIME = "time"
    DATABASE_TIME = "database_time"

    AMOUNT = "amount"

    SIDE = "side"

    PERIOD = "period"
    INTERVAL = "interval"

    PRICE = "price"
    PRICE_OPEN = "price_open"
    PRICE_CLOSE = "price_close"
    PRICE_HIGH = "price_high"
    PRICE_LOW = "price_low"
    VWAP = "vwap"
    VOLUME = "volume"
    VOLUME_USD = "candle_usd_volume"
    TRADES_COUNT = "candle_trades_count"

    ASK_PRICE = "ask_price"
    ASK_SIZE = "ask_size"
    BID_PRICE = "bid_price"
    BID_SIZE = "bid_size"

    LIQUIDITY_BID_ASK_SPREAD_PERCENT_1M = "liquidity_bid_ask_spread_percent_1m"

    VOLATILITY_REALIZED_USD_ROLLING_24H = "volatility_realized_usd_rolling_24h"

    IV_BID = "iv_bid"
    IV_ASK = "iv_ask"
    IV_MARK = "iv_mark"

    CONTRACT_COUNT = "contract_count"
    VALUE_USD = "value_usd"

    LIQUIDATIONS_BUY_UNITS_5M = "liquidations_reported_future_buy_units_5m"
    LIQUIDATIONS_BUY_USD_5M = "liquidations_reported_future_buy_units_5m"
    LIQUIDATIONS_SELL_UNITS_5M = "liquidations_reported_future_buy_units_5m"
    LIQUIDATIONS_SELL_USD_5M = "liquidations_reported_future_buy_units_5m"

    RATE = "rate"

    HASH_RATE = "HashRate"

    ADDR_COUNT_100K_USD = "AdrBalUSD100KCnt"
    ADDR_COUNT_1M_USD = "AdrBalUSD1MCnt"
    ADDR_COUNT_10M_USD = "AdrBalUSD10MCnt"

    MCTC = "MCTC"
    MCRC = "MCRC"
    MOMR = "MOMR"

    MARKET_CAP_USD = "CapMrktCurUSD"

    FLOW_MINER_NET_1_HOP_ALL_USD = "FlowMinerNet1HopAllUSD"


class CoinMetricsFeatureSource(FeatureSource):
    SOURCE_NAME = "CoinMetrics"

    _PAGE_SIZE = 10000

    _FREQUENCIES = {
        time_span_ms(seconds=1): "1s",
        time_span_ms(minutes=1): "1m",
        time_span_ms(minutes=5): "5m",
        time_span_ms(minutes=10): "10m",
        time_span_ms(minutes=15): "15m",
        time_span_ms(minutes=30): "30m",
        time_span_ms(hours=1): "1h",
        time_span_ms(hours=4): "4h",
        time_span_ms(days=1): "1d",
    }

    # Default is using last sample, so on only include other types
    _METRIC_COMPACTIONS = {
        CoinMetric.PRICE_OPEN: FeatureCompaction.FIRST,
        CoinMetric.PRICE_HIGH: FeatureCompaction.MAX,
        CoinMetric.PRICE_LOW: FeatureCompaction.MIN,
        CoinMetric.VWAP: FeatureCompaction.MEAN,
        CoinMetric.VOLUME: FeatureCompaction.SUM,
        CoinMetric.VOLUME_USD: FeatureCompaction.SUM,
        CoinMetric.TRADES_COUNT: FeatureCompaction.SUM,
    }

    def __init__(
        self,
        kind: str,
        interval_ms: int,
        feature_mappings: dict[FeatureID, CoinMetric],
        feature_dtypes: list[np.dtype] = None,
        default_dtype: np.dtype = np.dtype(np.float32),
        allow_empty_response=True,
        api_key: str = None,
        **kwargs,
    ):
        feature_ids = list(feature_mappings.keys())
        self.VALID_FEATURE_IDS = feature_ids
        super().__init__(feature_ids, feature_dtypes, default_dtype)

        if api_key is None:
            api_key = os.environ.get("CM_API_KEY")

        self._kind = kind
        self._interval_ms = interval_ms
        self._frequency = self._FREQUENCIES[interval_ms]
        self._feature_mappings = feature_mappings
        self._metrics = list(feature_mappings.values())
        self._convert_metrics = [CoinMetric.TIME, *self._metrics]
        self._allow_empty_response = allow_empty_response
        self._client = CoinMetricsClient(api_key=api_key, **kwargs)

    # noinspection PyMethodMayBeStatic
    def _convert_metric(self, metric: str, value):
        match metric:
            case CoinMetric.SIDE:
                if value == "buy":
                    value = 1
                elif value == "sell":
                    value = -1
                else:
                    value = 0
            case CoinMetric.PERIOD | CoinMetric.INTERVAL:
                value = parse_time_interval_ms(value)
            case _ if "time" in metric:
                value = datetime.parse(value).timestamp_ms()
            case _:
                value = float(value)
        return value

    def _convert_sample(self, sample: dict) -> dict:
        results = {}
        for metric in self._convert_metrics:
            results[metric] = self._convert_metric(metric, sample[metric])
        return results

    def _convert_samples(self, data_rows: list[dict]) -> list[dict]:
        return [self._convert_sample(row) for row in data_rows]

    def _get_empty_converted_samples(self, start_time_ms: int) -> list[dict]:
        empty_sample = {metric: 0 for metric in self._metrics}
        empty_sample[CoinMetric.TIME] = start_time_ms
        return [empty_sample]

    def _compact_samples(self, samples: list[dict]) -> dict:
        result = samples[-1].copy()
        for metric in self._metrics:
            compaction = self._METRIC_COMPACTIONS.get(metric, FeatureCompaction.LAST)
            if compaction == FeatureCompaction.LAST:
                continue
            elif compaction == FeatureCompaction.FIRST:
                result[metric] = samples[0][metric]
            else:
                values = [sample[metric] for sample in samples]
                match compaction:
                    case FeatureCompaction.MIN:
                        metric_result = min(values)
                    case FeatureCompaction.MAX:
                        metric_result = max(values)
                    case FeatureCompaction.MEAN:
                        metric_result = statistics.mean(values)
                    case FeatureCompaction.MEDIAN:
                        metric_result = statistics.median(values)
                    case FeatureCompaction.MODE:
                        metric_result = statistics.mode(values)
                    case _:
                        metric_result = math.fsum(values)
                result[metric] = metric_result
        return result

    # noinspection PyMethodMayBeStatic
    def _compact_samples_trades(self, samples: list[dict]) -> dict:
        result = samples[-1].copy()
        total_amount = 0
        side_amount = 0
        price_amount = 0
        for sample in samples:
            amount = sample[CoinMetric.AMOUNT]
            price = sample[CoinMetric.PRICE]
            side = sample[CoinMetric.SIDE]
            total_amount += amount
            side_amount += side * amount
            price_amount += price * amount
        result[CoinMetric.AMOUNT] = side_amount
        result[CoinMetric.PRICE] = price_amount / total_amount
        result[CoinMetric.SIDE] = side_amount / total_amount
        return result

    @abstractmethod
    def _query(self, start_time: str, end_time: str) -> DataCollection:
        pass

    def get_feature_samples(
        self,
        start_time_ms: int,
        interval_ms: int,
        sample_count: int,
    ) -> dict[FeatureID, ndarray]:
        query_start_time_ms = start_time_ms
        if interval_ms < self._interval_ms:
            query_start_time_ms -= self._interval_ms

        # Times must be preformatted because Coin Metrics rejects times with
        # the ISO timezone suffix for UTC ("+00:00") and their Python
        # library doesn't format it for their preference
        start_time = datetime.fromtimestamp_ms(query_start_time_ms)
        # TODO: Subtract 1 from sample_count
        end_time_ms = start_time_ms + (interval_ms * sample_count)
        end_time = datetime.fromtimestamp_ms(end_time_ms)
        start_time_string = start_time.to_iso8601_string()
        end_time_string = end_time.to_iso8601_string()

        response = self._query(start_time_string, end_time_string)
        data_rows = response.to_list()

        row_count = len(data_rows)
        if row_count == 0:
            if self._allow_empty_response:
                converted_samples = self._get_empty_converted_samples(start_time_ms)
                row_count = len(converted_samples)
            else:
                raise Exception()  # TODO: Implement
        else:
            converted_samples = self._convert_samples(data_rows)

        feature_samples = self._create_feature_samples(sample_count)

        sample_time_ms = start_time_ms
        interval_rows = []
        row_index = 0
        last_row_index = row_count - 1
        compact_samples = self._compact_samples
        for sample_index in range(sample_count):
            while True:
                row = converted_samples[row_index]
                row_time_ms = row[CoinMetric.TIME]
                if row_time_ms > sample_time_ms:
                    break
                interval_rows.append(row)
                if row_index == last_row_index:
                    break
                row_index += 1

            if len(interval_rows) > 1:
                row = compact_samples(interval_rows)

            for feature_index, metric in enumerate(self._metrics):
                feature_samples[feature_index][sample_index] = row[metric]

            interval_rows.clear()
            sample_time_ms += interval_ms

        results = {
            self.feature_ids[feature_index]: feature_samples[feature_index]
            for feature_index in range(self.feature_count)
        }

        return results


class CoinMetricsAssetMetrics(CoinMetricsFeatureSource):
    def _query(self, start_time: str, end_time: str) -> DataCollection:
        return self._client.get_asset_metrics(
            assets=self._kind,
            metrics=self._metrics,
            frequency=self._frequency,
            start_time=start_time,
            end_time=end_time,
            end_inclusive=False,
            paging_from=PagingFrom.START,
            page_size=self._PAGE_SIZE,
        )


class CoinMetricsExchangeMetrics(CoinMetricsFeatureSource):
    def _query(self, start_time: str, end_time: str) -> DataCollection:
        return self._client.get_exchange_metrics(
            exchanges=self._kind,
            metrics=self._metrics,
            frequency=self._frequency,
            start_time=start_time,
            end_time=end_time,
            end_inclusive=False,
            paging_from=PagingFrom.START,
            page_size=self._PAGE_SIZE,
        )


class CoinMetricsExchangeAssetMetrics(CoinMetricsFeatureSource):
    def _query(self, start_time: str, end_time: str) -> DataCollection:
        return self._client.get_exchange_asset_metrics(
            exchange_assets=self._kind,
            metrics=self._metrics,
            frequency=self._frequency,
            start_time=start_time,
            end_time=end_time,
            end_inclusive=False,
            paging_from=PagingFrom.START,
            page_size=self._PAGE_SIZE,
        )


class CoinMetricsMarketMetrics(CoinMetricsFeatureSource):
    def _query(self, start_time: str, end_time: str) -> DataCollection:
        return self._client.get_market_metrics(
            markets=self._kind,
            metrics=self._metrics,
            frequency=self._frequency,
            start_time=start_time,
            end_time=end_time,
            end_inclusive=False,
            paging_from=PagingFrom.START,
            page_size=self._PAGE_SIZE,
        )


class CoinMetricsPairMetrics(CoinMetricsFeatureSource):
    def _query(self, start_time: str, end_time: str) -> DataCollection:
        return self._client.get_pair_metrics(
            pairs=self._kind,
            metrics=self._metrics,
            frequency=self._frequency,
            start_time=start_time,
            end_time=end_time,
            end_inclusive=False,
            paging_from=PagingFrom.START,
            page_size=self._PAGE_SIZE,
        )


class CoinMetricsPairCandles(CoinMetricsFeatureSource):
    def _query(self, start_time: str, end_time: str) -> DataCollection:
        return self._client.get_pair_candles(
            pairs=self._kind,
            frequency=self._frequency,
            start_time=start_time,
            end_time=end_time,
            end_inclusive=False,
            paging_from=PagingFrom.START,
            page_size=self._PAGE_SIZE,
        )


class CoinMetricsInstitutionMetrics(CoinMetricsFeatureSource):
    def _query(self, start_time: str, end_time: str) -> DataCollection:
        return self._client.get_institution_metrics(
            institutions=self._kind,
            metrics=self._metrics,
            frequency=self._frequency,
            start_time=start_time,
            end_time=end_time,
            end_inclusive=False,
            paging_from=PagingFrom.START,
            page_size=self._PAGE_SIZE,
        )


class CoinMetricsMarketTrades(CoinMetricsFeatureSource):
    def _compact_samples(self, samples: list[dict]) -> dict:
        return self._compact_samples_trades(samples)

    def _query(self, start_time: str, end_time: str) -> DataCollection:
        return self._client.get_market_trades(
            markets=self._kind,
            start_time=start_time,
            end_time=end_time,
            end_inclusive=False,
            paging_from=PagingFrom.START,
            page_size=self._PAGE_SIZE,
        )


class CoinMetricsMarketOpenInterest(CoinMetricsFeatureSource):
    def _query(self, start_time: str, end_time: str) -> DataCollection:
        return self._client.get_market_open_interest(
            markets=self._kind,
            granularity=self._frequency,
            start_time=start_time,
            end_time=end_time,
            end_inclusive=False,
            paging_from=PagingFrom.START,
            page_size=self._PAGE_SIZE,
        )


class CoinMetricsMarketLiquidations(CoinMetricsFeatureSource):
    def _compact_samples(self, samples: list[dict]) -> dict:
        return self._compact_samples_trades(samples)

    def _query(self, start_time: str, end_time: str) -> DataCollection:
        return self._client.get_market_liquidations(
            markets=self._kind,
            start_time=start_time,
            end_time=end_time,
            end_inclusive=False,
            paging_from=PagingFrom.START,
            page_size=self._PAGE_SIZE,
        )


class CoinMetricsMarketFundingRates(CoinMetricsFeatureSource):
    def _query(self, start_time: str, end_time: str) -> DataCollection:
        return self._client.get_market_funding_rates(
            markets=self._kind,
            start_time=start_time,
            end_time=end_time,
            end_inclusive=False,
            paging_from=PagingFrom.START,
            page_size=self._PAGE_SIZE,
        )


class CoinMetricsMarketQuotes(CoinMetricsFeatureSource):
    def _compact_samples(self, samples: list[dict]) -> dict:
        result = samples[-1]
        ask_price = 0
        ask_size = 0
        bid_price = 0
        bid_size = 0
        for sample in samples:
            ask_price += sample[CoinMetric.ASK_PRICE]
            ask_size += sample[CoinMetric.ASK_SIZE]
            bid_price += sample[CoinMetric.BID_PRICE]
            bid_size += sample[CoinMetric.BID_SIZE]
        result[CoinMetric.ASK_PRICE] = ask_price / ask_size
        result[CoinMetric.ASK_SIZE] = ask_size
        result[CoinMetric.BID_PRICE] = bid_price / bid_size
        result[CoinMetric.BID_SIZE] = bid_size
        return result

    def _query(self, start_time: str, end_time: str) -> DataCollection:
        return self._client.get_market_quotes(
            markets=self._kind,
            granularity=self._frequency,
            start_time=start_time,
            end_time=end_time,
            end_inclusive=False,
            paging_from=PagingFrom.START,
            page_size=self._PAGE_SIZE,
        )


class CoinMetricsMarketCandles(CoinMetricsFeatureSource):
    def _query(self, start_time: str, end_time: str) -> DataCollection:
        return self._client.get_market_candles(
            markets=self._kind,
            frequency=self._frequency,
            start_time=start_time,
            end_time=end_time,
            end_inclusive=False,
            paging_from=PagingFrom.START,
            page_size=self._PAGE_SIZE,
        )


class CoinMetricsMarketContractPrices(CoinMetricsFeatureSource):
    def _query(self, start_time: str, end_time: str) -> DataCollection:
        return self._client.get_market_contract_prices(
            markets=self._kind,
            granularity=self._frequency,
            start_time=start_time,
            end_time=end_time,
            end_inclusive=False,
            paging_from=PagingFrom.START,
            page_size=self._PAGE_SIZE,
        )


class CoinMetricsMarketImpliedVolatility(CoinMetricsFeatureSource):
    def _query(self, start_time: str, end_time: str) -> DataCollection:
        return self._client.get_market_implied_volatility(
            markets=self._kind,
            granularity=self._frequency,
            start_time=start_time,
            end_time=end_time,
            end_inclusive=False,
            paging_from=PagingFrom.START,
            page_size=self._PAGE_SIZE,
        )


class CoinMetricsMarketGreeks(CoinMetricsFeatureSource):
    def _query(self, start_time: str, end_time: str) -> DataCollection:
        return self._client.get_market_greeks(
            markets=self._kind,
            granularity=self._frequency,
            start_time=start_time,
            end_time=end_time,
            end_inclusive=False,
            paging_from=PagingFrom.START,
            page_size=self._PAGE_SIZE,
        )


class CoinMetricsIndexCandles(CoinMetricsFeatureSource):
    def _query(self, start_time: str, end_time: str) -> DataCollection:
        return self._client.get_index_candles(
            indexes=self._kind,
            frequency=self._frequency,
            start_time=start_time,
            end_time=end_time,
            end_inclusive=False,
            paging_from=PagingFrom.START,
            page_size=self._PAGE_SIZE,
        )


class CoinMetricsIndexLevels(CoinMetricsFeatureSource):
    def _query(self, start_time: str, end_time: str) -> DataCollection:
        return self._client.get_index_levels(
            indexes=self._kind,
            frequency=self._frequency,
            start_time=start_time,
            end_time=end_time,
            end_inclusive=False,
            paging_from=PagingFrom.START,
            page_size=self._PAGE_SIZE,
        )
