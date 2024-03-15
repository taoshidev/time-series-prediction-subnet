# developer: taoshi-mbrown
# Copyright © 2024 Taoshi Inc
from features import FeatureID
from feature_sources import (
    YahooFinanceKlineField,
    YahooFinanceKlineFeatureSource,
)
from time_util import datetime, time_span_ms, previous_interval_ms
import unittest


class TestYahooFinanceKlineFeatureSource(unittest.TestCase):
    def test_yahoo_finance_feature_source_btc_usd_5m(self):
        _INTERVAL_MS = time_span_ms(minutes=5)
        _SKIP_COUNT = 10
        _SAMPLE_COUNT = 100
        _START_TIME_MS = previous_interval_ms(
            datetime.now().timestamp_ms(), _INTERVAL_MS
        ) - ((_SAMPLE_COUNT + _SKIP_COUNT) * _INTERVAL_MS)
        _BTC_USD_LOW_MIN = 40000
        _BTC_USD_HIGH_MAX = 120000

        test_source = YahooFinanceKlineFeatureSource(
            ticker="BTC-USD",
            source_interval_ms=_INTERVAL_MS,
            feature_mappings={
                FeatureID.BTC_USD_CLOSE: YahooFinanceKlineField.PRICE_CLOSE,
                FeatureID.BTC_USD_HIGH: YahooFinanceKlineField.PRICE_HIGH,
                FeatureID.BTC_USD_LOW: YahooFinanceKlineField.PRICE_LOW,
                FeatureID.BTC_USD_VOLUME: YahooFinanceKlineField.VOLUME,
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

        test_feature_samples = test_source.get_feature_samples(
            _START_TIME_MS, _INTERVAL_MS, _SAMPLE_COUNT
        )

        i = 0
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

    def test_yahoo_finance_feature_source_spx_usd_5m(self):
        _INTERVAL_MS = time_span_ms(minutes=5)
        _SKIP_COUNT = 10
        _SAMPLE_COUNT = 100
        # TODO: Adjust for market opening
        _START_DATE = datetime.parse("2024-03-12T15:00:00-04:00")
        _START_TIME_MS = previous_interval_ms(
            _START_DATE.timestamp_ms(), _INTERVAL_MS
        ) - ((_SAMPLE_COUNT + _SKIP_COUNT) * _INTERVAL_MS)
        _SPX_USD_LOW_MIN = 3000
        _SPX_USD_HIGH_MAX = 7000

        test_source = YahooFinanceKlineFeatureSource(
            ticker="^GSPC",
            source_interval_ms=_INTERVAL_MS,
            feature_mappings={
                FeatureID.SPX_USD_CLOSE: YahooFinanceKlineField.PRICE_CLOSE,
                FeatureID.SPX_USD_HIGH: YahooFinanceKlineField.PRICE_HIGH,
                FeatureID.SPX_USD_LOW: YahooFinanceKlineField.PRICE_LOW,
                FeatureID.SPX_USD_VOLUME: YahooFinanceKlineField.VOLUME,
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
            close = test_feature_samples[FeatureID.SPX_USD_CLOSE][i]
            high = test_feature_samples[FeatureID.SPX_USD_HIGH][i]
            low = test_feature_samples[FeatureID.SPX_USD_LOW][i]
            volume = test_feature_samples[FeatureID.SPX_USD_VOLUME][i]
            assert close != last_close
            assert high != last_high
            assert high >= close
            assert high < _SPX_USD_HIGH_MAX
            assert low != last_low
            assert low <= close
            assert low > _SPX_USD_LOW_MIN
            assert volume != last_volume

        test_feature_samples = test_source.get_feature_samples(
            _START_TIME_MS, _INTERVAL_MS, _SAMPLE_COUNT
        )

        i = 0
        close = test_feature_samples[FeatureID.SPX_USD_CLOSE][i]
        high = test_feature_samples[FeatureID.SPX_USD_HIGH][i]
        low = test_feature_samples[FeatureID.SPX_USD_LOW][i]
        volume = test_feature_samples[FeatureID.SPX_USD_VOLUME][i]
        assert close != last_close
        assert high != last_high
        assert high >= close
        assert high < _SPX_USD_HIGH_MAX
        assert low != last_low
        assert low <= close
        assert low > _SPX_USD_LOW_MIN
        assert volume != last_volume

    def test_yahoo_finance_feature_source_eur_usd_5m(self):
        _INTERVAL_MS = time_span_ms(minutes=5)
        _SKIP_COUNT = 10
        _SAMPLE_COUNT = 100
        _START_TIME_MS = previous_interval_ms(
            datetime.now().timestamp_ms(), _INTERVAL_MS
        ) - ((_SAMPLE_COUNT + _SKIP_COUNT) * _INTERVAL_MS)
        _EUR_USD_LOW_MIN = 1.0
        _EUR_USD_HIGH_MAX = 1.3

        test_source = YahooFinanceKlineFeatureSource(
            ticker="EURUSD=X",
            source_interval_ms=_INTERVAL_MS,
            feature_mappings={
                FeatureID.EUR_USD_CLOSE: YahooFinanceKlineField.PRICE_CLOSE,
                FeatureID.EUR_USD_HIGH: YahooFinanceKlineField.PRICE_HIGH,
                FeatureID.EUR_USD_LOW: YahooFinanceKlineField.PRICE_LOW,
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
            close = test_feature_samples[FeatureID.EUR_USD_CLOSE][i]
            high = test_feature_samples[FeatureID.EUR_USD_HIGH][i]
            low = test_feature_samples[FeatureID.EUR_USD_LOW][i]
            assert close != last_close
            assert high != last_high
            assert high >= close
            assert high < _EUR_USD_HIGH_MAX
            assert low != last_low
            assert low <= close
            assert low > _EUR_USD_LOW_MIN

        i = 0
        close = test_feature_samples[FeatureID.EUR_USD_CLOSE][i]
        high = test_feature_samples[FeatureID.EUR_USD_HIGH][i]
        low = test_feature_samples[FeatureID.EUR_USD_LOW][i]
        assert close != last_close
        assert high != last_high
        assert high >= close
        assert high < _EUR_USD_HIGH_MAX
        assert low != last_low
        assert low <= close
        assert low > _EUR_USD_LOW_MIN


if __name__ == "__main__":
    unittest.main()
