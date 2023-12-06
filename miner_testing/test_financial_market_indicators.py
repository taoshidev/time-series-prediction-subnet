import numpy as np
from matplotlib import pyplot as plt

from mining_objects.financial_market_indicators import FinancialMarketIndicators
from mining_objects.mining_utils import MiningUtils
from vali_tests.tests.base_objects.test_base import TestBase
from vali_tests.tests.test_exchange_data import TestExchangeData


class TestFinancialMarketIndicators(TestBase):

    def test_rsi(self):
        ds = TestExchangeData().generate_test_data(rows=1000)

        rsi = FinancialMarketIndicators.calculate_rsi(ds)
        sds = MiningUtils.scale_ds(np.array(ds), 0, 100)

        MiningUtils.plt_features({"rsi": rsi, "closes": sds[1]}, range(len(sds[0])))

    def test_macd(self):
        ds = TestExchangeData().generate_test_data(rows=1000)

        macd, signal = FinancialMarketIndicators.calculate_macd(ds)
        sds = MiningUtils.scale_ds(np.array(ds), 0, 100)

        MiningUtils.plt_features({"macd": macd, "signal": signal, "closes": sds[1]}, range(len(sds[0])))

    def test_bollinger_bands(self):
        ds = TestExchangeData().generate_test_data(rows=1000)

        mid, upp, low = FinancialMarketIndicators.calculate_bollinger_bands(ds)
        MiningUtils.plt_features({"mid": mid, "upp": upp, "low": low, "closes": ds[1]}, range(len(ds[0])))

    def test_ema(self):
        ds = TestExchangeData().generate_test_data(rows=1000)

        ema = FinancialMarketIndicators.calculate_ema(ds, window=100)
        MiningUtils.plt_features({"ema": ema, "closes": ds[1]}, range(len(ds[0])))

    def test_sma(self):
        ds = TestExchangeData().generate_test_data(rows=1000)

        ma = FinancialMarketIndicators.calculate_sma(ds, window=100)
        MiningUtils.plt_features({"ema": ma, "closes": ds[1]}, range(len(ds[0])))

    def test_stochastic_rsi(self):
        ds = TestExchangeData().generate_test_data(rows=1000)
        sds = MiningUtils.scale_ds(np.array(ds), 0, 100)

        srsi_k, srsi_d = FinancialMarketIndicators.calculate_stochastic_rsi(ds)
        MiningUtils.plt_features({"srsi_k": srsi_k, "srsi_d": srsi_d, "closes": sds[1]}, range(len(sds[0])))