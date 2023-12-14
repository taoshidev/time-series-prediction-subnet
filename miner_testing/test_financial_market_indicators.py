import numpy as np
from matplotlib import pyplot as plt

from mining_objects.financial_market_indicators import FinancialMarketIndicators
from mining_objects.mining_utils import MiningUtils
from vali_tests.tests.base_objects.test_base import TestBase
from vali_tests.tests.test_exchange_data import TestExchangeData


class TestFinancialMarketIndicators(TestBase):
    def get_ds_close(self, ds):
        return ds[1]

    def get_ds_volume(self, ds):
        return ds[4]

    def test_rsi(self):
        ds = TestExchangeData().generate_test_data(rows=1000)

        rsi = FinancialMarketIndicators.calculate_rsi(self.get_ds_close(ds))
        sds = MiningUtils.scale_values(np.array(ds), 0, 100)

        MiningUtils.plt_features({"rsi": rsi, "closes": sds[1]}, range(len(sds[0])))

    def test_macd(self):
        ds = TestExchangeData().generate_test_data(rows=1000)

        macd, signal = FinancialMarketIndicators.calculate_macd(self.get_ds_close(ds))
        sds = MiningUtils.scale_values(np.array(ds), 0, 100)

        MiningUtils.plt_features({"macd": macd, "signal": signal, "closes": sds[1]}, range(len(sds[0])))

    def test_bollinger_bands(self):
        ds = TestExchangeData().generate_test_data(rows=1000)

        mid, upp, low = FinancialMarketIndicators.calculate_bollinger_bands(self.get_ds_close(ds))
        MiningUtils.plt_features({"mid": mid, "upp": upp, "low": low, "closes": ds[1]}, range(len(ds[0])))

    def test_ema(self):
        ds = TestExchangeData().generate_test_data(rows=1000)

        ema = FinancialMarketIndicators.calculate_ema(self.get_ds_close(ds), window=100)
        MiningUtils.plt_features({"ema": ema, "closes": ds[1]}, range(len(ds[0])))

    def test_sma(self):
        ds = TestExchangeData().generate_test_data(rows=1000)

        ma = FinancialMarketIndicators.calculate_sma(self.get_ds_close(ds), window=100)
        MiningUtils.plt_features({"ema": ma, "closes": ds[1]}, range(len(ds[0])))

    def test_stochastic_rsi(self):
        ds = TestExchangeData().generate_test_data(rows=1000)
        sds = MiningUtils.scale_values(np.array(ds), 0, 100)

        srsi_k, srsi_d = FinancialMarketIndicators.calculate_stochastic_rsi(self.get_ds_close(ds))
        MiningUtils.plt_features({"srsi_k": srsi_k, "srsi_d": srsi_d, "closes": sds[1]}, range(len(sds[0])))

    def test_calculate_vwap(self):
        ds = TestExchangeData().generate_test_data(rows=1000)

        vwap = FinancialMarketIndicators.calculate_vwap(self.get_ds_close(ds), self.get_ds_volume(ds))
        self.assertEqual(vwap, 34793.54620066298)

    def test_calculate_vwap_interval(self):
        ds = TestExchangeData().generate_test_data(rows=1000)

        vwaps = FinancialMarketIndicators.calculate_vwap_interval(self.get_ds_close(ds), self.get_ds_volume(ds), 25, True)
        MiningUtils.plt_features({"vwaps": vwaps, "closes": ds[1]}, range(len(ds[0])))

    def test_calculate_vrvp(self):
        ds = TestExchangeData().generate_test_data(rows=1000)
        visible_range = 100
        vrvp = FinancialMarketIndicators.calculate_vrvp(self.get_ds_close(ds), self.get_ds_volume(ds), visible_range)
        s_vrvp = MiningUtils.scale_values(np.array(vrvp), 0, 100)

        plt.plot(s_vrvp.T.tolist(), [i for i in range(len(ds[0]))])

        plt.show()

    def test_calculate_sum_vrvp_per_close(self):
        ds = TestExchangeData().generate_test_data(rows=1000)
        visible_range = 100
        vrvp = FinancialMarketIndicators.calculate_sum_vrvp_per_close(self.get_ds_close(ds), self.get_ds_volume(ds), visible_range)
        vrvp = dict(sorted(vrvp.items()))

        closes = []
        vrvp_values = []

        for key, value in vrvp.items():
            closes.append(key)
            vrvp_values.append(value)

        plt.plot(vrvp_values, closes)

        plt.show()

    def test_calculate_sum_vrvp_per_grouping_size(self):
        ds = TestExchangeData().generate_test_data(rows=1000)
        visible_range = 100
        vrvp = FinancialMarketIndicators.calculate_sum_vrvp_per_grouping_size(self.get_ds_close(ds), self.get_ds_volume(ds), 10, visible_range)
        vrvp = dict(sorted(vrvp.items()))

        closes = []
        vrvp_values = []

        for key, value in vrvp.items():
            closes.append(key)
            vrvp_values.append(value)

        s_vrvp_values = MiningUtils.scale_values(np.array(vrvp_values), 0, len(ds[0]))

        plt.plot(s_vrvp_values.T.tolist(), closes, label="vrvp")
        plt.plot(range(len(ds[0])), ds[1], label="closes")

        plt.title("VRVP on Closing Data")

        plt.show()