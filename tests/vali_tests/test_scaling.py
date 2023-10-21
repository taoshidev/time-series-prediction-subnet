# developer: Taoshidev
# Copyright © 2023 Taoshi, LLC

import unittest
import random

import numpy as np

from vali_objects.scaling.scaling import Scaling


class TestScaling(unittest.TestCase):
    def test_count_decimal_places(self):
        self.assertEqual(3, Scaling.count_decimal_places(23.292))
        self.assertEqual(15, Scaling.count_decimal_places(23.2912391212947128491722))
        self.assertEqual(0, Scaling.count_decimal_places(23))

    def test_scale_unscale_values_exp(self):
        values = np.array([x for x in range(0, 1000)])
        avg, scaled_values = Scaling.scale_values_exp(values)

        self.assertEqual(avg, 499.5)

        self.assertLess(scaled_values.max(), 0.05)
        self.assertGreater(scaled_values.min(), -0.05)

        unscaled_values = Scaling.unscale_values_exp(avg, 0, scaled_values)
        self.assertTrue(np.array_equal(values, unscaled_values))

    def test_scale_unscale_values(self):
        values = np.array([x for x in range(0, 1000)])
        vmin, vmax, scaled_values = Scaling.scale_values(values)

        self.assertEqual(vmin, 0)
        self.assertEqual(vmax, 999)

        unscaled_values = Scaling.unscale_values(vmin, vmax, 0, scaled_values)
        self.assertTrue(np.array_equal(values, unscaled_values))

    def test_scale_unscale_values_w_range(self):
        values = np.array([x for x in range(500, 1000)])
        vmin, vmax, scaled_values = Scaling.scale_values(values)

        unscaled_additional_values = np.array([0.49, 0.5, .51, 0.52])
        expected_unscaled_values = np.array([ 250, 750, 1249, 1748])

        unscaled_values = Scaling.unscale_values(vmin, vmax, 0, unscaled_additional_values)
        self.assertTrue(np.array_equal(expected_unscaled_values, unscaled_values))

    def test_scale_unscale_values_against_an_indicator(self):
        def calculate_rsi(data, period=14):
            delta = [data[i] - data[i - 1] for i in range(1, len(data))]
            gains = [delta[i] if delta[i] > 0 else 0 for i in range(len(delta))]
            losses = [-delta[i] if delta[i] < 0 else 0 for i in range(len(delta))]

            avg_gain = sum(gains[:period]) / period
            avg_loss = sum(losses[:period]) / period

            rs_values = [avg_gain]

            for i in range(period, len(data) - 1):
                avg_gain = ((period - 1) * avg_gain + gains[i]) / period
                avg_loss = ((period - 1) * avg_loss + losses[i]) / period

                rs = avg_gain / avg_loss if avg_loss != 0 else 0
                rsi = 100 - (100 / (1 + rs))
                rs_values.append(rsi)

            return rs_values

        price_data = [random.uniform(45, 55) for _ in range(1000)]

        sv = Scaling.scale_values(price_data)

        unscaled_rsi_values = calculate_rsi(price_data)[1:99]
        scaled_rsi_values = calculate_rsi(sv[2])[1:99]

        for i in range(0, len(unscaled_rsi_values)):
            self.assertAlmostEqual(unscaled_rsi_values[i], scaled_rsi_values[i])

    def test_scale_data_structure(self):
        ds = []
        for i in range(0, 5):
            ds.append([x for x in range(0, 1000)])

        vmins, vmaxs, dps, ds_scale = Scaling.scale_ds_with_ts(ds)

        self.assertEqual(4, len(vmins))
        self.assertEqual(4, len(vmaxs))
        self.assertEqual(4, len(dps))
        self.assertEqual(5, len(ds_scale))

        for vmin in vmins:
            self.assertEqual(vmin, 0)
        for vmax in vmaxs:
            self.assertEqual(vmax, 999)
        for dp in dps:
            self.assertEqual(dp, 0)
        for dss in ds_scale[1:]:
            self.assertLess(dss.max(), 0.51)
            self.assertGreater(dss.min(), 0.49)
        self.assertEqual(ds[0][len(ds[0])-1], 999)

    def test_scale_with_min_max(self):
        values = np.array([x for x in range(500, 1000)])
        vmin, vmax, scaled_values = Scaling.scale_values(values)

        second_values_set = np.array([x for x in range(1000, 1010)])

        second_vmin, second_vmax, second_scaled_values = Scaling.scale_values(second_values_set,
                                                                              vmin = vmin,
                                                                              vmax = vmax)

        self.assertEqual(second_vmin, vmin)
        self.assertEqual(second_vmax, vmax)

        self.assertTrue(np.array_equal(np.array([0.5050200400801603, 0.5050400801603206, 0.505060120240481,
                                                 0.5050801603206413, 0.5051002004008016, 0.5051202404809619,
                                                 0.5051402805611223, 0.5051603206412826, 0.5051803607214429,
                                                 0.5052004008016032]), second_scaled_values))


if __name__ == '__main__':
    unittest.main()
