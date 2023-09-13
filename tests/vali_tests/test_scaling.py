import unittest

import numpy as np

from vali_objects.scaling.scaling import Scaling


class TestScaling(unittest.TestCase):
    def test_count_decimal_places(self):
        self.assertEqual(3, Scaling.count_decimal_places(23.292))
        self.assertEqual(15, Scaling.count_decimal_places(23.2912391212947128491722))
        self.assertEqual(0, Scaling.count_decimal_places(23))

    def test_scale_unscale_values(self):
        values = np.array([x for x in range(0, 1000)])
        avg, scaled_values = Scaling.scale_values(values)

        self.assertEqual(avg, 499.5)

        self.assertLess(scaled_values.max(), 0.05)
        self.assertGreater(scaled_values.min(), -0.05)

        unscaled_values = Scaling.unscale_values(avg, 0, scaled_values)
        self.assertTrue(np.array_equal(values, unscaled_values))

    def test_scale_data_structure(self):
        ds = []
        for i in range(0, 5):
            ds.append([x for x in range(0, 1000)])

        avgs, dps, ds_scale = Scaling.scale_data_structure(ds)

        self.assertEqual(5, len(avgs))
        self.assertEqual(5, len(dps))
        self.assertEqual(5, len(ds_scale))

        for avg in avgs:
            self.assertEqual(avg, 499.5)
        for dp in dps:
            self.assertEqual(dp, 0)
        for dss in ds_scale:
            self.assertLess(dss.max(), 0.05)
            self.assertGreater(dss.min(), -0.05)


if __name__ == '__main__':
    unittest.main()
