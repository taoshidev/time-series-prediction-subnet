import time
import unittest
from datetime import datetime

from tests.vali_tests.samples.testing_data import TestingData
from time_util.time_util import TimeUtil


class TestTimeUtil(unittest.TestCase):

    def test_now_in_millis(self):
        self.assertTrue(int(datetime.now().timestamp() * 1000) > 1694482321729)
        self.assertAlmostEqual(int(datetime.now().timestamp() * 1000), TimeUtil.now_in_millis())  # add assertion here

    def test_convert_millis_to_timestamp(self):
        time.sleep(1)
        ct = TimeUtil.convert_millis_to_timestamp(TimeUtil.now_in_millis())
        self.assertGreater(ct, self._t)

    def test_generate_start_timestamp(self):
        older_time = TimeUtil.generate_start_timestamp(1)
        self.assertGreater(self._t, older_time)

    def test_generate_range_timestamps(self):
        start = TestingData.test_start_time
        generated_timestamps = TimeUtil.generate_range_timestamps(start, 5)
        self.assertEqual(TestingData.test_generated_timestamps, generated_timestamps)

    def setUp(self) -> None:
        self._t = datetime.now()


if __name__ == '__main__':
    unittest.main()
