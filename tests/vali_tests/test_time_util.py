# developer: Taoshidev
# Copyright © 2023 Taoshi Inc
import time
from time_util import datetime
from time_util.time_util import TimeUtil
import unittest


class TestTimeUtil(unittest.TestCase):
    def test_now_in_millis(self):
        self.assertTrue(int(datetime.now().timestamp() * 1000) > 1694482321729)
        self.assertAlmostEqual(
            int(datetime.utcnow().timestamp() * 1000), TimeUtil.now_in_millis()
        )

    def test_convert_millis_to_timestamp(self):
        time.sleep(1)
        ct = TimeUtil.millis_to_timestamp(TimeUtil.now_in_millis())
        self.assertGreater(ct, self._t)

    def test_generate_start_convert_to_millis_back_to_timestamp(self):
        start = TimeUtil.generate_start_timestamp(0)
        ms = TimeUtil.timestamp_to_millis(start)
        updated_start = TimeUtil.millis_to_timestamp(ms)
        self.assertEqual(
            [updated_start.day, start.hour, start.minute],
            [updated_start.day, updated_start.hour, start.minute],
        )

    def test_generate_start_timestamp(self):
        older_time = TimeUtil.generate_start_timestamp(1)
        self.assertGreater(
            TimeUtil.timestamp_to_millis(self._t),
            TimeUtil.timestamp_to_millis(older_time),
        )

    def test_generating_start_end_results(self):
        dt = datetime(2023, 9, 19, 12, 0, 0)
        training_results_start = int(TimeUtil.timestamp_to_millis(dt))
        training_results_end = training_results_start + TimeUtil.minute_in_millis(25)
        self.assertEqual(
            (training_results_start, training_results_end),
            (1695124800000, 1695126300000),
        )

    def setUp(self) -> None:
        self._t = datetime.now()


if __name__ == "__main__":
    unittest.main()
