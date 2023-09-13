import unittest

import numpy as np

from vali_objects.dataclasses.prediction_output import PredictionOutput


class TestBaseDataClass(unittest.TestCase):
    def test_po_creation(self):
        try:
            po = PredictionOutput(
                client_uuid="test",
                stream_type=1,
                topic_id=1,
                request_uuid="testuid",
                miner_uid=123,
                start=1234,
                end=12345,
                avgs=[0.1, 0.2, 0.3],
                decimal_places=[1, 2, 3],
                predictions=np.array([1, 2, 3])
            )
        except Exception as e:
            self.assertTrue(e, TypeError)

        try:
            po = PredictionOutput(
                client_uuid="test",
                stream_type=1,
                topic_id=1,
                request_uuid="testuid",
                miner_uid="test",
                start=1234,
                end=12345,
                avgs=["test", "test", "test"],
                decimal_places=[1, 2, 3],
                predictions=np.array([1, 2, 3])
            )
        except Exception as e:
            self.assertTrue(e, TypeError)

        try:
            po = PredictionOutput(
                client_uuid="test",
                stream_type=1,
                topic_id=1,
                request_uuid="testuid",
                miner_uid="test",
                start=1234,
                end=12345,
                avgs=[1, 2, 3],
                decimal_places=[1, 2, 3],
                predictions=np.array(["1", "2"])
            )
        except Exception as e:
            self.assertTrue(e, TypeError)


if __name__ == '__main__':
    unittest.main()
