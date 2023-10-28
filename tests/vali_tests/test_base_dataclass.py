# developer: Taoshidev
# Copyright © 2023 Taoshi, LLC

import unittest
import random

import numpy as np

from vali_config import ValiConfig
from vali_objects.dataclasses.client_request import ClientRequest
from vali_objects.dataclasses.prediction_data_file import PredictionDataFile
from vali_objects.dataclasses.prediction_request import PredictionRequest


class TestBaseDataClass(unittest.TestCase):
    def test_pdf_creation(self):
        try:
            po = PredictionDataFile(
                client_uuid="test",
                stream_type="TEST",
                stream_id=1,
                topic_id=1,
                request_uuid="testuid",
                miner_uid=123,
                start=1234,
                end=12345,
                vmaxs=[0.1, 0.2, 0.3],
                vmins=[0.1, 0.2, 0.3],
                decimal_places=[1, 2, 3],
                predictions=np.array([1, 2, 3])
            )
        except Exception as e:
            self.assertTrue(e, TypeError)

        try:
            po = PredictionDataFile(
                client_uuid="test",
                stream_type="TEST",
                stream_id=1,
                topic_id=1,
                request_uuid="testuid",
                miner_uid="test",
                start=1234,
                end=12345,
                vmaxs=["test", "test", "test"],
                vmins=[0.1, 0.2, 0.3],
                decimal_places=[1, 2, 3],
                predictions=np.array([1, 2, 3])
            )
        except Exception as e:
            self.assertTrue(e, TypeError)

        try:
            po = PredictionDataFile(
                client_uuid="test",
                stream_type="TEST",
                stream_id=1,
                topic_id=1,
                request_uuid="testuid",
                miner_uid="test",
                start=1234,
                end=12345,
                vmaxs=[0.1, 0.2, 0.3],
                vmins=[0.1, 0.2, 0.3],
                decimal_places=[1, 2, 3],
                predictions=np.array(["1", "2"])
            )
        except Exception as e:
            self.assertTrue(e, TypeError)

        test = PredictionRequest(
                request_uuid="test",
                df="test",
                files="test",
                predictions="test"
            )

        test2 = ClientRequest(
            stream_type="TEST",
            topic_id=1,
            schema_id=1,
            feature_ids=[0.001, 0.002, 0.003, 0.004],
            prediction_size=int(random.uniform(ValiConfig.PREDICTIONS_MIN, ValiConfig.PREDICTIONS_MAX)),
            additional_details={
                "tf": 5,
                "trade_pair": "BTCUSD"
            }
        )



if __name__ == '__main__':
    unittest.main()
