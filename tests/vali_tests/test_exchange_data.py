import unittest
from datetime import datetime, timezone
import random

from data_generator.data_generator_handler import DataGeneratorHandler
from data_generator.financial_markets_generator.binance_data import BinanceData
from data_generator.financial_markets_generator.bybit_data import ByBitData
from time_util.time_util import TimeUtil
from vali_config import ValiConfig
from vali_objects.dataclasses.client_request import ClientRequest
from vali_objects.utils.vali_utils import ValiUtils


class TestExchangeData(unittest.TestCase):

    @staticmethod
    def generate_start_end_ms(start_dt):
        start_ms = TimeUtil.timestamp_to_millis(start_dt)
        end_ms = TimeUtil.timestamp_to_millis(start_dt) + TimeUtil.minute_in_millis(
            100 * 5)

        return start_ms, end_ms

    def test_exchange_data(self):

        client_request = ClientRequest(
            client_uuid="test_client_uuid",
            stream_type="BTCUSD-5m",
            topic_id=1,
            schema_id=1,
            feature_ids=[0.001, 0.002, 0.003, 0.004],
            prediction_size=int(random.uniform(ValiConfig.PREDICTIONS_MIN, ValiConfig.PREDICTIONS_MAX)),
            additional_details = {
                "tf": 5,
                "trade_pair": "BTCUSD"
            }
        )

        for ind, exchange in enumerate([BinanceData()]):
            start_dt = datetime(2023, 11, 1, 0, 0).replace(tzinfo=timezone.utc)
            start_ms, end_ms = TestExchangeData.generate_start_end_ms(start_dt)

            self.assertEqual(start_ms + TimeUtil.minute_in_millis(5) * 100, end_ms)

            data_structure = ValiUtils.get_standardized_ds()

            exchange.get_data_and_structure_data_points(
                client_request.additional_details["trade_pair"],
                client_request.additional_details["tf"],
                data_structure,
                (start_ms, end_ms)
            )

            # on the minute will pass back an extra value, handled by data generator handler
            self.assertEqual(len(data_structure[0]), 101)

            start_dt1 = datetime(2023, 11, 1, 0, 1).replace(tzinfo=timezone.utc)
            start_ms1, end_ms1 = TestExchangeData.generate_start_end_ms(start_dt1)

            data_structure = ValiUtils.get_standardized_ds()

            exchange.get_data_and_structure_data_points(
                client_request.additional_details["trade_pair"],
                client_request.additional_details["tf"],
                data_structure,
                (start_ms1, end_ms1)
            )

            exchange_start = TimeUtil.millis_to_timestamp(data_structure[0][0])
            exchange_end = TimeUtil.millis_to_timestamp(data_structure[0][len(data_structure[0]) - 1])

            # outside of on the minute will generate 100 properly
            self.assertEqual(len(data_structure[0]), 100)
            first_ten_close = data_structure[1][:10]

            if ind == 0:
                self.assertEqual(first_ten_close, [34597.8, 34587.68, 34611.80, 34553.15, 34545.82, 34570.00, 34579.61, 34508.41, 34529.8, 34505.97])
                # ordered in ascending order
                self.assertEqual([exchange_start.day, exchange_start.hour, exchange_start.minute], [1, 0, 5])
                self.assertEqual([exchange_end.day, exchange_end.hour, exchange_end.minute], [1, 8, 20])
            elif ind == 1:
                # ordered in ascending order
                self.assertEqual(exchange_start, datetime(2023, 11, 1, 0, 5, 0, 0))
                self.assertEqual(exchange_end, datetime(2023, 11, 1, 8, 20, 0, 0))

    # def test_sample_run(self):
    #     days = 1
    #     start = 5
    #
    #     ts_ranges = TimeUtil.convert_range_timestamps_to_millis(
    #         TimeUtil.generate_range_timestamps(
    #             TimeUtil.generate_start_timestamp(start), days))
    #     print(TimeUtil.millis_to_timestamp(ts_ranges[0][0]), TimeUtil.millis_to_timestamp(ts_ranges[0][1]))
    #     ds = ValiUtils.get_standardized_ds()
    #     data_generator_handler = DataGeneratorHandler()
    #     for ts_range in ts_ranges:
    #         print(TimeUtil.millis_to_timestamp(ts_range[0]), TimeUtil.millis_to_timestamp(ts_range[1]))
    #         # binance_data.get_data_and_structure_data_points(vali_request.stream_type,
    #         #                                                ds,
    #         #                                                ts_range)
    #         data_generator_handler.data_generator_handler(1,
    #                                                       0,
    #                                                       {
    #                                                           "tf": 5,
    #                                                           "trade_pair": "BTCUSD"
    #                                                       },
    #                                                       ds,
    #                                                       ts_range)
    #
    #     end_dt = ts_ranges[1][1]
    #     end_dt_conv = TimeUtil.millis_to_timestamp(end_dt)
    #     start_ms, end_ms = TestExchangeData.generate_start_end_ms(end_dt_conv)
    #
    #     print(TimeUtil.millis_to_timestamp(start_ms))
    #     print(TimeUtil.millis_to_timestamp(end_ms))
    #
    #     data_structure2 = ValiUtils.get_standardized_ds()
    #     data_generator_handler = DataGeneratorHandler()
    #     data_generator_handler.data_generator_handler(
    #         1,
    #         0,
    #         {
    #             "tf": 5,
    #             "trade_pair": "BTCUSD"
    #         },
    #         data_structure2,
    #         (start_ms, end_ms)
    #     )
    #     test = "test"


















