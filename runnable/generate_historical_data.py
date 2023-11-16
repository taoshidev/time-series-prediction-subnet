# developer: Taoshi
# Copyright © 2023 Taoshi, LLC
import hashlib
import uuid
import random

from data_generator.data_generator_handler import DataGeneratorHandler
from mining_objects.mining_utils import MiningUtils

from time_util.time_util import TimeUtil
from vali_objects.dataclasses.client_request import ClientRequest
from vali_objects.utils.vali_utils import ValiUtils
from vali_config import ValiConfig
from vali_objects.scaling.scaling import Scaling

import bittensor as bt


if __name__ == "__main__":
    use_local = False
    train = False
    train_new_data = False
    test = True

    # Testing everything locally outside of the bittensor infra to ensure logic works properly

    client_request = ClientRequest(
        client_uuid=str(uuid.uuid4()),
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

    # choose the range of days to look back
    # number of days back start
    days_back_start = 100
    # number of days forward since end day
    # for example start from 100 days ago and get 70 days from 100 days ago (100 days ago, 99 days ago, 98 days ago, etc.)
    days_back_end = 70

    ts_ranges = TimeUtil.convert_range_timestamps_to_millis(
        TimeUtil.generate_range_timestamps(
            TimeUtil.generate_start_timestamp(days_back_start), days_back_end, True))

    data_structure = ValiUtils.get_standardized_ds()

    data_generator_handler = DataGeneratorHandler()
    for ts_range in ts_ranges:
        data_generator_handler.data_generator_handler(client_request.topic_id, 0,
                                                      client_request.additional_details, data_structure, ts_range)

    vmins, vmaxs, dp_decimal_places, scaled_data_structure = Scaling.scale_ds_with_ts(data_structure)

    # close timestamp, close, high, low, volume
    samples = bt.tensor(scaled_data_structure)
    MiningUtils.write_file("/runnable/historical_financial_data/", "data.pickle", data_structure, True)