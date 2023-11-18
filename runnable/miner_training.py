# developer: Taoshidev
# Copyright © 2023 Taoshi, LLC

import hashlib
import random

from data_generator.data_generator_handler import DataGeneratorHandler
from mining_objects.base_mining_model import BaseMiningModel
from mining_objects.mining_utils import MiningUtils
from time_util.time_util import TimeUtil
from vali_objects.dataclasses.client_request import ClientRequest
from vali_objects.utils.vali_utils import ValiUtils
from vali_config import ValiConfig
from vali_objects.scaling.scaling import Scaling

import bittensor as bt


if __name__ == "__main__":

    # if you want the data to start from a certain location
    curr_iter = 0

    while True:
        # if you want to use the local historical btc data file
        use_local = True

        client_request = ClientRequest(
            client_uuid="test_client_uuid",
            stream_type="BTCUSD-5m",
            topic_id=1,
            schema_id=1,
            feature_ids=[0.001, 0.002, 0.003, 0.004],
            prediction_size=int(random.uniform(ValiConfig.PREDICTIONS_MIN, ValiConfig.PREDICTIONS_MAX)),
            additional_details={
                "tf": 5,
                "trade_pair": "BTCUSD"
            }
        )

        start_dt, end_dt, ts_ranges = ValiUtils.randomize_days(True)

        # numbers of rows to use in each sequence
        iter_add = 1000

        hash_object = hashlib.sha256(client_request.stream_type.encode())
        stream_id = hash_object.hexdigest()

        data_structure = ValiUtils.get_standardized_ds()

        # if you dont want to use the local file and want to gather historical data.
        # as set standard will use randomized historical data from above
        if use_local is False:
            print("start", start_dt)
            print("end", end_dt)
            data_generator_handler = DataGeneratorHandler()
            for ts_range in ts_ranges:
                data_generator_handler.data_generator_handler(client_request.topic_id, 0,
                                                              client_request.additional_details, data_structure, ts_range)

            vmins, vmaxs, dp_decimal_places, scaled_data_structure = Scaling.scale_ds_with_ts(data_structure)
            print(scaled_data_structure)

            # close, high, low, volume
            samples = bt.tensor(scaled_data_structure)
            # ValiUtils.save_predictions_request("test", data_structure)
        else:
            print("next iter", curr_iter)
            curr_iter += iter_add
            # historical doesnt have timestamps
            data_structure = MiningUtils.get_file("/runnable/historical_financial_data/data.pickle", True)
            data_structure = [data_structure[0][curr_iter:curr_iter+iter_add],
                              data_structure[1][curr_iter:curr_iter+iter_add],
                              data_structure[2][curr_iter:curr_iter+iter_add],
                              data_structure[3][curr_iter:curr_iter+iter_add],
                              data_structure[4][curr_iter:curr_iter+iter_add]]
            print(len(data_structure[0]))
            print("start", TimeUtil.millis_to_timestamp(data_structure[0][0]))
            print("end", TimeUtil.millis_to_timestamp(data_structure[0][len(data_structure[0])-1]))
            vmins, vmaxs, dp_decimal_places, scaled_data_structure = Scaling.scale_ds_with_ts(data_structure)
            samples = bt.tensor(scaled_data_structure)

        # will iterate and prepare the dataset and train the model as provided
        prep_dataset = BaseMiningModel.base_model_dataset(samples)
        base_mining_model = BaseMiningModel(len(prep_dataset.T)).set_model_dir('mining_models/model2.h5')
        base_mining_model.train(prep_dataset, epochs=25)
