# developer: Taoshidev
# Copyright © 2023 Taoshi, LLC

import random

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from mining_objects.base_mining_model import BaseMiningModel
from mining_objects.mining_utils import MiningUtils
from time_util.time_util import TimeUtil
from vali_objects.dataclasses.client_request import ClientRequest
from vali_config import ValiConfig

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

        # numbers of rows to use in each sequence
        iter_add = 1000

        print("current iteradd " + str(iter_add))
        print("next iter " + str(curr_iter))

        data_structure = MiningUtils.get_file(
            "/runnable/historical_financial_data/data_training.pickle", True)
        data_structure = [data_structure[0][curr_iter:curr_iter + iter_add],
                          data_structure[1][curr_iter:curr_iter + iter_add],
                          data_structure[2][curr_iter:curr_iter + iter_add],
                          data_structure[3][curr_iter:curr_iter + iter_add],
                          data_structure[4][curr_iter:curr_iter + iter_add]]
        print("start " + str(TimeUtil.millis_to_timestamp(data_structure[0][0])))
        print("end " + str(TimeUtil.millis_to_timestamp(data_structure[0][len(data_structure[0]) - 1])))
        curr_iter += iter_add

        sds_ndarray = np.array(data_structure).T

        scaler = MinMaxScaler(feature_range=(0, 1))

        scaled_data = scaler.fit_transform(sds_ndarray)
        scaled_data = scaled_data.T
        prep_dataset = BaseMiningModel.base_model_dataset(scaled_data)

        base_mining_model = BaseMiningModel(len(prep_dataset.T)) \
            .set_neurons([[128, 0], [128, 0.2], [128, 0.4], [128, 0.6]]) \
            .set_window_size(100) \
            .set_learning_rate(0.0001) \
            .set_batch_size(iter_add) \
            .set_model_dir(f'mining_models/model1.h5')
        base_mining_model.train(prep_dataset, epochs=25)

