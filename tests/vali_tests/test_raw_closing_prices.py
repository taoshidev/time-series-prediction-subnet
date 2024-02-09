from mining_objects.base_mining_model import BaseMiningModel
import numpy as np
import os
import unittest
from sklearn.preprocessing import MinMaxScaler
from time_util import datetime
from time_util.time_util import TimeUtil
from vali_config import ValiConfig
from vali_objects.dataclasses.client_request import ClientRequest
from vali_objects.dataclasses.prediction_data_file import PredictionDataFile
from vali_objects.scaling.scaling import Scaling
from vali_objects.utils.vali_utils import ValiUtils


class TestRawClosingPrices(unittest.TestCase):
    def test_predict_close(self):
        model_chosen = {
            "creation_id": "model2308",
            "model_dir": ValiConfig.BASE_DIR + "/mining_models/model_v4_1.h5",
            "window_size": 100,
        }

        base_mining_model = (
            BaseMiningModel(4)
            .set_window_size(model_chosen["window_size"])
            .set_model_dir(model_chosen["model_dir"])
            .load_model()
        )

        client_request = ClientRequest(
            client_uuid="test_client_uuid",
            stream_type="BTCUSD-5m",
            topic_id=1,
            schema_id=1,
            feature_ids=[0.001, 0.002, 0.003, 0.004],
            prediction_size=100,
            additional_details={"tf": 5, "trade_pair": "BTCUSD"},
        )

        start_dt = datetime(2023, 11, 1, 0, 1)

        start_dt, end_dt, ts_ranges = ValiUtils.randomize_days(True)

        start_ms = TimeUtil.timestamp_to_millis(start_dt)
        end_ms = TimeUtil.timestamp_to_millis(end_dt)

        data_structure = ValiUtils.get_standardized_ds()

        data_generator_handler = DataGeneratorHandler()
        for ts_range in ts_ranges:
            data_generator_handler.data_generator_handler(
                client_request.topic_id,
                0,
                client_request.additional_details,
                data_structure,
                ts_range,
            )

        data_structure = np.array(data_structure)

        request_uuid = "test_scaled_closing_prices"

        ts = TimeUtil.minute_in_millis(client_request.prediction_size * 5)
        last_price = data_structure[1][len(data_structure[1]) - 1]

        data_structure = data_structure.T[-601:, :].T

        # scaling data
        sds_ndarray = data_structure.T
        scaler = MinMaxScaler(feature_range=(0, 1))

        scaled_data = scaler.fit_transform(sds_ndarray)
        scaled_data = scaled_data.T

        prep_dataset_cp = BaseMiningModel.base_model_dataset(scaled_data)

        last_close = prep_dataset_cp.T[0][len(prep_dataset_cp) - 1]
        predicted_close = base_mining_model.predict(
            prep_dataset_cp,
        )[
            0
        ].tolist()[0]
        total_movement = predicted_close - last_close
        total_movement_increment = total_movement / client_request.prediction_size

        predicted_closes = []
        curr_price = last_close
        for x in range(client_request.prediction_size):
            curr_price += total_movement_increment
            predicted_closes.append(curr_price[0])

        close_column = data_structure[1].reshape(-1, 1)

        refit_scaler = MinMaxScaler()
        refit_scaler.fit(close_column)

        reshaped_predicted_closes = np.array(predicted_closes).reshape(-1, 1)
        predicted_closes = refit_scaler.inverse_transform(reshaped_predicted_closes)

        predicted_closes = predicted_closes.T.tolist()[0]
        print(predicted_closes)

    def tearDown(self):
        preds_to_complete = ValiUtils.get_predictions_to_complete()
        for preds in preds_to_complete:
            for file in preds.files:
                os.remove(file)
