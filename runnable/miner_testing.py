# developer: Taoshidev
# Copyright © 2023 Taoshi, LLC

import hashlib
import os
import uuid
import random
import time
from datetime import datetime

import numpy as np

from data_generator.data_generator_handler import DataGeneratorHandler
from mining_objects.base_mining_model import BaseMiningModel
from template.protocol import Forward
from time_util.time_util import TimeUtil
from vali_objects.cmw.cmw_objects.cmw_client import CMWClient
from vali_objects.cmw.cmw_objects.cmw_miner import CMWMiner
from vali_objects.cmw.cmw_objects.cmw_stream_type import CMWStreamType
from vali_objects.cmw.cmw_util import CMWUtil
from vali_objects.dataclasses.client_request import ClientRequest
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_config import ValiConfig
from vali_objects.dataclasses.prediction_data_file import PredictionDataFile
from vali_objects.scaling.scaling import Scaling
from vali_objects.scoring.scoring import Scoring

import bittensor as bt


if __name__ == "__main__":

    days_processed = []
    curr_iter = 0

    start_iter = []

    while True:

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

        # if set to true will use data now and not historical
        start_dt, end_dt, ts_ranges = ValiUtils.randomize_days(True)
        print(start_dt)

        if start_dt.year >= 2023 and start_dt.month > 6:
            pass
        else:
            continue

        '''
        ========================================================================
        RECOMMEND: define the timestamps you want to test 
        against using the start_dt & end_dt. In this logic I dont use
        the historical BTC data file but you can certainly choose to
        reference it rather than pull historical data from APIs.
        ========================================================================
        '''

        iter_add = 0

        request_uuid = str(uuid.uuid4())
        test_vali_hotkey = str(uuid.uuid4())
        # should be a hash of the vali hotkey & stream type (1 is fine)
        hash_object = hashlib.sha256(client_request.stream_type.encode())
        stream_id = hash_object.hexdigest()
        ts = TimeUtil.minute_in_millis(client_request.prediction_size * 5)

        data_structure = ValiUtils.get_standardized_ds()
        data_structure_orig = ValiUtils.get_standardized_ds()

        print("start", start_dt)
        print("end", end_dt)
        data_generator_handler = DataGeneratorHandler()
        for ts_range in ts_ranges:
            data_generator_handler.data_generator_handler(client_request.topic_id, 0,
                                                          client_request.additional_details, data_structure, ts_range)

        vmins, vmaxs, dp_decimal_places, scaled_data_structure = Scaling.scale_ds_with_ts(data_structure)
        print(scaled_data_structure)

        samples = bt.tensor(scaled_data_structure)

        forward_proto = Forward(
            request_uuid=request_uuid,
            stream_id=stream_id,
            samples=samples,
            topic_id=client_request.topic_id,
            feature_ids=client_request.feature_ids,
            schema_id=client_request.schema_id,
            prediction_size=client_request.prediction_size
        )

        vm = ValiUtils.get_vali_records()
        client = vm.get_client(client_request.client_uuid)
        if client is None:
            print("client doesnt exist")
            cmw_client = CMWClient().set_client_uuid(client_request.client_uuid)
            cmw_client.add_stream(CMWStreamType().set_stream_id(stream_id).set_topic_id(client_request.topic_id))
            vm.add_client(cmw_client)
        else:
            client_stream_type = client.get_stream(stream_id)
            if client_stream_type is None:
                client.add_stream(CMWStreamType().set_stream_id(stream_id).set_topic_id(client_request.topic_id))
        print(CMWUtil.dump_cmw(vm))
        ValiUtils.set_vali_memory_and_bkp(CMWUtil.dump_cmw(vm))

        print("number of predictions needed", client_request.prediction_size)

        '''
        ========================================================================
        Fill in the mining model you want to test here & how it can create
        its dataset for testing.
        ========================================================================
        '''

        mining_models = {
            "../mining_models/base_model.h5": {
                "window_size": 12,
                "id": 1,
                "mining_model": BaseMiningModel.base_model_dataset(samples)
            }
        }

        for model_name, mining_details in mining_models.items():
            prep_dataset = mining_details["mining_model"]
            base_mining_model = BaseMiningModel(len(prep_dataset.T))\
                .set_window_size(mining_details["window_size"])\
                .set_model_dir(model_name)\
                .load_model()

            prep_dataset_cp = prep_dataset[:]

            predicted_closes = []
            for i in range(client_request.prediction_size):
                predictions = base_mining_model.predict(prep_dataset_cp,)[0]
                prep_dataset_cp = np.concatenate((prep_dataset, predictions), axis=0)
                predicted_closes.append(predictions.tolist()[0][0])

            print(predicted_closes)

            output_uuid = str(uuid.uuid4())
            miner_uuid = "miner" + str(mining_details["id"])
            # just the vali hotkey for now

            pdf = PredictionDataFile(
                client_uuid=client_request.client_uuid,
                stream_type=client_request.stream_type,
                stream_id=stream_id,
                topic_id=client_request.topic_id,
                request_uuid=request_uuid,
                miner_uid=miner_uuid,
                start=TimeUtil.timestamp_to_millis(end_dt),
                end=TimeUtil.timestamp_to_millis(end_dt) + ts,
                vmins=vmins,
                vmaxs=vmaxs,
                decimal_places=dp_decimal_places,
                predictions=np.array(predicted_closes),
                prediction_size=client_request.prediction_size,
                additional_details=client_request.additional_details
            )
            ValiUtils.save_predictions_request(output_uuid, pdf)

        # creating some additional miners who generate noise to compare against
        for a in range(0, 25):
            # s_predictions = np.array(
            #     [random.uniform(0.499, 0.501) for i in range(0, client_request.prediction_size)])
            s_predictions = np.array(
                [random.uniform(samples.numpy()[1][len(samples.numpy()[1]) - 1] - .0001,
                                samples.numpy()[1][len(samples.numpy()[1]) - 1] + .0001) for i in
                 range(0, client_request.prediction_size)])

            # using proto for testing
            forward_proto.predictions = bt.tensor(s_predictions)

            # unscaled_data_structure = Scaling.unscale_values(vmins[0],
            #                                                  vmaxs[0],
            #                                                  dp_decimal_places[0],
            #                                                  forward_proto.predictions.numpy())
            output_uuid = str(uuid.uuid4())
            miner_uuid = "noise-miner-" + str(a)
            # just the vali hotkey for now

            pdf = PredictionDataFile(
                client_uuid=client_request.client_uuid,
                stream_type=client_request.stream_type,
                stream_id=stream_id,
                topic_id=client_request.topic_id,
                request_uuid=request_uuid,
                miner_uid=miner_uuid,
                start=TimeUtil.timestamp_to_millis(end_dt),
                end=TimeUtil.timestamp_to_millis(end_dt) + ts,
                vmins=vmins,
                vmaxs=vmaxs,
                decimal_places=dp_decimal_places,
                predictions=forward_proto.predictions.numpy(),
                prediction_size=client_request.prediction_size,
                additional_details=client_request.additional_details
            )
            ValiUtils.save_predictions_request(output_uuid, pdf)

        print("remove stale files")
        ValiBkpUtils.delete_stale_files(ValiBkpUtils.get_vali_predictions_dir())

        predictions_to_complete = []

        # this logic will only continue to iterate if you're testing against live data otherwise
        # it will fire immediately with historical as its already prepared
        while True:
            break_now = False
            predictions_to_complete.extend(ValiUtils.get_predictions_to_complete())
            if len(predictions_to_complete) > 0:
                break
            print("going back to sleep to wait...")
            time.sleep(60)

        print("request time done!")

        updated_vm = ValiUtils.get_vali_records()

        # logic to gather and score predictions
        for request_details in predictions_to_complete:
            request_df = request_details.df
            data_structure = ValiUtils.get_standardized_ds()
            data_generator_handler = DataGeneratorHandler()
            data_generator_handler.data_generator_handler(request_df.topic_id,
                                                          request_df.prediction_size,
                                                          request_df.additional_details,
                                                          data_structure,
                                                          (request_df.start, request_df.end))
            scores = {}
            for miner_uid, miner_preds in request_details.predictions.items():
                scores[miner_uid] = Scoring.score_response(miner_preds, data_structure[1])

            scaled_scores = Scoring.simple_scale_scores(scores)
            stream_id = updated_vm.get_client(request_df.client_uuid).get_stream(request_df.stream_id)

            sorted_scores = sorted(scaled_scores.items(), key=lambda x: x[1], reverse=True)
            winning_scores = sorted_scores[:10]

            weighed_scores = Scoring.weigh_miner_scores(winning_scores)
            weighed_winning_scores = weighed_scores[:10]
            weighed_winning_scores_dict = {score[0]:score[1] for score in weighed_winning_scores}

            print("winning distribution", weighed_winning_scores)
            print("weighed_scores_dict", weighed_winning_scores_dict)

            time_now = TimeUtil.now_in_millis()
            # add scaled scores
            for miner_uid, scaled_score in scaled_scores.items():
                stream_miner = stream_id.get_miner(miner_uid)
                if stream_miner is None:
                    stream_miner = CMWMiner(miner_uid)
                    stream_id.add_miner(stream_miner)
                stream_miner.add_unscaled_score([time_now, scores[miner_uid]])
                if miner_uid in weighed_winning_scores_dict:
                    stream_miner.add_win_score([time_now, weighed_winning_scores_dict[miner_uid]])
            for file in request_details.files:
                os.remove(file)

        # end results are stored in the path validation/backups/valiconfig.json (same as it goes on the subnet)
        ValiUtils.set_vali_memory_and_bkp(CMWUtil.dump_cmw(updated_vm))
