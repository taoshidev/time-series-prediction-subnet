# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Taoshi
# Copyright © 2023 TARVIS Labs, LLC

import uuid
import random
import time

import numpy as np

from data_generator.binance_data import BinanceData
from template.protocol import Forward
from time_util.time_util import TimeUtil
from vali_objects.cmw.cmw_objects.cmw_client import CMWClient
from vali_objects.cmw.cmw_objects.cmw_miner import CMWMiner
from vali_objects.cmw.cmw_objects.cmw_stream_type import CMWStreamType
from vali_objects.cmw.cmw_util import CMWUtil
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_config import ValiConfig
from vali_objects.predictions.prediction_output import PredictionOutput
from vali_objects.scaling.scaling import Scaling
from vali_objects.scoring.scoring import Scoring

import bittensor as bt


if __name__ == "__main__":
    # Testing everything outside of the bittensor infra
    days = int(random.uniform(ValiConfig.HISTORICAL_DATA_LOOKBACK_DAYS_MIN,
                                  ValiConfig.HISTORICAL_DATA_LOOKBACK_DAYS_MAX))

    ts_ranges = TimeUtil.convert_range_timestamps_to_millis(
        TimeUtil.generate_range_timestamps(
            TimeUtil.generate_start_timestamp(days), days))

    data_structure = [[], [], [], []]

    for ts_range in ts_ranges:
        BinanceData.get_data_and_structure_data_points(data_structure, ts_range)

    averages, dp_decimal_places, scaled_data_structure = Scaling.scale_data_structure(data_structure)

    client_uuid = str(uuid.uuid4())
    request_uuid = str(uuid.uuid4())
    topic_id = 1
    schema_id = 1
    # should be a hash of the vali hotkey & stream type (1 is fine)
    stream_type = 1
    # close, high, low, volume
    feature_ids = [0.001, 0.002, 0.003, 0.004]
    prediction_size = int(random.uniform(ValiConfig.PREDICTIONS_MIN, ValiConfig.PREDICTIONS_MAX))
    samples = bt.tensor(scaled_data_structure)

    forward_proto = Forward(
        request_uuid=request_uuid,
        stream_type=stream_type,
        feature_ids=feature_ids,
        samples=samples,
        topic_id=topic_id,
        schema_id=schema_id,
        prediction_size=prediction_size
    )

    vm = ValiUtils.get_vali_records()
    client = vm.get_client(client_uuid)
    if client is None:
        print("client doesnt exist")
        cmw_client = CMWClient().set_client_uuid(client_uuid)
        cmw_client.add_stream(CMWStreamType().set_stream_type(stream_type).set_topic_id(topic_id))
        vm.add_client(cmw_client)
    else:
        client_stream_type = client.stream_exists(stream_type)
        if client_stream_type is None:
            client.add_stream(CMWStreamType().set_stream_type(stream_type).set_topic_id(topic_id))
    print(CMWUtil.load_cmw(vm))
    ValiUtils.set_vali_memory_and_bkp(CMWUtil.load_cmw(vm))

    preds = 2
    for a in range(0, 25):
        s_predictions = np.array([random.uniform(-0.05,0.05) for i in range(0, preds)])

        #using proto for testing
        forward_proto.predictions = bt.tensor(s_predictions)
        # convert to millis
        ts = TimeUtil.minute_in_millis(preds * 5)

        unscaled_data_structure = Scaling.unscale_values(averages[0], dp_decimal_places[0], s_predictions)
        output_uuid = str(uuid.uuid4())
        miner_uuid = str(uuid.uuid4())
        # just the vali hotkey for now

        po = PredictionOutput(
            client_uuid=client_uuid,
            stream_type=stream_type,
            request_uuid=request_uuid,
            miner_uid=miner_uuid,
            start=TimeUtil.now_in_millis(),
            end=TimeUtil.now_in_millis() + ts,
            avgs=averages,
            decimal_places=dp_decimal_places,
            predictions=s_predictions
        )
        ValiUtils.save_predictions_request(output_uuid, po)

    all_files = ValiBkpUtils.get_all_files_in_dir(ValiBkpUtils.get_vali_responses_dir())
    print(all_files)
    request_to_complete = {}

    while True:
        break_now = False
        for file in all_files:
            unpickled_data_file = ValiUtils.get_vali_predictions(file)
            if TimeUtil.now_in_millis() > unpickled_data_file.end:
                unpickled_unscaled_data_structure = Scaling.unscale_values(unpickled_data_file.avgs[0],
                                                                           unpickled_data_file.decimal_places[0],
                                                                           unpickled_data_file.predictions)
                if unpickled_data_file.request_uuid not in request_to_complete:
                    request_to_complete[unpickled_data_file.request_uuid] = {
                        "start": unpickled_data_file.start,
                        "end": unpickled_data_file.end,
                        "client_uuid": client_uuid,
                        "stream_type": stream_type,
                        "predictions": {}
                    }
                request_to_complete[
                    unpickled_data_file.request_uuid][
                    "predictions"][unpickled_data_file.miner_uid] = unpickled_unscaled_data_structure
        if len(request_to_complete) > 0:
            break
        print("going back to sleep to wait...")
        time.sleep(60)

    print("request time done!")
    print(request_to_complete)

    updated_vm = ValiUtils.get_vali_records()
    print(updated_vm)

    for request_uuid, request_details in request_to_complete.items():
        data_structure = [[], [], [], []]
        BinanceData.get_data_and_structure_data_points(data_structure,
                                                       [request_details["start"], request_details["end"]])
        print("live results:", data_structure[0])
        scores = {}
        for miner_uid, miner_preds in request_details["predictions"].items():
            scores[miner_uid] = Scoring.score_response(miner_preds, data_structure[0])

        scaled_scores = Scoring.scale_scores(scores)
        stream_type = updated_vm.get_client(request_details["client_uuid"]).get_stream(request_details["stream_type"])

        # add scaled scores
        for miner_uid, scaled_score in scaled_scores.items():
            stream_miner = stream_type.get_miner(miner_uid)
            if stream_miner is None:
                stream_miner = CMWMiner(miner_uid, 0, 0, [])
                stream_type.add_miner(stream_miner)
            stream_miner.add_score(scaled_score)

        sorted_scores = sorted(scaled_scores.items(), key=lambda x: x[1], reverse=True)
        top_winners = sorted_scores[:5]
        top_winner_miner_uids = [item[0] for item in top_winners]

        print("winning scores", top_winners)
        print("winning miner uids", top_winner_miner_uids)
        # for miner_uid, score in scores.items():
        #     if score < lowest_score:
        #         lowest_score = score
        #         lowest_scoring_miner = miner_uid
        # total = sum(scores.values())
        # average = total / len(scores)
        # print("winning miner, score, and avg", lowest_scoring_miner, lowest_score, average)
    print(CMWUtil.load_cmw(updated_vm))
    ValiUtils.set_vali_memory_and_bkp(CMWUtil.load_cmw(updated_vm))

    #TODO - remove files that have been read already




