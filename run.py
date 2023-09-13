# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Taoshi
# Copyright © 2023 TARVIS Labs, LLC

import os
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
from vali_objects.dataclasses.client_request import ClientRequest
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_config import ValiConfig
from vali_objects.dataclasses.prediction_output import PredictionOutput
from vali_objects.scaling.scaling import Scaling
from vali_objects.scoring.scoring import Scoring

import bittensor as bt


def randomize_days() -> (int, int, list[list[int, int]]):
    days = int(random.uniform(ValiConfig.HISTORICAL_DATA_LOOKBACK_DAYS_MIN,
                              ValiConfig.HISTORICAL_DATA_LOOKBACK_DAYS_MAX))
    # if 1 then historical lookback, otherwise live
    if random.randint(1, 1):
        print("generating historical")
        start = int(random.uniform(30,1460))
    else:
        print("generating live")
        start = days
    TimeUtil.generate_start_timestamp(start)
    return TimeUtil.generate_start_timestamp(start), \
           TimeUtil.generate_start_timestamp(start-days), \
           TimeUtil.convert_range_timestamps_to_millis(
        TimeUtil.generate_range_timestamps(
            TimeUtil.generate_start_timestamp(start), days))


if __name__ == "__main__":
    # Testing everything locally outside of the bittensor infra to ensure logic works properly

    client_request = ClientRequest(
        client_uuid=str(uuid.uuid4()),
        stream_type=1,
        topic_id=1,
        schema_id=1,
        feature_ids=[0.001, 0.002, 0.003, 0.004],
        prediction_size=int(random.uniform(ValiConfig.PREDICTIONS_MIN, ValiConfig.PREDICTIONS_MAX))
    )

    start_dt, end_dt, ts_ranges = randomize_days()
    print("start", start_dt)
    print("end", end_dt)

    data_structure = [[], [], [], []]

    for ts_range in ts_ranges:
        BinanceData.get_data_and_structure_data_points(data_structure, ts_range)

    averages, dp_decimal_places, scaled_data_structure = Scaling.scale_data_structure(data_structure)

    request_uuid = str(uuid.uuid4())
    test_vali_hotkey = str(uuid.uuid4())
    # should be a hash of the vali hotkey & stream type (1 is fine)
    stream_type = hash(str(client_request.stream_type)+test_vali_hotkey)
    # close, high, low, volume
    samples = bt.tensor(scaled_data_structure)

    forward_proto = Forward(
        request_uuid=request_uuid,
        stream_type=stream_type,
        feature_ids=client_request.feature_ids,
        samples=samples,
        topic_id=client_request.topic_id,
        schema_id=client_request.schema_id,
        prediction_size=client_request.prediction_size
    )

    vm = ValiUtils.get_vali_records()
    client = vm.get_client(client_request.client_uuid)
    if client is None:
        print("client doesnt exist")
        cmw_client = CMWClient().set_client_uuid(client_request.client_uuid)
        cmw_client.add_stream(CMWStreamType().set_stream_type(stream_type).set_topic_id(client_request.topic_id))
        vm.add_client(cmw_client)
    else:
        client_stream_type = client.stream_exists(stream_type)
        if client_stream_type is None:
            client.add_stream(CMWStreamType().set_stream_type(stream_type).set_topic_id(client_request.topic_id))
    print(CMWUtil.dump_cmw(vm))
    ValiUtils.set_vali_memory_and_bkp(CMWUtil.dump_cmw(vm))

    # feel free to override for live data
    # preds = 2

    print("number of preds", client_request.prediction_size)
    for a in range(0, 25):
        s_predictions = np.array([random.uniform(-0.1,0.1) for i in range(0, client_request.prediction_size)])

        #using proto for testing
        forward_proto.predictions = bt.tensor(s_predictions)
        # convert to millis
        ts = TimeUtil.minute_in_millis(client_request.prediction_size * 5)

        unscaled_data_structure = Scaling.unscale_values(averages[0], dp_decimal_places[0], s_predictions)
        output_uuid = str(uuid.uuid4())
        miner_uuid = str(uuid.uuid4())
        # just the vali hotkey for now

        po = PredictionOutput(
            client_uuid=client_request.client_uuid,
            stream_type=stream_type,
            topic_id=client_request.topic_id,
            request_uuid=request_uuid,
            miner_uid=miner_uuid,
            start=TimeUtil.timestamp_to_millis(end_dt),
            end=TimeUtil.timestamp_to_millis(end_dt)+ts,
            avgs=averages,
            decimal_places=dp_decimal_places,
            predictions=s_predictions
        )
        ValiUtils.save_predictions_request(output_uuid, po)

    print("remove stale files")
    ValiBkpUtils.delete_stale_files(ValiBkpUtils.get_vali_predictions_dir())

    print("get all files")
    all_files = ValiBkpUtils.get_all_files_in_dir(ValiBkpUtils.get_vali_predictions_dir())
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
                        "data_file": unpickled_data_file,
                        "predictions": {}
                    }
                request_to_complete[
                    unpickled_data_file.request_uuid][
                    "predictions"][unpickled_data_file.miner_uid] = unpickled_unscaled_data_structure
                os.remove(file)
        if len(request_to_complete) > 0:
            break
        print("going back to sleep to wait...")
        time.sleep(60)

    print("request time done!")
    # print(request_to_complete)

    updated_vm = ValiUtils.get_vali_records()

    for request_uuid, request_details in request_to_complete.items():
        request_data_file = request_details["data_file"]
        data_structure = [[], [], [], []]
        BinanceData.get_data_and_structure_data_points(data_structure,
                                                       [request_data_file.start, request_data_file.end])
        print("results:", data_structure[0])
        scores = {}
        for miner_uid, miner_preds in request_details["predictions"].items():
            scores[miner_uid] = Scoring.score_response(miner_preds, data_structure[0])

        # scores["test"] = Scoring.score_response([row*1 for row in data_structure[0]], data_structure[0])
        # scores["test1"] = Scoring.score_response([row*0.975 for row in data_structure[0]], data_structure[0])
        # scores["test2"] = Scoring.score_response([row*0.95 for row in data_structure[0]], data_structure[0])

        scaled_scores = Scoring.scale_scores(scores)
        stream_type = updated_vm.get_client(request_data_file.client_uuid).get_stream(request_data_file.stream_type)

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

        weighed_scores = Scoring.weigh_miner_scores(top_winners)

        print("winning scores", top_winners)
        print("winning distribution", weighed_scores)

    print(CMWUtil.dump_cmw(updated_vm))
    ValiUtils.set_vali_memory_and_bkp(CMWUtil.dump_cmw(updated_vm))
