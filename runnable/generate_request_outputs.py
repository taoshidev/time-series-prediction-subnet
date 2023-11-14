import json
import os
import time
from datetime import datetime

from time_util.time_util import TimeUtil
from vali_objects.cmw.cmw_util import CMWUtil
from vali_objects.scaling.scaling import Scaling
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils


def prepare_latest_predictions(return_result=False):
    # TODO this is fine for now, but eventually needs to be split
    #  out to multiple topics and streams will leave as is for now to move faster

    # =====================================
    # Prepare latest predictions
    # =====================================

    def append_to_latest_predictions_files(lp, updf):
        updf.predictions = Scaling.unscale_values(updf.vmins[0],
                               updf.vmaxs[0],
                               updf.decimal_places[0],
                               updf.predictions).tolist()
        lp[INITIAL_STREAM_TYPE].append(updf.__dict__)

    LATEST_PREDICTIONS = "latest_predictions"
    LATEST_PREDICTIONS_FILENAME = LATEST_PREDICTIONS + ".json"
    LATEST_PREDICTIONS_LOCATION = ValiBkpUtils.get_vali_outputs_dir() + LATEST_PREDICTIONS_FILENAME

    INITIAL_STREAM_TYPE = "BTCUSD-5m"

    predictions_dir = ValiBkpUtils.get_vali_predictions_dir()
    prediction_files = os.listdir(predictions_dir)

    # handle predictions output first

    # 1. get the latest file
    # 2. get the latest files request uuid
    # 3. get all files that have the same request uuid
    # 4. convert all values back to normalized values
    # 5. store all the values as a dictionary, "latest_predictions": [] in dir validation/latest_predictions/

    if len(prediction_files) > 0:
        latest_predictions = {
            INITIAL_STREAM_TYPE : []
        }

        prediction_files.sort(key=lambda x: os.path.getctime(os.path.join(predictions_dir, x)), reverse=True)
        # 1. get the latest file
        latest_file = prediction_files[0]
        # 2. get the latest files request uuid
        unpickled_file_pdf = ValiUtils.get_vali_predictions(predictions_dir + latest_file)
        lf_request_uuid = unpickled_file_pdf.request_uuid
        # adding first to latest predictions
        append_to_latest_predictions_files(latest_predictions, unpickled_file_pdf)

        # 3. get all files that have the same request uuid
        for file in prediction_files:
            unpickled_file_pdf = ValiUtils.get_vali_predictions(predictions_dir + file)
            if unpickled_file_pdf.request_uuid == lf_request_uuid:
                append_to_latest_predictions_files(latest_predictions, unpickled_file_pdf)
        ValiBkpUtils.make_dir(ValiBkpUtils.get_vali_outputs_dir())
        ValiBkpUtils.write_to_vali_dir(LATEST_PREDICTIONS_LOCATION, latest_predictions)
        if return_result:
            return latest_predictions
    else:
        print("The directory is empty.")


def prepare_cmw_object():
    # =====================================
    # Prepare CMW
    # =====================================
    cmw_object = {}

    _miner_id = "miner_id"
    _unscaled_scores = "unscaled_scores"
    _win_scores = "win_scores"
    _timestamp = "timestamp"
    _score = "score"

    '''
    win_scores = {
        <timestamp>: [(miner_uid, win_score, unscaled_score)]
    }
    '''
    win_scores = {}

    cmw_files = os.listdir(ValiBkpUtils.get_vali_bkp_dir())
    for cmw_file in cmw_files:
        cmw_file_json = json.loads(ValiBkpUtils.get_vali_file(ValiBkpUtils.get_vali_bkp_dir() + cmw_file))
        loaded_cmw = CMWUtil.load_cmw(cmw_file_json)
        for client in loaded_cmw.clients:
            for stream in client.streams:
                if stream.topic_id not in cmw_object:
                    cmw_object[stream.topic_id] = {}
                if stream.stream_id not in cmw_object[stream.topic_id]:
                    cmw_object[stream.topic_id][stream.stream_id] = {}
                for miner in stream.miners:
                    miner_win_scores = {mws[0]: (miner.miner_id, mws[1], mus[1]) for mws in miner.win_scores for mus in miner.unscaled_scores if mws[0] == mus[0]}
                    for key, value in miner_win_scores.items():
                        if key not in win_scores:
                            win_scores[key] = []
                        win_scores[key].append(value)
                    # for mws in miner.win_scores:
                    #     if mws[0] not in win_scores:
                    #         win_scores[mws[0]] = []
                    #
                    # if
                    # if miner.miner_id not in cmw_object[stream.topic_id][stream.stream_id]:
                    #     cmw_object[stream.topic_id][stream.stream_id][miner.miner_id] = {}
                    #     cmw_object[stream.topic_id][stream.stream_id][miner.miner_id]["unscaled_scores"] = []
                    #     cmw_object[stream.topic_id][stream.stream_id][miner.miner_id]["win_scores"] = []
                    # cmw_object[stream.topic_id][stream.stream_id][miner.miner_id]["unscaled_scores"].extend(miner.unscaled_scores)
                    # cmw_object[stream.topic_id][stream.stream_id][miner.miner_id]["win_scores"].extend(miner.win_scores)

    # subnet performance
    subnet_performance_dict = {}
    total_value = 0
    for ts, win_values in win_scores.items():
        total_value = 0
        for win_value in win_values:
            total_value += win_value[2]
        # assuming sum to 1
        subnet_performance_dict[ts] = total_value / len(win_values)

    subnet_performance = [{"timestamp": key, "score": value} for key, value in subnet_performance_dict.items()]
    subnet_performance = sorted(subnet_performance, key=lambda x: x['timestamp'])

    # top miners past 7 days
    top_miners_seven_days_dict = {}

    seven_days_ago = TimeUtil.timestamp_to_millis(TimeUtil.generate_start_timestamp(7))

    win_scores_seven_days_ago = {time: data for time, data in win_scores.items() if time > seven_days_ago}
    for ts, win_values in win_scores_seven_days_ago.items():
        for win_value in win_values:
            if win_value[0] not in top_miners_seven_days_dict:
                top_miners_seven_days_dict[win_value[0]] = 0
            top_miners_seven_days_dict[win_value[0]] += win_value[1]

    top_miners_seven_days = [{"miner_uid": key, "score": value} for key, value in top_miners_seven_days_dict.items()]
    top_miners_seven_days = sorted(top_miners_seven_days, key=lambda x: x['score'], reverse=True)

    top_twenty_five_miners_seven_days = top_miners_seven_days[:25]
    top_twenty_five_miners_seven_days_ids = [value["miner_uid"] for value in top_twenty_five_miners_seven_days]

    latest_predictions = prepare_latest_predictions(True)

    # get top predictions
    top_predictions = []
    for key, values in latest_predictions.items():
        for value in values:
            curr_timestamp = value["start"]
            if value["miner_uid"] in top_twenty_five_miners_seven_days_ids:
                curr_predictions = {
                    "miner_uid": value["miner_uid"],
                    "timestamp": value["start"],
                    "predictions": [],
                }
                for row in value["predictions"]:
                    curr_timestamp += TimeUtil.minute_in_millis(5)
                    curr_predictions["predictions"].append({
                        "timestamp": curr_timestamp,
                        "prediction": row
                    })
                top_predictions.append(curr_predictions)

    cmw_aggregated_obj = {
        "subnet_performance": subnet_performance,
        "top_miners_7": top_twenty_five_miners_seven_days,
        "top_miners_7_predictions": top_predictions
    }

    ValiBkpUtils.make_dir(ValiBkpUtils.get_vali_outputs_dir())
    ValiBkpUtils.write_to_vali_dir(ValiBkpUtils.get_vali_outputs_dir() + "cmw.json", cmw_aggregated_obj)


if __name__ == "__main__":
    print("generate request outputs")
    while True:
        now = datetime.utcnow()
        if True:
            print(f"{now}: outputting cmw object")
            prepare_cmw_object()
            print(f"{now}: successfully outputted cmw object")
            time.sleep(60)


