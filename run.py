# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Taoshi
# Copyright © 2023 Taoshi, LLC
import hashlib
import os
import uuid
import random
import time
from copy import deepcopy
from datetime import datetime

import math
import numpy as np

from data_generator.data_generator_handler import DataGeneratorHandler
from data_generator.financial_markets_generator.binance_data import BinanceData
from data_generator.financial_markets_generator.bybit_data import ByBitData
from data_generator.financial_markets_generator.kraken_data import KrakenData
from mining_objects.base_mining_model import BaseMiningModel
from mining_objects.financial_market_indicators import FinancialMarketIndicators
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
    # curr_iter = 0
    # iters_chosen = [[6000, 9000], [9000, 12000], [21000, 24000], [24000, 27000], [33000, 36000],
    #                 [48000, 51000], [51000, 54000], [72000, 75000], [75000, 78000], [78000, 81000],
    #                 [90000, 93000], [93000, 96000], [99000, 102000], [111000, 114000], [117000, 120000],
    #                 [123000, 126000], [126000, 129000], [135000, 138000], [141000, 144000], [144000, 147000],
    #                 [162000, 165000], [165000, 168000], [183000, 186000], [192000, 195000], [201000, 204000],
    #                 [207000, 210000]]
    curr_iter = 150000

    start_iter = []
    # for row in iters_chosen:
    #     start_iter.append(row[0])

    while True:
        print(start_iter)

        use_local = False
        train = False
        train_new_data = False
        test = True

        # Testing everything locally outside of the bittensor infra to ensure logic works properly

        client_request = ClientRequest(
            client_uuid=str(uuid.uuid4()),
            stream_type="BTCUSD",
            topic_id=1,
            schema_id=1,
            feature_ids=[0.001, 0.002, 0.003, 0.004],
            prediction_size=int(random.uniform(ValiConfig.PREDICTIONS_MIN, ValiConfig.PREDICTIONS_MAX))
        )

        start_dt, end_dt, ts_ranges = ValiUtils.randomize_days(True)

        # ts_ranges = TimeUtil.convert_range_timestamps_to_millis(
        #     TimeUtil.generate_range_timestamps(
        #         TimeUtil.generate_start_timestamp(750), 750))
        #
        # start_dt = TimeUtil.generate_start_timestamp(750)
        # end_dt = datetime.utcnow()

        if use_local is False and start_dt < datetime(2023, 6, 1):
            print("passing on st ed", start_dt, end_dt)
            continue
        #
        # for row in days_processed:
        #     if row[0] < start_dt < row[1] or row[0] < end_dt < row[1] and start_dt > datetime(2022, 1, 1):
        #         continue
        # days_processed.append([start_dt, end_dt])

        # print("start", start_dt)
        # print("end", end_dt)

        iter_add = 3000

        request_uuid = str(uuid.uuid4())
        test_vali_hotkey = str(uuid.uuid4())
        # should be a hash of the vali hotkey & stream type (1 is fine)
        hash_object = hashlib.sha256(client_request.stream_type.encode())
        stream_id = hash_object.hexdigest()

        data_structure = [[], [], [], []]
        data_structure_orig = [[], [], [], []]

        if use_local is False:
            print("start", start_dt)
            print("end", end_dt)
            data_generator_handler = DataGeneratorHandler()
            for ts_range in ts_ranges:
                data_generator_handler.data_generator_handler(client_request.topic_id, 0,
                                                              client_request.stream_type, data_structure, ts_range)

            vmins, vmaxs, dp_decimal_places, scaled_data_structure = Scaling.scale_data_structure(data_structure)
            print(scaled_data_structure)

            # close, high, low, volume
            samples = bt.tensor(scaled_data_structure)
            # ValiUtils.save_predictions_request("test", data_structure)
        else:
            print("next iter", curr_iter)
            curr_iter += iter_add
            # if random.randint(0, 1) == 1:
                # print("iters chosen", iters_chosen)
                # iters_chosen.append([curr_iter-iter_add, curr_iter])
            data_structure = ValiUtils.get_vali_predictions(
                "runnable/historical_financial_data/historical_btc_data_2022_01_01_2023_06_01.pickle")
            print(len(data_structure[0]))

            data_structure = [data_structure[0][curr_iter:curr_iter+iter_add],
                              data_structure[1][curr_iter:curr_iter+iter_add],
                              data_structure[2][curr_iter:curr_iter+iter_add],
                              data_structure[3][curr_iter:curr_iter+iter_add]]
            vmins, vmaxs, dp_decimal_places, scaled_data_structure = Scaling.scale_data_structure(data_structure)
            samples = bt.tensor(scaled_data_structure)
            # if curr_iter not in start_iter:
            #     data_structure_orig = ValiUtils.get_vali_predictions("validation/test.pickle")
            #     print(len(data_structure[0]))
            #     data_structure = [data_structure_orig[0][curr_iter:curr_iter+iter_add],
            #                       data_structure_orig[1][curr_iter:curr_iter+iter_add],
            #                       data_structure_orig[2][curr_iter:curr_iter+iter_add],
            #                       data_structure_orig[3][curr_iter:curr_iter+iter_add]]
            #     vmins, vmaxs, dp_decimal_places, scaled_data_structure = Scaling.scale_data_structure(data_structure)
            #     samples = bt.tensor(scaled_data_structure)
            #     curr_iter += iter_add
            # else:
            #     curr_iter += iter_add
            #     continue

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

        # feel free to override for live data
        # preds = 2

        # convert to millis
        ts = TimeUtil.minute_in_millis(client_request.prediction_size * 5)

        print("number of preds", client_request.prediction_size)
        for a in range(0, 3):
            s_predictions = np.array([random.uniform(0.499, 0.501) for i in range(0, client_request.prediction_size)])

            #using proto for testing
            forward_proto.predictions = bt.tensor(s_predictions)

            # unscaled_data_structure = Scaling.unscale_values(vmins[0],
            #                                                  vmaxs[0],
            #                                                  dp_decimal_places[0],
            #                                                  forward_proto.predictions.numpy())
            output_uuid = str(uuid.uuid4())
            miner_uuid = str(uuid.uuid4())
            # just the vali hotkey for now

            pdf = PredictionDataFile(
                client_uuid=client_request.client_uuid,
                stream_type=client_request.stream_type,
                stream_id=stream_id,
                topic_id=client_request.topic_id,
                request_uuid=request_uuid,
                miner_uid=miner_uuid,
                start=TimeUtil.timestamp_to_millis(end_dt),
                end=TimeUtil.timestamp_to_millis(end_dt)+ts,
                vmins=vmins,
                vmaxs=vmaxs,
                decimal_places=dp_decimal_places,
                predictions=forward_proto.predictions.numpy(),
                prediction_size=client_request.prediction_size
            )
            ValiUtils.save_predictions_request(output_uuid, pdf)

        rsi = FinancialMarketIndicators.calculate_rsi(samples.tolist())
        rsi_50 = FinancialMarketIndicators.calculate_rsi(samples.tolist(), period = 50)
        rsi_100 = FinancialMarketIndicators.calculate_rsi(samples.tolist(), period = 100)
        rsi_200 = FinancialMarketIndicators.calculate_rsi(samples.tolist(), period = 200)
        rsi_500 = FinancialMarketIndicators.calculate_rsi(samples.tolist(), period=500)

        # macd, signal = FinancialMarketIndicators.calculate_macd(samples.tolist())
        middle, upper, lower = FinancialMarketIndicators.calculate_bollinger_bands(samples.tolist())
        middle_100, upper_100, lower_100 = FinancialMarketIndicators.calculate_bollinger_bands(samples.tolist(), window=100)
        middle_200, upper_200, lower_200 = FinancialMarketIndicators.calculate_bollinger_bands(samples.tolist(),
                                                                                               window=200)
        middle_500, upper_500, lower_500 = FinancialMarketIndicators.calculate_bollinger_bands(samples.tolist(),
                                                                                               window=500)
        middle_1000, upper_1000, lower_1000 = FinancialMarketIndicators.calculate_bollinger_bands(samples.tolist(),
                                                                                               window=1000)
        ema_100 = FinancialMarketIndicators.calculate_ema(samples.tolist(), 100)
        ema_200 = FinancialMarketIndicators.calculate_ema(samples.tolist(), 200)
        ema_500 = FinancialMarketIndicators.calculate_ema(samples.tolist(), 500)
        ema_1000 = FinancialMarketIndicators.calculate_ema(samples.tolist(), 1000)

        min_cutoff = 0
        for i, val in enumerate(ema_1000):
            if (val is None or math.isnan(val)) and i >= min_cutoff:
                min_cutoff = i
            if val is not None and math.isnan(val) is False:
                break

        for i, val in enumerate(middle_1000):
            if (val is None or math.isnan(val)) and i >= min_cutoff:
                min_cutoff = i
            if val is not None and math.isnan(val) is False:
                break

        min_cutoff += 1

        vmin_rsi, vmax_rsi, cutoff_rsi = Scaling.scale_values(rsi[min_cutoff:])
        vmin_rsi_50, vmax_rsi_50, cutoff_rsi_50 = Scaling.scale_values(rsi[min_cutoff:])
        vmin_rsi_100, vmax_rsi_100, cutoff_rsi_100 = Scaling.scale_values(rsi[min_cutoff:])
        vmin_rsi_200, vmax_rsi_200, cutoff_rsi_200 = Scaling.scale_values(rsi[min_cutoff:])
        vmin_rsi_500, vmax_rsi_500, cutoff_rsi_500 = Scaling.scale_values(rsi[min_cutoff:])
        vmin_rsi_1000, vmax_rsi_1000, cutoff_rsi_1000 = Scaling.scale_values(rsi[min_cutoff:])

        cutoff_middle = middle[min_cutoff:]
        cutoff_upper = upper[min_cutoff:]
        cutoff_lower = lower[min_cutoff:]

        cutoff_middle_100 = middle[min_cutoff:]
        cutoff_upper_100 = upper[min_cutoff:]
        cutoff_lower_100 = lower[min_cutoff:]

        cutoff_middle_200 = middle[min_cutoff:]
        cutoff_upper_200 = upper[min_cutoff:]
        cutoff_lower_200 = lower[min_cutoff:]

        cutoff_middle_500 = middle[min_cutoff:]
        cutoff_upper_500 = upper[min_cutoff:]
        cutoff_lower_500 = lower[min_cutoff:]

        cutoff_middle_1000 = middle[min_cutoff:]
        cutoff_upper_1000 = upper[min_cutoff:]
        cutoff_lower_1000 = lower[min_cutoff:]

        cutoff_ema_100 = ema_100[min_cutoff:]
        cutoff_ema_200 = ema_200[min_cutoff:]
        cutoff_ema_500 = ema_500[min_cutoff:]
        cutoff_ema_1000 = ema_1000[min_cutoff:]

        cutoff_close = samples.tolist()[0][min_cutoff:]
        cutoff_high = samples.tolist()[1][min_cutoff:]
        cutoff_low = samples.tolist()[2][min_cutoff:]
        cutoff_volume = samples.tolist()[3][min_cutoff:]

        prep_dataset = np.array([cutoff_close,
                                 cutoff_high,
                                 cutoff_low,
                                 cutoff_volume,
                                 cutoff_rsi,
                                 cutoff_rsi_100,
                                 cutoff_rsi_200,
                                 cutoff_rsi_500,
                                 cutoff_rsi_1000,
                                 cutoff_middle,
                                 cutoff_upper,
                                 cutoff_lower,
                                 cutoff_middle_100,
                                 cutoff_upper_100,
                                 cutoff_lower_100,
                                 cutoff_middle_200,
                                 cutoff_upper_200,
                                 cutoff_lower_200,
                                 cutoff_middle_500,
                                 cutoff_upper_500,
                                 cutoff_lower_500,
                                 cutoff_middle_1000,
                                 cutoff_upper_1000,
                                 cutoff_lower_1000,
                                 cutoff_ema_100,
                                 cutoff_ema_200,
                                 cutoff_ema_500,
                                 cutoff_ema_1000]).T

        base_mining_model = BaseMiningModel(len(prep_dataset.T))
        if train:
            base_mining_model.train(stream_id, prep_dataset)
        if test:
            prep_dataset_cp = prep_dataset[:]

            mining_models = {
                "mining_models/model1.keras": {
                    "window_size": 500,
                    "id": 1
                },
                "mining_models/model2.keras": {
                    "window_size": 100,
                    "id": 2
                },
                "mining_models/model3.keras": {
                    "window_size": 100,
                    "id": 3
                },
            }

            for model_name, mining_details in mining_models.items():
                base_mining_model = BaseMiningModel(len(prep_dataset.T),
                                                    window_size=mining_details["window_size"],
                                                    model=model_name)
                if train_new_data:
                    base_mining_model.train(stream_id, prep_dataset)
                predicted_closes = []
                for i in range(client_request.prediction_size):
                    predictions = base_mining_model.predict(stream_id, prep_dataset_cp)[0]
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
                    predictions=forward_proto.predictions.numpy(),
                    prediction_size=client_request.prediction_size
                )
                ValiUtils.save_predictions_request(output_uuid, pdf)

            print("remove stale files")
            ValiBkpUtils.delete_stale_files(ValiBkpUtils.get_vali_predictions_dir())

            predictions_to_complete = []

            while True:
                break_now = False
                predictions_to_complete.extend(ValiUtils.get_predictions_to_complete())
                # for file in all_files:
                #     unpickled_data_file = ValiUtils.get_vali_predictions(file)
                #     if TimeUtil.now_in_millis() > unpickled_data_file.end:
                #         unpickled_unscaled_data_structure = Scaling.unscale_values(unpickled_data_file.vmins[0],
                #                                                                    unpickled_data_file.vmaxs[0],
                #                                                                    unpickled_data_file.decimal_places[0],
                #                                                                    unpickled_data_file.predictions)
                #         if unpickled_data_file.request_uuid not in request_to_complete:
                #             request_to_complete[unpickled_data_file.request_uuid] = {
                #                 "po": unpickled_data_file,
                #                 "predictions": {}
                #             }
                #         request_to_complete[
                #             unpickled_data_file.request_uuid][
                #             "predictions"][unpickled_data_file.miner_uid] = unpickled_unscaled_data_structure
                #         os.remove(file)
                if len(predictions_to_complete) > 0:
                    break
                print("going back to sleep to wait...")
                time.sleep(60)

            print("request time done!")
            # print(request_to_complete)

            updated_vm = ValiUtils.get_vali_records()

            for request_details in predictions_to_complete:
                request_df = request_details.df
                data_structure = [[], [], [], []]
                if use_local is False:
                    data_generator_handler = DataGeneratorHandler()
                    data_generator_handler.data_generator_handler(request_df.topic_id,
                                                                  request_df.prediction_size,
                                                                  request_df.stream_type,
                                                                  data_structure,
                                                                  (request_df.start, request_df.end))
                else:
                    data_structure = [data_structure_orig[0][curr_iter:curr_iter+client_request.prediction_size],
                                      data_structure_orig[1][curr_iter:curr_iter + client_request.prediction_size],
                                      data_structure_orig[2][curr_iter:curr_iter + client_request.prediction_size],
                                      data_structure_orig[3][curr_iter:curr_iter + client_request.prediction_size]]
                print("results:", data_structure[0])
                scores = {}
                for miner_uid, miner_preds in request_details.predictions.items():
                    if "miner" in miner_uid:
                        print(miner_uid, "miner preds:", miner_preds)
                    scores[miner_uid] = Scoring.score_response(miner_preds, data_structure[0])
                print("scores", scores)

                # scores["test"] = Scoring.score_response([row*0.999 for row in data_structure[0]], data_structure[0])
                # scores["test1"] = Scoring.score_response([row*0.9975 for row in data_structure[0]], data_structure[0])
                # scores["test2"] = Scoring.score_response([row*0.995 for row in data_structure[0]], data_structure[0])
                # scores["test3"] = Scoring.score_response([row*0.995 for row in data_structure[0]], data_structure[0])
                # scores["test4"] = Scoring.score_response([row * 0.995 for row in data_structure[0]], data_structure[0])
                # scores["test5"] = Scoring.score_response([row * 0.9925 for row in data_structure[0]], data_structure[0])
                # scores["test6"] = Scoring.score_response([row * 0.99 for row in data_structure[0]], data_structure[0])
                # scores["test7"] = Scoring.score_response([row * 0.99 for row in data_structure[0]], data_structure[0])
                # scores["test8"] = Scoring.score_response([row * 0.9875 for row in data_structure[0]], data_structure[0])
                # scores["test9"] = Scoring.score_response([row * 0.9875 for row in data_structure[0]], data_structure[0])
                # scores["test10"] = Scoring.score_response([row * 0.985 for row in data_structure[0]], data_structure[0])
                # scores["test11"] = Scoring.score_response([row * 0.985 for row in data_structure[0]], data_structure[0])
                # scores["test12"] = Scoring.score_response([row * 0.98 for row in data_structure[0]], data_structure[0])

                scaled_scores = Scoring.simple_scale_scores(scores)
                stream_id = updated_vm.get_client(request_df.client_uuid).get_stream(request_df.stream_id)

                sorted_scores = sorted(scaled_scores.items(), key=lambda x: x[1], reverse=True)
                winning_scores = sorted_scores[:10]

                weighed_scores = Scoring.weigh_miner_scores(winning_scores)
                weighed_winning_scores = weighed_scores[:10]
                weighed_winning_scores_dict = {score[0]:score[1] for score in weighed_winning_scores}

                print("winning distribution", weighed_winning_scores)
                print("weighed_scores_dict", weighed_winning_scores_dict)

                # add scaled scores
                for miner_uid, scaled_score in scaled_scores.items():
                    stream_miner = stream_id.get_miner(miner_uid)
                    if stream_miner is None:
                        stream_miner = CMWMiner(miner_uid, 0, 0, [])
                        stream_id.add_miner(stream_miner)
                    stream_miner.add_score(scaled_score)
                    if miner_uid in weighed_winning_scores_dict:
                        stream_miner.add_win_value(weighed_winning_scores_dict[miner_uid])
                        stream_miner.add_win()

                for file in request_details.files:
                    os.remove(file)

            print(CMWUtil.dump_cmw(updated_vm))
            ValiUtils.set_vali_memory_and_bkp(CMWUtil.dump_cmw(updated_vm))
