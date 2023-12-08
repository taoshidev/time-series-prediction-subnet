
# developer: Taoshidev
# Copyright © 2023 Taoshi, LLC

import os
import uuid
import random
import time

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from mining_objects.base_mining_model import BaseMiningModel
from mining_objects.mining_utils import MiningUtils
from time_util.time_util import TimeUtil
from vali_objects.dataclasses.client_request import ClientRequest
from vali_objects.exceptions.incorrect_prediction_size_error import IncorrectPredictionSizeError
from vali_objects.exceptions.min_responses_exception import MinResponsesException
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.dataclasses.prediction_data_file import PredictionDataFile
from vali_objects.scoring.scoring import Scoring

import matplotlib.pyplot as plt

if __name__ == "__main__":

    '''
    ==========================================================================================
    CONFIG VALUES
    ==========================================================================================
    '''
    # if you'd like to view the output of predictions plotted
    plot_predictions = False
    # if you'd like to view weights over time being set
    plot_weights = True
    # if you want to include noise miners
    noise_miners = True
    # number of noise miners which will be used if noise_miners is true
    number_of_noise_miners = 5

    # starting position in the miner_training file where testing should begin
    curr_iter = 0
    # the number of rows to add to each iteration of testing. The minimum value should be the
    # minimum "rows" value from the mining_models object
    iter_add = 601
    # reducing from the iter_add at the end to ensure you dont skip any windows of predictions
    # this is used post processing of each line.
    # for example, we'll add 601 to the curr_iter from iter_add, this will create a prediction window
    # from the 601st to 701st row (100 predictions) we'll then reduce curr_iter by 501 at the end
    # to make sure the next run gets the 701st to 801st rows for predictions.
    # if set to 0 then we'll get predictions at the 601st to 701st window, then 1202nd to 1203rd window
    # and we'll end up skipping prediction windows
    iter_add_reduction = 501

    # global variables holding
    weights = []
    historical_weights = {}

    '''
    ==========================================================================================
    Add any models you want to test or test against here.
    
    model dir - name of the model in mining_models dir
    window size - based on training the lookback used by the model
    id - a unique identifier for the model
    features - the feature dataset used to train the model
    prediction fx - the function you want to leverage to generate your prediction
    rows - the number of rows used to scale the data (similar to training)
    
    ==========================================================================================
    '''

    mining_models = {
        "1": {
            "model_dir": "model_v4_1.h5",
            "window_size": 100,
            "id": 1,
            "features": BaseMiningModel.base_model_dataset,
            "prediction_fx": MiningUtils.open_model_v4_prediction_generation,
            "rows": 601
        },
        "2": {
            "model_dir": "model_v4_2.h5",
            "window_size": 500,
            "id": 2,
            "features": BaseMiningModel.base_model_dataset,
            "prediction_fx": MiningUtils.open_model_v4_prediction_generation,
            "rows": 601
        },
        "3": {
            "model_dir": "model_v4_3.h5",
            "window_size": 100,
            "id": 3,
            "features": BaseMiningModel.base_model_dataset,
            "prediction_fx": MiningUtils.open_model_v4_prediction_generation,
            "rows": 601
        },
        "4": {
            "model_dir": "model_v4_4.h5",
            "window_size": 100,
            "id": 4,
            "features": BaseMiningModel.base_model_dataset,
            "prediction_fx": MiningUtils.open_model_v4_prediction_generation,
            "rows": 601
        },
        "5": {
            "model_dir": "model_v4_5.h5",
            "window_size": 100,
            "id": 5,
            "features": BaseMiningModel.base_model_dataset,
            "prediction_fx": MiningUtils.open_model_v4_prediction_generation,
            "rows": 601
        },
        "6": {
            "model_dir": "model_v4_6.h5",
            "window_size": 100,
            "id": 6,
            "features": BaseMiningModel.base_model_dataset,
            "prediction_fx": MiningUtils.open_model_v4_prediction_generation,
            "rows": 601
        },
    }

    # standardized client request to reference in some of the logic. Should remain untouched.
    client_request = ClientRequest(
        client_uuid="test_client_uuid",
        stream_type="BTCUSD-5m",
        topic_id=1,
        schema_id=1,
        feature_ids=[0.001, 0.002, 0.003, 0.004],
        prediction_size=100,
        additional_details={
            "tf": 5,
            "trade_pair": "BTCUSD"
        }
    )

    # check on the iter_add following the rules
    for id, model in mining_models.items():
        if model["rows"] > iter_add:
            raise Exception("you need to increase the size of iter_add to be the minimum from the mining_models")

    # used to calculate the total summed weights set over the testing period
    totals = {}
    total_weights = {}

    # this should be a precreated training file thats then referencable here
    # you can make it by using generate_historical_data.py and then changing the name of the output file
    miner_training_ds = MiningUtils.get_file(
        "/runnable/historical_financial_data/data_testing.pickle", True)
    length_of_ds = len(miner_training_ds[0]) - curr_iter
    print("length of miner training file provided", length_of_ds)
    # need to stop once before the end of iters to ensure that data is available for the window
    iterations = int(length_of_ds / (iter_add - iter_add_reduction)) - int(iter_add / (iter_add - iter_add_reduction))

    for iteration in range(iterations):
        data_structure = ValiUtils.get_standardized_ds()
        for i in range(len(miner_training_ds)):
            data_structure[i] = miner_training_ds[i][curr_iter:curr_iter + iter_add]
        print("start", TimeUtil.millis_to_timestamp(data_structure[0][0]))
        print("end", TimeUtil.millis_to_timestamp(data_structure[0][len(data_structure[0]) - 1]))
        try:
            start_dt = TimeUtil.millis_to_timestamp(data_structure[0][0])
            end_dt = TimeUtil.millis_to_timestamp(data_structure[0][len(data_structure[0]) - 1])
            curr_iter += iter_add

            data_structure = np.array(data_structure)
            samples = data_structure

            # every vali request has a designated request uuid to be able to reference later
            request_uuid = str(uuid.uuid4())
            ts = TimeUtil.minute_in_millis(client_request.prediction_size * 5)

            print("----- all miners making predictions on closes -----")

            for model_name, mining_details in mining_models.items():

                '''
                ==========================================================================================
                
                leveraging your defined function for how you want to predict closes, defined in 
                prediction_fx

                ==========================================================================================
                '''

                predicted_closes = mining_details["prediction_fx"](samples,
                                                                mining_details,
                                                                client_request.prediction_size)

                output_uuid = str(uuid.uuid4())
                miner_uuid = "miner" + str(mining_details["id"])

                pdf = PredictionDataFile(
                    client_uuid=client_request.client_uuid,
                    stream_type=client_request.stream_type,
                    stream_id=client_request.stream_type,
                    topic_id=client_request.topic_id,
                    request_uuid=request_uuid,
                    miner_uid=miner_uuid,
                    start=TimeUtil.timestamp_to_millis(end_dt),
                    end=TimeUtil.timestamp_to_millis(end_dt) + ts,
                    predictions=np.array(predicted_closes),
                    prediction_size=client_request.prediction_size,
                    additional_details=client_request.additional_details
                )
                ValiUtils.save_predictions_request(output_uuid, pdf)

            # will create some additional miners who generate noise to compare against if flag is set
            if noise_miners:
                for a in range(number_of_noise_miners):
                    last_close = samples[1][len(samples[1]) - 1]
                    s_predictions = np.array(
                        [random.uniform(last_close - .00025 * last_close,
                                        last_close + .00025 * last_close) for i in
                         range(0, client_request.prediction_size)])

                    predictions = s_predictions

                    output_uuid = str(uuid.uuid4())
                    miner_uuid = str(a)

                    pdf = PredictionDataFile(
                        client_uuid=client_request.client_uuid,
                        stream_type=client_request.stream_type,
                        stream_id=client_request.stream_type,
                        topic_id=client_request.topic_id,
                        request_uuid=request_uuid,
                        miner_uid=miner_uuid,
                        start=TimeUtil.timestamp_to_millis(end_dt),
                        end=TimeUtil.timestamp_to_millis(end_dt) + ts,
                        predictions=np.array(predictions),
                        prediction_size=client_request.prediction_size,
                        additional_details=client_request.additional_details
                    )
                    ValiUtils.save_predictions_request(output_uuid, pdf)

            '''
            ==========================================================================================
            Validator logic to compare results against actual closes
            ==========================================================================================
            '''
            print("----- core validation logic on requests that are ready to be reviewed -----")

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
                for i in range(len(miner_training_ds)):
                    data_structure[i] = miner_training_ds[i][curr_iter:curr_iter + client_request.prediction_size]
                print("start", TimeUtil.millis_to_timestamp(data_structure[0][0]))
                print("end", TimeUtil.millis_to_timestamp(data_structure[0][len(data_structure[0]) - 1]))
                start_dt = TimeUtil.millis_to_timestamp(data_structure[0][0])
                end_dt = TimeUtil.millis_to_timestamp(data_structure[0][len(data_structure[0]) - 1])

                print("number of results gathered", len(data_structure[1]))

                scores = {}

                NUM_COLORS = len(mining_models) + number_of_noise_miners
                cm = plt.get_cmap('gist_rainbow')
                colors = [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)]
                x_values = range(len(data_structure[1]))

                color_chosen = 0
                for miner_uid, miner_preds in request_details.predictions.items():
                    if plot_predictions and "miner" in miner_uid:
                        plt.plot(x_values, miner_preds, label=miner_uid, color=colors[color_chosen])
                        color_chosen += 1
                    scores[miner_uid] = Scoring.score_response(miner_preds, data_structure[1])

                # lower scores are better with the decaying RMSE scores
                print("decaying RMSE scores", scores)

                # plot if the flag is set
                if plot_predictions:
                    plt.plot(x_values, data_structure[1], label="results", color=colors[color_chosen])
                    plt.legend()
                    plt.show()

                scaled_scores = Scoring.simple_scale_scores(scores)
                stream_id = updated_vm.get_client(request_df.client_uuid).get_stream(request_df.stream_id)

                sorted_scores = sorted(scaled_scores.items(), key=lambda x: x[1], reverse=True)
                winning_scores = sorted_scores

                weighed_scores = Scoring.weigh_miner_scores(winning_scores)
                weighed_winning_scores_dict, weight = Scoring.update_weights_using_historical_distributions(weighed_scores, data_structure)

                for key, score in scores.items():
                    if key not in totals:
                        totals[key] = 0
                    totals[key] += score

                if plot_weights:
                    for key, value in weighed_winning_scores_dict.items():
                        if key not in historical_weights:
                            historical_weights[key] = []
                        historical_weights[key].append(value)

                    weights.append(weight)

                    print("curr iter", curr_iter)
                    if iteration == iterations-1:
                        x_values = range(len(weights))
                        for key, value in historical_weights.items():
                            print(key, sum(value))
                            plt.plot(x_values, value, label=key, color=colors[color_chosen])
                            color_chosen += 1
                        plt.legend()
                        plt.show()
                else:
                    if iteration == iterations-1:
                        x_values = range(len(weights))
                        # bigger is better
                        print("outputting final weights set over the training period")
                        for key, value in historical_weights.items():
                            print(key, sum(value))

                # bigger is better, this is for the current iteration
                print("weights for each miner", weighed_winning_scores_dict)

                # remove files once processed
                time_now = TimeUtil.now_in_millis()
                for file in request_details.files:
                    os.remove(file)

                # reduce from the current iteration if you want to ensure you dont skip any prediction windows
                curr_iter -= iter_add_reduction

        except (MinResponsesException, IncorrectPredictionSizeError) as e:
            print(e)
            print("removing files in validation/predictions")
            for file in ValiBkpUtils.get_all_files_in_dir(ValiBkpUtils.get_vali_predictions_dir()):
                os.remove(file)
