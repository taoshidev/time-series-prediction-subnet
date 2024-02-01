
# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

import hashlib
import os
import uuid
import random
import time

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from data_generator.data_generator_handler import DataGeneratorHandler
from mining_objects.base_mining_model import BaseMiningModel,MiningModelNHITS
from mining_objects.mining_utils import MiningUtils
from time_util.time_util import TimeUtil
from vali_objects.cmw.cmw_objects.cmw_client import CMWClient
from vali_objects.cmw.cmw_objects.cmw_stream_type import CMWStreamType
from vali_objects.cmw.cmw_util import CMWUtil
from vali_objects.dataclasses.client_request import ClientRequest
from vali_objects.exceptions.min_responses_exception import MinResponsesException
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.dataclasses.prediction_data_file import PredictionDataFile
from vali_objects.scoring.scoring import Scoring

import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from mining_objects.base_mining_model import BaseMiningModel,MiningModelNHITS
from vali_config import ValiConfig
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils

import pandas as pd 

def handler_to_model_input_format(ds):  
    '''Prepares the data for the model input
    '''
    df = pd.concat([pd.DataFrame(i) for i in ds],axis=1)
    df.columns = [
        'ds',
        'close',
        'high',
        'low',
        'volume',
    ]
    df.ds= pd.to_datetime(df.ds, unit='ms')
    timestamp = pd.to_datetime(df.ds, unit='ms')
    df['year'] = timestamp.dt.year
    df['month'] = timestamp.dt.month
    df['hour'] = timestamp.dt.hour
    df['minute'] = timestamp.dt.minute
    df['day'] = timestamp.dt.day
    df['dayofweek'] = timestamp.dt.isocalendar().day
    df['weekofyear'] = timestamp.dt.isocalendar().week
    df['quarter'] = timestamp.dt.quarter
    # Add more features as reuired
    df['unique_id'] ='BTCUSD'
    df['y'] = df['close']
    df = df.drop('close',axis=1)
    return df 




import pandas as pd
import numpy as np
from pandas.tseries.offsets import Minute

def prepare_futr_datset(df, num_rows=100):
    # Check if the DataFrame is empty or the 'ds' column doesn't exist
    if df.empty or 'ds' not in df.columns:
        raise ValueError("DataFrame is empty or 'ds' column is not present.")

    # Get the last datetime from the 'ds' column
    last_datetime = df['ds'].iloc[-1]

    # Generate a date range starting from the last datetime plus 5 minutes
    datetimes = pd.date_range(start=last_datetime + Minute(5), periods=num_rows, freq='5T')

    # Create a DataFrame with the new datetimes
    expected_futr_df = pd.DataFrame({'ds': datetimes})
    timestamp = pd.to_datetime(expected_futr_df.ds, unit='ms')
    expected_futr_df['year'] = timestamp.dt.year
    expected_futr_df['month'] = timestamp.dt.month
    expected_futr_df['hour'] = timestamp.dt.hour
    expected_futr_df['minute'] = timestamp.dt.minute
    expected_futr_df['day'] = timestamp.dt.day
    expected_futr_df['dayofweek'] = timestamp.dt.isocalendar().day
    expected_futr_df['weekofyear'] = timestamp.dt.isocalendar().week
    expected_futr_df['quarter'] = timestamp.dt.quarter
    expected_futr_df['unique_id'] = df['unique_id'][0]
    # Set other columns to NaN


    # Append the new DataFrame to the original one
    return expected_futr_df



# import bittensor as bt
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # if you'd like to view the output plotted
    plot_predictions = False

    # if you'd like to view weights over time being set
    plot_weights = True
    weights = []
    historical_weights = {}

    days_processed = []
    curr_iter = 0

    start_iter = []

    totals = {}
    total_weights = {}

    while True:
        try:
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

            iter_add = 601

            data_structure = MiningUtils.get_file(
                "/runnable/historical_financial_data/data.pickle", True)
            data_structure = [data_structure[0][curr_iter:curr_iter + iter_add],
                              data_structure[1][curr_iter:curr_iter + iter_add],
                              data_structure[2][curr_iter:curr_iter + iter_add],
                              data_structure[3][curr_iter:curr_iter + iter_add],
                              data_structure[4][curr_iter:curr_iter + iter_add]]
            print("start", TimeUtil.millis_to_timestamp(data_structure[0][0]))
            print("end", TimeUtil.millis_to_timestamp(data_structure[0][len(data_structure[0]) - 1]))
            start_dt = TimeUtil.millis_to_timestamp(data_structure[0][0])
            end_dt = TimeUtil.millis_to_timestamp(data_structure[0][len(data_structure[0]) - 1])
            curr_iter += iter_add

            data_structure = np.array(data_structure)
            samples = data_structure

            '''
            ========================================================================
            RECOMMEND: define the timestamps you want to test 
            against using the start_dt & end_dt. In this logic I dont use
            the historical BTC data file but you can certainly choose to
            reference it rather than pull historical data from APIs.
            ========================================================================
            '''

            request_uuid = str(uuid.uuid4())
            test_vali_hotkey = str(uuid.uuid4())
            # should be a hash of the vali hotkey & stream type (1 is fine)
            hash_object = hashlib.sha256(client_request.stream_type.encode())
            stream_id = hash_object.hexdigest()
            ts = TimeUtil.minute_in_millis(client_request.prediction_size * 5)

            # data_structure = ValiUtils.get_standardized_ds()
            # data_structure_orig = ValiUtils.get_standardized_ds()
            #
            # data_generator_handler = DataGeneratorHandler()
            # for ts_range in ts_ranges:
            #     data_generator_handler.data_generator_handler(client_request.topic_id, 0,
            #                                                   client_request.additional_details, data_structure, ts_range)
            #
            # vmins, vmaxs, dp_decimal_places, scaled_data_structure = Scaling.scale_ds_with_ts(data_structure)

            # samples = scaled_data_structure

            # forward_proto = Forward(
            #     request_uuid=request_uuid,
            #     stream_id=stream_id,
            #     samples=samples,
            #     topic_id=client_request.topic_id,
            #     feature_ids=client_request.feature_ids,
            #     schema_id=client_request.schema_id,
            #     prediction_size=client_request.prediction_size
            # )

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
            ValiUtils.set_vali_memory_and_bkp(CMWUtil.dump_cmw(vm))

            print("number of predictions needed", client_request.prediction_size)

            '''
            ========================================================================
            Fill in the mining model you want to test here & how it can create
            its dataset for testing.
            ========================================================================
            '''

            mining_models = {
              
                "chaotic_v1_1": {
                    "model_dir": "mining_models/chaotic_v1_1/",
                    "window_size": 100,
                    "id": "chaotic_v1_1",
                    "features": MiningModelNHITS.base_model_dataset,
                    "rows": 601

                },
            }
            # CHANGE THIS 
            for model_name, mining_details in mining_models.items():
                #prep_dataset = mining_details["mining_model"]
                base_mining_model = MiningModelNHITS() \
                    .set_model_dir(mining_details["model_dir"]) \
                    .load_model()

                input = handler_to_model_input_format(data_structure)
                futr = prepare_futr_datset(input,100)
      

                predicted_closes = MiningModelNHITS.predict(df=input,futr=futr)

                predicted_closes = predicted_closes['NHITS'].tolist()[0]
                
                
   

                plot_length = range(len(predicted_closes))

                close_column = data_structure[1].reshape(-1, 1)


                reshaped_predicted_closes = np.array(predicted_closes).reshape(-1, 1)


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
                    predictions=np.array(predicted_closes),
                    prediction_size=client_request.prediction_size,
                    additional_details=client_request.additional_details
                )
                ValiUtils.save_predictions_request(output_uuid, pdf)

            # creating some additional miners who generate noise to compare against
            for a in range(0, 5):
                last_close = samples[1][len(samples[1]) - 1]
                s_predictions = np.array(
                    [random.uniform(last_close - .0001 * last_close,
                                    last_close + .0001 * last_close) for i in
                     range(0, client_request.prediction_size)])

                predictions = s_predictions

                output_uuid = str(uuid.uuid4())
                miner_uuid = str(a)

                pdf = PredictionDataFile(
                    client_uuid=client_request.client_uuid,
                    stream_type=client_request.stream_type,
                    stream_id=stream_id,
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

                data_structure = MiningUtils.get_file(
                    "/runnable/historical_financial_data/data.pickle", True)
                data_structure = [data_structure[0][curr_iter:curr_iter + client_request.prediction_size],
                                  data_structure[1][curr_iter:curr_iter + client_request.prediction_size],
                                  data_structure[2][curr_iter:curr_iter + client_request.prediction_size],
                                  data_structure[3][curr_iter:curr_iter + client_request.prediction_size],
                                  data_structure[4][curr_iter:curr_iter + client_request.prediction_size]]
                print("start", TimeUtil.millis_to_timestamp(data_structure[0][0]))
                print("end", TimeUtil.millis_to_timestamp(data_structure[0][len(data_structure[0]) - 1]))
                start_dt = TimeUtil.millis_to_timestamp(data_structure[0][0])
                end_dt = TimeUtil.millis_to_timestamp(data_structure[0][len(data_structure[0]) - 1])
                # vmins, vmaxs, dp_decimal_places, scaled_data_structure = Scaling.scale_ds_with_ts(data_structure)

                print("number of results gathered: ", len(data_structure[1]))

                scores = {}

                NUM_COLORS = 10
                cm = plt.get_cmap('gist_rainbow')
                colors = [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)]
                x_values = range(len(data_structure[1]))

                color_chosen = 0
                for miner_uid, miner_preds in request_details.predictions.items():
                    if plot_predictions and "miner" in miner_uid:
                        plt.plot(x_values, miner_preds, label=miner_uid, color=colors[color_chosen])
                        color_chosen += 1
                    scores[miner_uid] = Scoring.score_response(miner_preds, data_structure[1])

                print("scores ", scores)
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
                # weighed_winning_scores_dict = {score[0]: score[1] for score in weighed_winning_scores}

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
                    if (curr_iter-1) % 10000 == 0:
                        x_values = range(len(weights))
                        for key, value in historical_weights.items():
                            print(key, sum(value))
                            plt.plot(x_values, value, label=key, color=colors[color_chosen])
                            color_chosen += 1
                        plt.legend()
                        plt.show()

                # for key, score in weighed_winning_scores_dict.items():
                #     if key not in total_weights:
                #         total_weights[key] = 0
                #     total_weights[key] += score

                # print("winning distribution", weighed_scores)
                print("updated weights", weighed_winning_scores_dict)

                # print("updated totals:", totals)
                # print("updated total weights:", total_weights)

                time_now = TimeUtil.now_in_millis()
                for file in request_details.files:
                    os.remove(file)

                curr_iter -= 501

            # end results are stored in the path validation/backups/valiconfig.json (same as it goes on the subnet)
            # ValiUtils.set_vali_memory_and_bkp(CMWUtil.dump_cmw(updated_vm))
        except MinResponsesException as e:
            print(e)
            print("removing files in validation/predictions")
            for file in ValiBkpUtils.get_all_files_in_dir(ValiBkpUtils.get_vali_predictions_dir()):
                os.remove(file)
