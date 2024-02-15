# developer: taoshi-mbrown
# Copyright © 2024 Taoshi Inc
import hashlib
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from features import FeatureCollector
from mining_objects.base_mining_model import BaseMiningModel
import os
from neurons.miner import get_predictions
from streams.btcusd_5m import (
    INTERVAL_MS,
    model_feature_sources,
    prediction_feature_ids,
    PREDICTION_COUNT,
    PREDICTION_LENGTH,
    SAMPLE_COUNT,
    legacy_model_feature_scaler,
    legacy_model_feature_sources,
    legacy_model_feature_ids,
    validator_feature_source,
)
from streams.btcusd_5m import model_feature_scaler as new_model_feature_scaler
from streams.btcusd_5m import model_feature_ids as new_model_feature_ids
import time
from time_util import datetime, time_span_ms
from time_util.time_util import TimeUtil
import uuid

from vali_config import ValiConfig
from vali_objects.cmw.cmw_objects.cmw_client import CMWClient
from vali_objects.cmw.cmw_objects.cmw_stream_type import CMWStreamType
from vali_objects.cmw.cmw_util import CMWUtil
from vali_objects.dataclasses.client_request import ClientRequest
from vali_objects.exceptions.min_responses_exception import MinResponsesException
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.dataclasses.prediction_data_file import PredictionDataFile
from vali_objects.scoring.scoring import Scoring


if __name__ == "__main__":
    TESTING_LOOKBACK_DAYS = 30
    PREDICTION_LENGTH_MS = INTERVAL_MS * PREDICTION_LENGTH

    GRAPH_SERIES_COLORS = list(TABLEAU_COLORS.values())
    PLOT_WEIGHT_ITERATIONS = 20

    now = datetime.now()
    now_time_ms = now.timestamp_ms()

    testing_start_time_ms = now_time_ms - time_span_ms(days=TESTING_LOOKBACK_DAYS)
    testing_end_time_ms = now_time_ms - PREDICTION_LENGTH_MS

    # if you'd like to view the output plotted
    plot_predictions = True

    # if you'd like to view weights over time being set
    plot_weights = True
    weights = []
    historical_weights = {}

    days_processed = []
    plot_weight_iteration = 0

    start_iter = []

    totals = {}
    total_weights = {}

    legacy_feature_source = FeatureCollector(
        sources=legacy_model_feature_sources,
        feature_ids=legacy_model_feature_ids,
    )

    new_feature_source = FeatureCollector(
        sources=model_feature_sources,
        feature_ids=new_model_feature_ids,
    )

    """
    ========================================================================
    Fill in the mining model you want to test here
    ========================================================================
    """

    base_mining_models = {
        "model_v4_1": {
            "filename": "model_v4_1.h5",
            "sample_count": 100,
            "id": "model2308",
            "prediction_count": 1,
        },
        "model_v4_2": {
            "filename": "model_v4_2.h5",
            "sample_count": 500,
            "id": "model3005",
            "prediction_count": 1,
        },
        "model_v4_3": {
            "filename": "model_v4_3.h5",
            "sample_count": 100,
            "id": "model3103",
            "prediction_count": 1,
        },
        "model_v4_4": {
            "filename": "model_v4_4.h5",
            "sample_count": 100,
            "id": "model3104",
            "prediction_count": 1,
        },
        "model_v4_5": {
            "filename": "model_v4_5.h5",
            "sample_count": 100,
            "id": "model3105",
            "prediction_count": 1,
        },
        "model_v4_6": {
            "id": "model3106",
            "filename": "model_v4_6.h5",
            "sample_count": 100,
            "prediction_count": 1,
        },
        "model_v5_1": {
            "id": "model5000",
            "filename": "model_v5_1.h5",
            "sample_count": SAMPLE_COUNT,
            "prediction_count": PREDICTION_COUNT,
            "legacy_model": False,
        },
    }

    stream_mining_solutions = []

    for model_name, mining_details in base_mining_models.items():
        legacy = mining_details.get("legacy_model", True)
        if legacy:
            model_feature_ids = legacy_model_feature_ids
            model_feature_source = legacy_feature_source
            model_feature_scaler = legacy_model_feature_scaler
        else:
            model_feature_ids = new_model_feature_ids
            model_feature_source = new_feature_source
            model_feature_scaler = new_model_feature_scaler

        model_filename = (
            ValiConfig.BASE_DIR + "/mining_models/" + mining_details["filename"]
        )

        model = BaseMiningModel(
            filename=model_filename,
            mode="r",
            feature_count=len(model_feature_ids),
            sample_count=mining_details["sample_count"],
            prediction_feature_count=len(prediction_feature_ids),
            prediction_count=mining_details["prediction_count"],
            prediction_length=PREDICTION_LENGTH,
        )

        stream_mining_solutions.append(
            {
                "id": mining_details["id"],
                "model": model,
                "feature_source": model_feature_source,
                "feature_scaler": model_feature_scaler,
            }
        )

    start_time_ms = testing_start_time_ms
    while start_time_ms < testing_end_time_ms:
        try:
            client_request = ClientRequest(
                client_uuid="test_client_uuid",
                stream_type="BTCUSD-5m",
                topic_id=1,
                schema_id=1,
                feature_ids=[0.001, 0.002, 0.003, 0.004],
                prediction_size=100,
                additional_details={"tf": 5, "trade_pair": "BTCUSD"},
            )

            """
            ========================================================================
            RECOMMEND: define the timestamps you want to test 
            against using the start_dt & end_dt. In this logic I dont use
            the historical BTC data file but you can certainly choose to
            reference it rather than pull historical data from APIs.
            ========================================================================
            """

            request_uuid = str(uuid.uuid4())
            test_vali_hotkey = str(uuid.uuid4())
            # should be a hash of the vali hotkey & stream type (1 is fine)
            hash_object = hashlib.sha256(client_request.stream_type.encode())
            stream_id = hash_object.hexdigest()

            vm = ValiUtils.get_vali_records()
            client = vm.get_client(client_request.client_uuid)
            if client is None:
                print("client doesnt exist")
                cmw_client = CMWClient().set_client_uuid(client_request.client_uuid)
                cmw_client.add_stream(
                    CMWStreamType()
                    .set_stream_id(stream_id)
                    .set_topic_id(client_request.topic_id)
                )
                vm.add_client(cmw_client)
            else:
                client_stream_type = client.get_stream(stream_id)
                if client_stream_type is None:
                    client.add_stream(
                        CMWStreamType()
                        .set_stream_id(stream_id)
                        .set_topic_id(client_request.topic_id)
                    )
            ValiUtils.set_vali_memory_and_bkp(CMWUtil.dump_cmw(vm))

            print("number of predictions needed", client_request.prediction_size)

            for stream_mining_solution in stream_mining_solutions:
                model = stream_mining_solution["model"]
                feature_source = stream_mining_solution["feature_source"]
                feature_scaler = stream_mining_solution["feature_scaler"]

                prediction_array = get_predictions(
                    start_time_ms,
                    feature_source,
                    feature_scaler,
                    model,
                )

                predicted_closes = prediction_array.flatten()

                output_uuid = str(uuid.uuid4())
                miner_uuid = "miner" + str(stream_mining_solution["id"])
                # just the vali hotkey for now

                pdf = PredictionDataFile(
                    client_uuid=client_request.client_uuid,
                    stream_type=client_request.stream_type,
                    stream_id=stream_id,
                    topic_id=client_request.topic_id,
                    request_uuid=request_uuid,
                    miner_uid=miner_uuid,
                    start=start_time_ms,
                    end=start_time_ms + PREDICTION_LENGTH_MS,
                    predictions=predicted_closes,
                    prediction_size=client_request.prediction_size,
                    additional_details=client_request.additional_details,
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

                feature_samples = validator_feature_source.get_feature_samples(
                    request_df.start, INTERVAL_MS, PREDICTION_LENGTH
                )

                validation_array = validator_feature_source.feature_samples_to_array(
                    feature_samples, prediction_feature_ids
                )

                validation_array = validation_array.flatten()

                scores = {}

                x_values = range(len(validation_array))
                graph_series_index = 0
                for miner_uid, miner_preds in request_details.predictions.items():
                    if plot_predictions and "miner" in miner_uid:
                        plt.plot(
                            x_values,
                            miner_preds,
                            label=miner_uid,
                            color=GRAPH_SERIES_COLORS[graph_series_index],
                        )
                        graph_series_index += 1
                    scores[miner_uid] = Scoring.score_response(
                        miner_preds, validation_array
                    )

                print("scores ", scores)
                if plot_predictions:
                    plt.plot(
                        x_values,
                        validation_array,
                        label="results",
                        color="k",
                    )

                    plt.legend()
                    plt.show()

                scaled_scores = Scoring.simple_scale_scores(scores)
                stream_id = updated_vm.get_client(request_df.client_uuid).get_stream(
                    request_df.stream_id
                )

                sorted_scores = sorted(
                    scaled_scores.items(), key=lambda x: x[1], reverse=True
                )
                winning_scores = sorted_scores

                weighed_scores = Scoring.weigh_miner_scores(winning_scores)
                (
                    weighed_winning_scores_dict,
                    weight,
                ) = Scoring.update_weights_using_historical_distributions(
                    weighed_scores, validation_array
                )
                # weighed_winning_scores_dict = {score[0]: score[1] for score in weighed_winning_scores}

                for key, score in scores.items():
                    if key not in totals:
                        totals[key] = 0
                    totals[key] += score

                if plot_weights:
                    plot_weight_iteration += 1

                    for key, value in weighed_winning_scores_dict.items():
                        if key not in historical_weights:
                            historical_weights[key] = []
                        historical_weights[key].append(value)

                    weights.append(weight)

                    if (plot_weight_iteration % PLOT_WEIGHT_ITERATIONS) == 0:
                        x_values = range(len(weights))
                        graph_series_index = 0
                        for key, value in historical_weights.items():
                            print(key, sum(value))
                            plt.plot(
                                x_values,
                                value,
                                label=key,
                                color=GRAPH_SERIES_COLORS[graph_series_index],
                            )
                            graph_series_index += 1
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

                start_time_ms += INTERVAL_MS

            # end results are stored in the path validation/backups/valiconfig.json (same as it goes on the subnet)
            # ValiUtils.set_vali_memory_and_bkp(CMWUtil.dump_cmw(updated_vm))
        except MinResponsesException as e:
            print(e)
            print("removing files in validation/predictions")
            for file in ValiBkpUtils.get_all_files_in_dir(
                ValiBkpUtils.get_vali_predictions_dir()
            ):
                os.remove(file)
