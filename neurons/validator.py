# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Taoshidev
# Copyright © 2023 Taoshi Inc
import argparse
import bittensor as bt
from hashing_utils import HashingUtils
from miner_config import MinerConfig
import numpy as np
import os
from scipy.stats import yeojohnson
from streams.btcusd_5m import (
    INTERVAL_MS,
    prediction_feature_ids,
    validator_feature_source,
)
from template.protocol import (
    LiveForward,
    LiveBackward,
    LiveForwardHash,
)
import time
from time_util import datetime
from time_util.time_util import TimeUtil
import traceback
import uuid
from vali_config import ValiConfig
from vali_objects.cmw.cmw_objects.cmw import CMW
from vali_objects.cmw.cmw_objects.cmw_client import CMWClient
from vali_objects.cmw.cmw_objects.cmw_miner import CMWMiner
from vali_objects.cmw.cmw_objects.cmw_stream_type import CMWStreamType
from vali_objects.cmw.cmw_util import CMWUtil
from vali_objects.dataclasses.base_objects.base_request_dataclass import (
    BaseRequestDataClass,
)
from vali_objects.dataclasses.client_request import ClientRequest
from vali_objects.dataclasses.prediction_data_file import PredictionDataFile
from vali_objects.dataclasses.prediction_request import PredictionRequest
from vali_objects.exceptions.incorrect_live_results_count_exception import (
    IncorrectLiveResultsCountException,
)
from vali_objects.exceptions.incorrect_prediction_size_error import (
    IncorrectPredictionSizeError,
)
from vali_objects.exceptions.min_responses_exception import MinResponsesException
from vali_objects.scaling.scaling import Scaling
from vali_objects.scoring.scoring import Scoring
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils


def get_config():
    parser = argparse.ArgumentParser()
    # TODO(developer): Adds your custom validator arguments to the parser.
    parser.add_argument(
        "--test_only_historical",
        default=0,
        help="if you only want to pull in " "historical data for testing.",
    )
    parser.add_argument(
        "--continuous_data_feed",
        default=0,
        help="this will have the validator ping every 5 mins "
        "for updated predictions",
    )
    # Adds override arguments for network and netuid.
    parser.add_argument("--netuid", type=int, default=1, help="The chain subnet uid.")
    # Adds subtensor specific arguments i.e. --subtensor.chain_endpoint ... --subtensor.network ...
    bt.subtensor.add_args(parser)
    # Adds logging specific arguments i.e. --logging.debug ..., --logging.trace .. or --logging.logging_dir ...
    bt.logging.add_args(parser)
    # Adds wallet specific arguments i.e. --wallet.name ..., --wallet.hotkey ./. or --wallet.path ...
    bt.wallet.add_args(parser)
    # Parse the config (will take command-line arguments if provided)
    # To print help message, run python3 template/miner.py --help
    config = bt.config(parser)

    # Step 3: Set up logging directory
    # Logging is crucial for monitoring and debugging purposes.
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            "validator",
        )
    )
    # Ensure the logging directory exists.
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)

    # Return the parsed config.
    return config


def run_time_series_validation(
    wallet, config, metagraph, vali_requests: list[BaseRequestDataClass]
):
    # Set up initial scoring weights for validation
    # bt.logging.info("Building validation weights.")
    # scores = torch.ones_like(metagraph.S, dtype=torch.float32)
    # bt.logging.info(f"Weights: {scores}")

    pred_metagraph_hotkeys = []

    for vali_request in vali_requests:
        # standardized request identifier for miners to tie together forward/backprop
        request_uuid = str(uuid.uuid4())

        if isinstance(vali_request, ClientRequest):
            # stream_type = hash(str(vali_request.stream_type) + wallet.hotkey.ss58_address)
            # stream_type = hash(str(vali_request.stream_type))
            # hash_object = hashlib.sha256(vali_request.stream_type.encode())
            # stream_type = hash_object.hexdigest()

            stream_type = vali_request.stream_type

            if vali_request.client_uuid is None:
                vali_request.client_uuid = wallet.hotkey.ss58_address

            live_hash_proto = LiveForwardHash(
                request_uuid=request_uuid,
                stream_id=stream_type,
                topic_id=vali_request.topic_id,
                feature_ids=vali_request.feature_ids,
                schema_id=vali_request.schema_id,
                prediction_size=vali_request.prediction_size,
            )

            live_proto = LiveForward(
                request_uuid=request_uuid,
                stream_id=stream_type,
                topic_id=vali_request.topic_id,
                feature_ids=vali_request.feature_ids,
                schema_id=vali_request.schema_id,
                prediction_size=vali_request.prediction_size,
            )

            try:
                hashed_responses = dendrite.query(
                    metagraph.axons, live_hash_proto, deserialize=True, timeout=30
                )

                # wait to allow sending at correct expected intervals
                time.sleep(60)

                responses = dendrite.query(
                    metagraph.axons, live_proto, deserialize=True, timeout=30
                )

                # # check to see # of responses
                bt.logging.info(
                    f"number of responses to requested data: [{len(responses)}]"
                )

                # FOR DEBUG PURPOSES
                # for i, respi in enumerate(responses):
                #     predictions = respi.predictions.numpy()
                #     print(predictions)
                #     if respi is not None \
                #             and len(respi) == vali_request.prediction_size:
                #         bt.logging.debug(f"index [{i}] number of responses to requested data [{len(respi)}]")
                #     else:
                #         bt.logging.debug(f"index [{i}] has no proper response")
                end_dt = TimeUtil.generate_start_timestamp(0)

                prediction_start_time = TimeUtil.timestamp_to_millis(end_dt)
                prediction_end_time = TimeUtil.timestamp_to_millis(
                    end_dt
                ) + TimeUtil.minute_in_millis(
                    vali_request.prediction_size * vali_request.additional_details["tf"]
                )

                bt.logging.debug(
                    f"prediction start time [{prediction_start_time}], [{TimeUtil.millis_to_timestamp(prediction_start_time)}]"
                )
                bt.logging.debug(
                    f"prediction end time [{prediction_end_time}], [{TimeUtil.millis_to_timestamp(prediction_end_time)}]"
                )

                for i, resp_i in enumerate(responses):
                    if resp_i.predictions is not None:
                        miner_hotkey = metagraph.axons[i].hotkey
                        try:
                            predictions = resp_i.predictions.numpy()
                            hashed_predictions = HashingUtils.hash_predictions(
                                miner_hotkey, str(predictions.tolist())
                            )
                            miner_provided_hashed_predictions = hashed_responses[
                                i
                            ].hashed_predictions

                            if (
                                len(predictions) == vali_request.prediction_size
                                and len(predictions.shape) == 1
                            ):
                                if (
                                    hashed_predictions
                                    == miner_provided_hashed_predictions
                                ):
                                    pred_metagraph_hotkeys.append(miner_hotkey)
                                    # for file name
                                    output_uuid = str(uuid.uuid4())
                                    bt.logging.debug(
                                        f"axon hotkey has correctly responded: [{miner_hotkey}]"
                                    )

                                    # has the right number of predictions made
                                    pdf = PredictionDataFile(
                                        client_uuid=vali_request.client_uuid,
                                        stream_type=vali_request.stream_type,
                                        stream_id=stream_type,
                                        topic_id=vali_request.topic_id,
                                        request_uuid=request_uuid,
                                        miner_uid=miner_hotkey,
                                        start=prediction_start_time,
                                        end=prediction_end_time,
                                        predictions=predictions,
                                        prediction_size=vali_request.prediction_size,
                                        additional_details=vali_request.additional_details,
                                    )
                                    ValiUtils.save_predictions_request(output_uuid, pdf)
                                else:
                                    bt.logging.debug(
                                        f"incorrect match between "
                                        f"hashed predictions [{miner_provided_hashed_predictions}] "
                                        f"and predictions provided [{hashed_predictions}] "
                                        f"for miner [{miner_hotkey}]"
                                    )
                            else:
                                bt.logging.debug(
                                    f"incorrectly provided "
                                    f"prediction size [{len(predictions)}] "
                                    f"or shape [{len(predictions.shape)}] "
                                    f"for miner [{miner_hotkey}]"
                                )
                        except Exception:  # noqa
                            bt.logging.debug(
                                f"not correctly configured predictions: [{miner_hotkey}]"
                            )
                            continue
                bt.logging.info(
                    f"all hotkeys of accurately formatted predictions received {pred_metagraph_hotkeys}"
                )
                bt.logging.info("completed storing all predictions")

            # If we encounter an unexpected error, log it for debugging.
            except RuntimeError as e:
                bt.logging.error(e)
                traceback.print_exc()

        elif isinstance(vali_request, PredictionRequest):
            bt.logging.info("processing predictions ready to be weighed")
            # handle results ready to score and weigh
            request_df = vali_request.df
            stream_type = request_df.stream_type
            try:
                bt.logging.info("getting results from live predictions")

                start_time_ms = request_df.start
                sample_count = request_df.prediction_size

                bt.logging.debug(
                    f"requested results start: [{datetime.fromtimestamp_ms(start_time_ms)}]"
                )
                bt.logging.debug(
                    f"requested results end: [{datetime.fromtimestamp_ms(request_df.end)}]"
                )

                feature_samples = validator_feature_source.get_feature_samples(
                    start_time_ms, INTERVAL_MS, sample_count
                )

                validation_array = validator_feature_source.feature_samples_to_array(
                    feature_samples, prediction_feature_ids
                )

                # TODO: Improve validators to allow multiple features in predictions
                validation_array = validation_array.flatten()

                bt.logging.info(
                    "results gathered sending back to miners via backprop and weighing"
                )

                # Send back the results for backprop so miners can learn
                results = bt.tensor(validation_array)

                results_backprop_proto = LiveBackward(
                    request_uuid=request_uuid,
                    stream_id=stream_type,
                    samples=results,
                    topic_id=request_df.topic_id,
                )

                try:
                    dendrite.query(
                        metagraph.axons, results_backprop_proto, deserialize=True
                    )
                    bt.logging.info("live results sent back to miners")
                except Exception:  # noqa
                    traceback.print_exc()
                    bt.logging.info(
                        "failed sending back results to miners and continuing..."
                    )

                scores = {}
                for miner_uid, miner_preds in vali_request.predictions.items():
                    try:
                        scores[miner_uid] = Scoring.score_response(
                            miner_preds, validation_array
                        )
                    except IncorrectPredictionSizeError as e:
                        bt.logging.error(e)
                        traceback.print_exc()

                if len(scores) > 0:
                    bt.logging.debug(f"unscaled scores [{scores}]")
                    scores_list = np.array(
                        [score for miner_uid, score in scores.items()]
                    )
                    variance = np.var(scores_list)

                    if variance == 0:
                        bt.logging.debug(
                            "homogenous dataset, going to equally distribute scores"
                        )
                        weighed_scores = [
                            (miner_uid, 1 / len(scores))
                            for miner_uid, score in scores.items()
                        ]
                        bt.logging.debug(f"weighed scores [{weighed_scores}]")
                        (
                            weighed_winning_scores_dict,
                            weight,
                        ) = Scoring.update_weights_using_historical_distributions(
                            weighed_scores, validation_array
                        )

                    else:
                        scaled_scores = Scoring.simple_scale_scores(scores)

                        # store weights for results
                        sorted_scores = sorted(
                            scaled_scores.items(), key=lambda x: x[1], reverse=True
                        )
                        winning_scores = sorted_scores

                        # choose top 10
                        weighed_scores = Scoring.weigh_miner_scores(winning_scores)

                        bt.logging.debug(f"weighed scores [{weighed_scores}]")
                        bt.logging.debug(f"validation array [{validation_array}]")
                        (
                            weighed_winning_scores_dict,
                            weight,
                        ) = Scoring.update_weights_using_historical_distributions(
                            weighed_scores, validation_array
                        )
                        # weighed_winning_scores_dict = {score[0]: score[1] for score in weighed_winning_scores}

                        bt.logging.debug(f"weight for the predictions: [{weight}]")
                        bt.logging.debug(f"scaled scores: [{scaled_scores}]")
                        bt.logging.debug(
                            f"weighed winning scores: [{weighed_winning_scores_dict}]"
                        )

                    values_list = np.array(
                        [v for k, v in weighed_winning_scores_dict.items()]
                    )

                    mean = np.mean(values_list)
                    std_dev = np.std(values_list)

                    lower_bound = mean - 3 * std_dev
                    bt.logging.debug(f"scores lower bound: [{lower_bound}]")

                    if lower_bound < 0:
                        lower_bound = 0

                    filtered_results = [
                        (k, v)
                        for k, v in weighed_winning_scores_dict.items()
                        if lower_bound < v
                    ]
                    filtered_scores = np.array([x[1] for x in filtered_results])

                    # Normalize the list using Z-score normalization
                    transformed_results = yeojohnson(filtered_scores, lmbda=500)
                    scaled_transformed_list = Scaling.min_max_scalar_list(
                        transformed_results
                    )
                    filtered_winning_scores_dict = {
                        filtered_results[i][0]: scaled_transformed_list[i]
                        for i in range(len(filtered_results))
                    }

                    bt.logging.debug(
                        f"filtered weighed winning scores: [{filtered_winning_scores_dict}]"
                    )

                    # bt.logging.debug(f"finalized weighed winning scores [{weighed_winning_scores}]")
                    weights = []
                    converted_uids = []

                    deregistered_mineruids = []

                    for (
                        miner_uid,
                        weighed_winning_score,
                    ) in filtered_winning_scores_dict.items():
                        try:
                            converted_uids.append(
                                metagraph.uids[metagraph.hotkeys.index(miner_uid)]
                            )
                            weights.append(weighed_winning_score)
                        except Exception:  # noqa
                            deregistered_mineruids.append(miner_uid)
                            bt.logging.info(
                                f"not able to find miner hotkey, "
                                f"likely deregistered [{miner_uid}]"
                            )

                    Scoring.update_weights_remove_deregistrations(
                        deregistered_mineruids
                    )

                    bt.logging.debug(f"converted uids [{converted_uids}]")
                    bt.logging.debug(f"weights gathered [{weights}]")

                    result = subtensor.set_weights(
                        netuid=config.netuid,  # Subnet to set weights on.
                        wallet=wallet,  # Wallet to sign set weights using hotkey.
                        uids=converted_uids,  # Uids of the miners to set weights for.
                        weights=weights,  # Weights to set for the miners.
                        # wait_for_inclusion=True,
                    )
                    if result:
                        bt.logging.success("Successfully set weights.")
                        bt.logging.info("removing processed files")
                        # remove files that have been properly processed & weighed
                        for file in vali_request.files:
                            os.remove(file)
                        bt.logging.info(
                            f"removed [{len(vali_request.files)}] processed files"
                        )

                    else:
                        bt.logging.error("Failed to set weights.")
                    bt.logging.info("weights set and stored")
                    bt.logging.info("adding to cmw")

                    time_now = TimeUtil.now_in_millis()
                    try:
                        new_cmw = CMW()
                        cmw_client = CMWClient().set_client_uuid(request_df.client_uuid)
                        cmw_client.add_stream(
                            CMWStreamType()
                            .set_stream_id(stream_type)
                            .set_topic_id(request_df.topic_id)
                        )
                        new_cmw.add_client(cmw_client)
                        cmw_client.add_stream(
                            CMWStreamType()
                            .set_stream_id(stream_type)
                            .set_topic_id(request_df.topic_id)
                        )
                        stream = cmw_client.get_stream(stream_type)
                        for miner_uid, score in scores.items():
                            bt.logging.debug(f"adding mineruid [{miner_uid}]")
                            stream_miner = CMWMiner(miner_uid)
                            stream.add_miner(stream_miner)
                            bt.logging.debug("miner added")
                            stream_miner.add_unscaled_score(
                                [time_now, scores[miner_uid]]
                            )
                            if miner_uid in weighed_winning_scores_dict:
                                if weighed_winning_scores_dict[miner_uid] != 0:
                                    bt.logging.debug(
                                        f"adding winning miner [{miner_uid}]"
                                    )
                                    stream_miner.add_win_score(
                                        [
                                            time_now,
                                            weighed_winning_scores_dict[miner_uid],
                                        ]
                                    )
                        ValiUtils.save_cmw_results(
                            request_df.request_uuid, CMWUtil.dump_cmw(new_cmw)
                        )
                        bt.logging.info("cmw saved: ", request_df.request_uuid)
                    except Exception as e:
                        # if fail to store cmw for some reason print & continue
                        bt.logging.error(e)
                        traceback.print_exc()

                    bt.logging.info("scores attempted to be stored in cmw")
                    bt.logging.info("run complete.")
                else:
                    bt.logging.info(
                        "there are no predictions to score that have the right number of predictions"
                    )
            # If we encounter an unexpected error, log it for debugging.
            except RuntimeError as e:
                bt.logging.error(e)
                traceback.print_exc()
            except MinResponsesException as e:
                bt.logging.info(
                    "removing processed files as min responses "
                    "not met to not continue to iterate over them"
                )
                for file in vali_request.files:
                    os.remove(file)
                bt.logging.error(e)
                traceback.print_exc()
            except IncorrectLiveResultsCountException as e:
                bt.logging.info(
                    "removing processed files as can't get accurate live results"
                )
                for file in vali_request.files:
                    os.remove(file)
                bt.logging.error(e)
                traceback.print_exc()
            except ValueError as e:
                bt.logging.info(
                    "value error during live result weight setting process"
                )
                for file in vali_request.files:
                    os.remove(file)
                bt.logging.error(e)
                traceback.print_exc()
            except Exception as e:
                bt.logging.error(e)
                traceback.print_exc()


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    config = get_config()

    # base setup for valis

    # Set up logging with the provided configuration and directory.
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(
        f"Running validator for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint} with config:"
    )
    # Log the configuration for reference.
    bt.logging.info(config)

    # Step 4: Build Bittensor validator objects
    # These are core Bittensor classes to interact with the network.
    bt.logging.info("Setting up bittensor objects.")

    # The wallet holds the cryptographic key pairs for the validator.
    wallet = bt.wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")

    # The subtensor is our connection to the Bittensor blockchain.
    subtensor = bt.subtensor(config=config)
    bt.logging.info(f"Subtensor: {subtensor}")

    # Dendrite is the RPC client; it lets us send messages to other nodes (axons) in the network.
    dendrite = bt.dendrite(wallet=wallet)
    bt.logging.info(f"Dendrite: {dendrite}")

    # The metagraph holds the state of the network, letting us know about other validators and miners.
    metagraph = subtensor.metagraph(config.netuid)
    bt.logging.info(f"Metagraph: {metagraph}")

    # Step 5: Connect the validator to the network
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(
            f"\nYour validator: {wallet} if not registered to chain connection: {subtensor} \nRun btcli register and try again."
        )
        exit()

    # Each validator gets a unique identity (UID) in the network for differentiation.
    my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    bt.logging.info(f"Running validator on uid: {my_subnet_uid}")

    # Step 7: The Main Validation Loop
    bt.logging.info("Starting validator loop.")
    while True:
        current_time = datetime.now().time()

        if current_time.minute in ValiConfig.METAGRAPH_UPDATE_INTERVALS:
            bt.logging.info("Updating metagraph.")
            # updating metagraph before run
            metagraph.sync(subtensor=subtensor)
            bt.logging.info(f"Metagraph updated: {metagraph}")

        if current_time.minute in MinerConfig.ACCEPTABLE_INTERVALS_HASH:
            requests = []
            # see if any files exist, if not then generate a client request (a live prediction)
            all_files = ValiBkpUtils.get_all_files_in_dir(
                ValiBkpUtils.get_vali_predictions_dir()
            )
            # if len(all_files) == 0 or int(config.continuous_data_feed) == 1:

            # standardizing getting request
            requests.append(ValiUtils.generate_standard_request(ClientRequest))

            predictions_to_complete = ValiUtils.get_predictions_to_complete()

            bt.logging.info(
                f"Have [{len(predictions_to_complete)}] requests prepared to have weights set for"
            )

            if len(predictions_to_complete) > 0:
                # add one request of predictions to complete
                requests.append(predictions_to_complete[0])

            bt.logging.info(f"Number of requests being handled [{len(requests)}]")
            run_time_series_validation(wallet, config, metagraph, requests)
            time.sleep(60)