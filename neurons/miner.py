# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: taoshi-mbrown
# Copyright © 2024 Taoshi, LLC
import argparse
import bittensor as bt
from features import FeatureCollector, FeatureSource, FeatureScaler
from hashing_utils import HashingUtils
from miner_config import MinerConfig
from mining_objects import BaseMiningModel,MiningModelStack
import numpy as np
import os
from typing import Tuple

from streams.btcusd_5m import (
    INTERVAL_MS,
    model_feature_ids,
    model_feature_scaler,
    model_feature_sources,
    prediction_feature_ids,
    PREDICTION_COUNT,
    PREDICTION_LENGTH,
    SAMPLE_COUNT,
)
import sys
import template
import threading
import time
from time_util import datetime
import traceback
from vali_config import ValiConfig
from vali_objects.dataclasses.stream_prediction import StreamPrediction
from vali_objects.request_templates import RequestTemplates
from neuralforecast import NeuralForecast
import pandas as pd 
from sklearn.metrics import mean_squared_error
import os 

FEATURE_COLLECTOR_TIMEOUT = 10.0

btcusd_5m_feature_source = FeatureCollector(
    sources=model_feature_sources,
    feature_ids=model_feature_ids,
    cache_results=True,
    timeout=FEATURE_COLLECTOR_TIMEOUT,
)

base_mining_model: MiningModelStack| None = None
base_model_id = None

# Cached miner predictions
miner_preds = {}
sent_preds = {}

import pandas as pd
import numpy as np
from pandas.tseries.offsets import Minute
def linear_pred(last_close,preds,prediction_size):

    total_movement =  preds[-1] - last_close
    total_movement_increment = total_movement / prediction_size

    predicted_closes = []
    curr_price = last_close
    for x in range(prediction_size):
        curr_price += total_movement_increment
        predicted_closes.append(curr_price)

    return predicted_closes



def get_config():
    # Set up the configuration parser
    # This function initializes the necessary command-line arguments.
    # Using command-line arguments allows users to customize various miner settings.
    parser = argparse.ArgumentParser()
    # TODO(developer): Adds your custom miner arguments to the parser.
    parser.add_argument(
        "--custom", default="my_custom_value", help="Adds a custom value to the parser."
    )
    # Adds override arguments for network and netuid.
    parser.add_argument("--netuid", type=int, default=1, help="The chain subnet uid.")
    parser.add_argument(
        "--base_model",
        type=str,
        default="chaotic_multi",
        help="Choose the base model you want to run (if youre not using a custom one).",
    )
    # Adds subtensor specific arguments i.e. --subtensor.chain_endpoint ... --subtensor.network ...
    bt.subtensor.add_args(parser)
    # Adds logging specific arguments i.e. --logging.debug ..., --logging.trace .. or --logging.logging_dir ...
    bt.logging.add_args(parser)
    # Adds wallet specific arguments i.e. --wallet.name ..., --wallet.hotkey ./. or --wallet.path ...
    bt.wallet.add_args(parser)
    # Adds axon specific arguments i.e. --axon.port ...
    bt.axon.add_args(parser)
    # Activating the parser to read any command-line inputs.
    # To print help message, run python3 template/miner.py --help
    config = bt.config(parser)

    # Set up logging directory
    # Logging captures events for diagnosis or understanding miner's behavior.
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            "miner",
        )
    )
    # Ensure the directory for logging exists, else create one.
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)
    return config


# TODO: Move this into a stream definition, so that each stream can be predicted using different sources, models, etc...
def get_predictions(
    tims_ms: int,
    feature_source: FeatureSource,
    feature_scaler: FeatureScaler,
    model: BaseMiningModel,
):
    # TODO: interval should come from the stream definition
    lookback_time_ms = tims_ms - (model.sample_count * INTERVAL_MS)

    feature_samples = feature_source.get_feature_samples(
        lookback_time_ms, INTERVAL_MS, model.sample_count
    )
    # remove 
    feature_scaler.scale_feature_samples(feature_samples)

    model_input = feature_source.feature_samples_to_array(feature_samples)
    predictions = model.predict(model_input)

    predicted_feature_samples = feature_source.array_to_feature_samples(
        predictions, prediction_feature_ids
    )

    # Special case with one prediction that must be extrapolated using the last sample.
    # The predicted features must also exist in the sampled features. If it does not,
    # then the prediction will be flat across the prediction length.
    prediction_length = model.prediction_length
    if (model.prediction_count == 1) and (prediction_length > 1):
        for feature_id in prediction_feature_ids:
            samples = feature_samples.get(feature_id)
            if samples is not None:
                predicted_samples = predicted_feature_samples[feature_id]
                last_sample = samples[feature_id][-1]
                prediction_sample = predicted_samples[0]
                increment = (prediction_sample - last_sample) / prediction_length

                extrapolation = last_sample
                for i in range(prediction_length):
                    extrapolation += increment
                    predicted_samples[i] = extrapolation

    feature_scaler.unscale_feature_samples(predicted_feature_samples)

    prediction_array = feature_source.feature_samples_to_array(
        predicted_feature_samples, prediction_feature_ids
    )

    return prediction_array


def update_predictions(
    stream_predictions: list[StreamPrediction],
):
    while True:
        current_time = datetime.now()
        if current_time.second < 15:
            bt.logging.debug(f"running update of predictions [{current_time}]")

            for stream_prediction in stream_predictions:
                try:
                    stream_type = stream_prediction.stream_type
                    if stream_type not in miner_preds:
                        miner_preds[stream_type] = []
                        bt.logging.info(
                            f"stream type doesn't exist, setting to an empty list for [{stream_type}]"
                        )

                    bt.logging.debug(
                        f"current predicted closes in memory [{miner_preds[stream_type]}]"
                    )
                    bt.logging.info(f"setting predictions for [{stream_type}]")

                    prediction_array = get_predictions(
                        current_time.timestamp_ms(),
                        btcusd_5m_feature_source,
                        model_feature_scaler,
                        base_mining_model,
                    )

                    # TODO: Improve validators to allow multiple features in predictions
                    predicted_closes = prediction_array.flatten()

                    bt.logging.debug(f"predicted closes [{predicted_closes}]")

                    # set preds in memory
                    miner_preds[stream_type] = predicted_closes
                    bt.logging.info(
                        f"done setting predictions for [{stream_type}] "
                        f"in memory with length [{len(predicted_closes)}]"
                    )

                # Log errors and continue operations
                except Exception:  # noqa
                    bt.logging.error(traceback.format_exc())
                continue

            time.sleep(15)
            
def get_predictions_stack(
    tims_ms: int,
    feature_source: FeatureSource,
    feature_scaler: FeatureScaler,
    model: MiningModelStack,
):
    # TODO: interval should come from the stream definition
    lookback_time_ms = tims_ms - (model.sample_count * INTERVAL_MS)

    feature_samples = feature_source.get_feature_samples(
        lookback_time_ms, INTERVAL_MS, model.sample_count,
    )
    # remove 
    #feature_scaler.scale_feature_samples(feature_samples)

    model_input = feature_source.feature_samples_to_pandas(feature_samples,start_time = lookback_time_ms,interval_ms=INTERVAL_MS)
    #futr = prepare_futr_datset(model_input)
    last_set = model_input.iloc[-1200:-25] # drop last 100 candles 

    # check this 
    best_model =  model.select_model(df=last_set,ground_truth=model_input['close'].tail(25))
    model_name = best_model.models[0]
    predicted_closes = best_model.predict(df=model_input)
    prediction_size = model.prediction_length
    predicted_closes = predicted_closes.drop(columns='ds').iloc[:,0].tolist()# change this
    if prediction_size== 101 : 
            predicted_closes.append(predicted_closes[-1])


        
    predicted_closes= linear_pred(model_input['close'].tail(1),predicted_closes,prediction_size)
    
    return predicted_closes # needs to be a list
    ## do what we do here

    return predicted_closes


def update_predictions_stack(
    stream_predictions: list[StreamPrediction],
):
    while True:
        current_time = datetime.now()
        if current_time.second < 15:
            bt.logging.debug(f"running update of predictions [{current_time}]")

            for stream_prediction in stream_predictions:
                try:
                    stream_type = stream_prediction.stream_type
                    if stream_type not in miner_preds:
                        miner_preds[stream_type] = []
                        bt.logging.info(
                            f"stream type doesn't exist, setting to an empty list for [{stream_type}]"
                        )

                    bt.logging.debug(
                        f"current predicted closes in memory [{miner_preds[stream_type]}]"
                    )
                    bt.logging.info(f"setting predictions for [{stream_type}]")

                    prediction_array = get_predictions_stack(
                        current_time.timestamp_ms(),
                        btcusd_5m_feature_source,
                        model_feature_scaler,
                        base_mining_model,
                    )

                    # TODO: Improve validators to allow multiple features in predictions
                    # predicted_closes = prediction_array.flatten()
                    predicted_closes = np.array(prediction_array).flatten()
                    bt.logging.debug(f"predicted closes [{predicted_closes}]")

                    # set preds in memory
                    miner_preds[stream_type] = predicted_closes
                    bt.logging.info(
                        f"done setting predictions for [{stream_type}] "
                        f"in memory with length [{len(predicted_closes)}]"
                    )

                # Log errors and continue operations
                except Exception:  # noqa
                    bt.logging.error(traceback.format_exc())
                continue

            time.sleep(15)

def get_model_dir(model):
    return ValiConfig.BASE_DIR + model


def is_invalid_validator(metagraph, hotkey, acceptable_intervals):
    """
    - step 1: check to see if the vali is in the metagraph
    - step 2: check to see if the vali has min threshold
    - step 3: ensure the request is in the required time window
    """

    # step 1 - check to see if the vali is in the metagraph
    if hotkey not in metagraph.hotkeys:
        bt.logging.trace(f"Hotkey does not exist in metagraph [{hotkey}]")
        return True

    bt.logging.info(f"Valid hotkey [{hotkey}]")

    # step 2: check to see if the vali has min threshold
    uid = None
    for k, v in enumerate(metagraph.axons):
        if v.hotkey == hotkey:
            uid = k
            break

    stake = metagraph.neurons[uid].stake.tao
    bt.logging.debug(f"stake of [{hotkey}]: [{stake}]")

    if stake < MinerConfig.MIN_VALI_SIZE:
        bt.logging.info(
            f"Denied due to low stake. Min threshold [{MinerConfig.MIN_VALI_SIZE}]"
        )
        return True

    # ADDING WITH V5.1.0
    # step 3: ensure the request is in the required time window
    # current_time = datetime.datetime.now()
    #
    # bt.logging.debug(f"Acceptable intervals for requests [{acceptable_intervals}]")
    #
    # if current_time.minute not in acceptable_intervals:
    #     bt.logging.info(f"Denied due to incorrect interval [{current_time.minute}m]")
    #     return True

    return False


# Main takes the config and starts the miner.
def main(config):
    base_mining_models = {
        "model_v4_1": {
            "id": "model2308",
            "model_dir": "/mining_models/model_v4_1.h5",
            "sample_count": 100,
            "prediction_count": 1,
        },
        "model_v4_2": {
            "id": "model3005",
            "model_dir": "/mining_models/model_v4_2.h5",
            "sample_count": 500,
            "prediction_count": 1,
        },
        "model_v4_3": {
            "id": "model3103",
            "model_dir": "/mining_models/model_v4_3.h5",
            "sample_count": 100,
            "prediction_count": 1,
        },
        "model_v4_4": {
            "id": "model3104",
            "model_dir": "/mining_models/model_v4_4.h5",
            "sample_count": 100,
            "prediction_count": 1,
        },
        "model_v4_5": {
            "id": "model3105",
            "model_dir": "/mining_models/model_v4_5.h5",
            "sample_count": 100,
            "prediction_count": 1,
        },
        "model_v4_6": {
            "id": "model3106",
            "model_dir": "/mining_models/model_v4_6.h5",
            "sample_count": 100,
            "prediction_count": 1,
        },
        "model_v5_1": {
            "id": "model5000",
            "filename": "/mining_models/model_v5_1.h5",
            "sample_count": SAMPLE_COUNT,
            "prediction_count": PREDICTION_COUNT,
        },
        "chaotic_multi": {
            "id": "chaotic_multi",
            "filename": "/mining_models/chaotic_multi/",
            "sample_count": SAMPLE_COUNT,
            "prediction_count": PREDICTION_COUNT,
        },
    }

    # Activating Bittensor's logging with the set configurations.
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(
        f"Running miner for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint} with config:"
    )

    global base_mining_model
    global base_model_id
    global miner_preds
    global sent_preds

    base_mining_model = None
    base_model_id = config.base_model

    if base_model_id is not None and base_model_id in base_mining_models:
        bt.logging.debug(f"using an existing base model [{config.base_model}]")

        model_chosen = base_mining_models[base_model_id]
        model_filename = get_model_dir(model_chosen["filename"])
        
        if base_model_id == 'chaotic_multi': 
            base_mining_model = MiningModelStack(
                filename=model_filename,
                mode="r",
                feature_count=len(model_feature_ids),
                sample_count=model_chosen["sample_count"],
                prediction_feature_count=len(prediction_feature_ids),
                prediction_count=model_chosen["prediction_count"],
                prediction_length=PREDICTION_LENGTH) \
                .set_model_dir(model_filename) \
                .load_models()
            
            
        else : 
            

            base_mining_model = BaseMiningModel(
                filename=model_filename,
                mode="r",
                feature_count=len(model_feature_ids),
                sample_count=model_chosen["sample_count"],
                prediction_feature_count=len(prediction_feature_ids),
                prediction_count=model_chosen["prediction_count"],
                prediction_length=PREDICTION_LENGTH,
            )

    else:
        bt.logging.debug("base model not chosen.")

    # This logs the active configuration to the specified logging directory for review.
    bt.logging.info(config)

    # Step 4: Initialize Bittensor miner objects
    # These classes are vital to interact and function within the Bittensor network.
    bt.logging.info("Setting up bittensor objects.")

    # Wallet holds cryptographic information, ensuring secure transactions and communication.
    wallet = bt.wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")

    # subtensor manages the blockchain connection, facilitating interaction with the Bittensor blockchain.
    subtensor = bt.subtensor(config=config)
    bt.logging.info(f"Subtensor: {subtensor}")

    # metagraph provides the network's current state, holding state about other participants in a subnet.
    metagraph = subtensor.metagraph(config.netuid)
    bt.logging.info(f"Metagraph: {metagraph}")

    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(
            f"\nYour miner: {wallet} is not registered to chain connection: {subtensor} \nRun btcli register and try again. "
        )
        exit()

    # Each miner gets a unique identity (UID) in the network for differentiation.
    my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    bt.logging.info(f"Running miner on uid: {my_subnet_uid}")

    # def tf_blacklist_fn(synapse: template.protocol.TrainingForward) -> tuple[bool, str]:
    #     # standardizing not accepting tf and tb for now
    #     return False, synapse.dendrite.hotkey
    #
    # def tf_priority_fn(synapse: template.protocol.TrainingForward) -> float:
    #     caller_uid = metagraph.hotkeys.index( synapse.dendrite.hotkey ) # Get the caller index.
    #     prirority = float( metagraph.S[ caller_uid ] ) # Return the stake as the priority.
    #     bt.logging.trace(f'Prioritizing {synapse.dendrite.hotkey} with value: ', prirority)
    #     return prirority
    #
    # def training_f( synapse: template.protocol.TrainingForward ) -> template.protocol.TrainingForward:
    #     bt.logging.debug(f'received tf')
    #     predictions = np.array([random.uniform(0.499, 0.501) for i in range(0, synapse.prediction_size)])
    #     synapse.predictions = bt.tensor(predictions)
    #     bt.logging.debug(f'sending tf with length {len(predictions)}')
    #     return synapse
    #
    # def tb_blacklist_fn( synapse: template.protocol.TrainingBackward ) -> tuple[bool, str]:
    #     # standardizing not accepting tf and tb for now
    #     return False, synapse.dendrite.hotkey
    #
    # def tb_priority_fn( synapse: template.protocol.TrainingBackward ) -> float:
    #     caller_uid = metagraph.hotkeys.index( synapse.dendrite.hotkey ) # Get the caller index.
    #     prirority = float( metagraph.S[ caller_uid ] ) # Return the stake as the priority.
    #     bt.logging.trace(f'Prioritizing {synapse.dendrite.hotkey} with value: ', prirority)
    #     return prirority
    #
    # def training_b( synapse: template.protocol.TrainingBackward ) -> template.protocol.TrainingBackward:
    #     bt.logging.debug(f'received lb with length {len(synapse.samples.numpy())}')
    #     synapse.received = True
    #     return synapse


    def lf_hash_blacklist_fn(
        synapse: template.protocol.LiveForwardHash,
    ) -> Tuple[bool, str]:
        _is_invalid_validator = is_invalid_validator(
            metagraph, synapse.dendrite.hotkey, MinerConfig.ACCEPTABLE_INTERVALS_HASH
        )
        return _is_invalid_validator, synapse.dendrite.hotkey

    def lf_hash_priority_fn(synapse: template.protocol.LiveForwardHash) -> float:
        caller_uid = metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority = float(metagraph.S[caller_uid])
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", priority
        )
        return priority

    def live_hash_f(
        synapse: template.protocol.LiveForwardHash,
    ) -> template.protocol.LiveForwardHash:
        bt.logging.debug(
            f"received lf hash request on stream type [{synapse.stream_id}] "
            f"by vali [{synapse.dendrite.hotkey}]"
        )
        # Convert the string back to a list using literal_eval
        try:
            if synapse.stream_id in miner_preds:
                if synapse.stream_id not in sent_preds:
                    sent_preds[synapse.stream_id] = {}
                stream_preds = miner_preds[synapse.stream_id]
                sent_preds[synapse.stream_id][synapse.dendrite.hotkey] = stream_preds
                hashed_preds = HashingUtils.hash_predictions(
                    wallet.hotkey.ss58_address, str(stream_preds)
                )

                synapse.hashed_predictions = hashed_preds
                bt.logging.debug(f"sending hash lf [{stream_preds}]")
                bt.logging.debug(f"sending hash lf with length [{len(stream_preds)}]")
                return synapse
            else:
                bt.logging.error(f"miner preds not stored properly in memory")
        except Exception as e:
            bt.logging.error(f"error returning synapse to vali: {e}")
            
            
    def lf_blacklist_fn(synapse: template.protocol.LiveForward) -> Tuple[bool, str]:
        _is_invalid_validator = is_invalid_validator(metagraph, synapse.dendrite.hotkey)
        return _is_invalid_validator, synapse.dendrite.hotkey           
    
    # orig -> to reveert
    #def lf_blacklist_fn(synapse: template.protocol.LiveForward) -> tuple[bool, str]:
    #    _is_invalid_validator = is_invalid_validator(
    #        metagraph,
    #        synapse.dendrite.hotkey,
    #        MinerConfig.ACCEPTABLE_INTERVALS_PREDICTIONS,
    #    )
    #    return _is_invalid_validator, synapse.dendrite.hotkey

    def lf_priority_fn(synapse: template.protocol.LiveForward) -> float:
        caller_uid = metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority = float(metagraph.S[caller_uid])
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", priority
        )
        return priority

    def live_f(synapse: template.protocol.LiveForward) -> template.protocol.LiveForward:
        bt.logging.debug(
            f"received lf request on stream type [{synapse.stream_id}] "
            f"by vali [{synapse.dendrite.hotkey}]"
        )

        # need to determine if the validator has correctly updated to the latest logic. This is
        # important for backward compatibility so miners dont lose incentive
        try:
            if (
                synapse.stream_id in sent_preds
                and synapse.dendrite.hotkey in sent_preds[synapse.stream_id]
            ):
                stream_preds = sent_preds[synapse.stream_id][synapse.dendrite.hotkey]
                synapse.predictions = bt.tensor(np.array(stream_preds))
            else:
                bt.logging.warning(
                    f"suspecting validator has not updated to V5, sending back non-hash based preds"
                )
                stream_preds = miner_preds[synapse.stream_id]
                synapse.predictions = bt.tensor(np.array(stream_preds))
            bt.logging.debug(f"sending lf [{stream_preds}]")
            bt.logging.debug(f"sending lf with length [{len(stream_preds)}]")
            return synapse
        except Exception as e:
            bt.logging.error(f"error returning synapse to vali: {e}")

    # def lb_blacklist_fn(synapse: template.protocol.LiveBackward) -> tuple[bool, str]:
    #     # standardizing not accepting lb for now. Miner can override if they'd like.
    #     return False, synapse.dendrite.hotkey
    #
    # def lb_priority_fn(synapse: template.protocol.LiveBackward) -> float:
    #     caller_uid = metagraph.hotkeys.index( synapse.dendrite.hotkey ) # Get the caller index.
    #     prirority = float( metagraph.S[ caller_uid ] ) # Return the stake as the priority.
    #     bt.logging.trace(f'Prioritizing {synapse.dendrite.hotkey} with value: ', prirority)
    #     return prirority

    # def live_b(synapse: template.protocol.LiveBackward) -> template.protocol.LiveBackward:
    #     bt.logging.debug(f'received lb with length {len(synapse.samples.numpy())}')
    #     synapse.received = True
    #     return synapse

    # Build and link miner functions to the axon.
    # The axon handles request processing, allowing validators to send this process requests.
    bt.logging.info(f"setting port [{config.axon.port}]")
    bt.logging.info(f"setting external port [{config.axon.external_port}]")
    axon = bt.axon(
        wallet=wallet, port=config.axon.port, external_port=config.axon.external_port
    )
    bt.logging.info(f"Axon {axon}")

    # Attach determiners which functions are called when servicing a request.
    bt.logging.info(f"Attaching live forward functions to axon.")
    # axon.attach(
    #     forward_fn = training_f,
    #     blacklist_fn = tf_blacklist_fn,
    #     priority_fn = tf_priority_fn,
    # )
    # axon.attach(
    #     forward_fn = training_b,
    #     blacklist_fn = tb_blacklist_fn,
    #     priority_fn = tb_priority_fn,
    # )
    axon.attach(
        forward_fn=live_f,
        blacklist_fn=lf_blacklist_fn,
        priority_fn=lf_priority_fn,
    )
    axon.attach(
        forward_fn=live_hash_f,
        blacklist_fn=lf_hash_blacklist_fn,
        priority_fn=lf_hash_priority_fn,
    )

    # Will leave live backward running from valis in case as a miner you'd like to use.
    # Left out for majority of miners.

    # axon.attach(
    #     forward_fn = live_b,
    #     blacklist_fn = lb_blacklist_fn,
    #     priority_fn = lb_priority_fn,
    # )

    # Serve passes the axon information to the network + netuid we are hosting on.
    # This will auto-update if the axon port of external ip have changed.
    bt.logging.info(
        f"Serving attached axons on network:"
        f" {config.subtensor.chain_endpoint} with netuid: {config.netuid}"
    )
    axon.serve(netuid=config.netuid, subtensor=subtensor)

    # Start  starts the miner's axon, making it active on the network.
    bt.logging.info(f"Starting axon server on port: {config.axon.port}")
    axon.start()

    # Keep the miner alive
    # This loop maintains the miner's operations until intentionally stopped.
    bt.logging.info(f"Starting main loop")
    step = 0

    if base_mining_model is None:
        bt.logging.error(
            f"base model not chosen, please pass using --base_model input arg"
        )
        sys.exit(0)

    stream_predictions = [
        StreamPrediction.init_stream_prediction(request_template)
        for request_template in RequestTemplates().templates
    ]

    run_update_predictions = threading.Thread(
        target=update_predictions_stack,
        args=(stream_predictions,),
    )
    run_update_predictions.start()

    while True:
        try:
            # Below: Periodically update our knowledge of the network graph.
            if step % 5 == 0:
                metagraph = subtensor.metagraph(config.netuid)
                log = (
                    f"Step:{step} | "
                    f"Block:{metagraph.block.item()} | "
                    f"Stake:{metagraph.S[my_subnet_uid]} | "
                    f"Rank:{metagraph.R[my_subnet_uid]} | "
                    f"Trust:{metagraph.T[my_subnet_uid]} | "
                    f"Consensus:{metagraph.C[my_subnet_uid] } | "
                    f"Incentive:{metagraph.I[my_subnet_uid]} | "
                    f"Emission:{metagraph.E[my_subnet_uid]}"
                )
                bt.logging.info(log)
            step += 1
            time.sleep(1)

        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            axon.stop()
            bt.logging.success("Miner killed by keyboard interrupt.")
            break

        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception as main_loop_exception:  # noqa
            bt.logging.error(traceback.format_exc())
            continue

    run_update_predictions.join()


# This is the main function, which runs the miner.
if __name__ == "__main__":
    main(get_config())
