# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Taoshidev
# Copyright © 2023 Taoshi Inc


# Step 1: Import necessary libraries and modules
import datetime
import hashlib
import os
import sys
import threading
import time
from typing import Tuple

import numpy as np

import template
import argparse
import traceback
import bittensor as bt

from data_generator.data_generator_handler import DataGeneratorHandler
from hashing_utils import HashingUtils
from miner_config import MinerConfig
from mining_objects.base_mining_model import BaseMiningModel
from mining_objects.mining_utils import MiningUtils
from time_util.time_util import TimeUtil
from vali_config import ValiConfig
from vali_objects.dataclasses.stream_prediction import StreamPrediction
from vali_objects.request_templates import RequestTemplates
from vali_objects.utils.vali_utils import ValiUtils

base_mining_model = None
base_model_id = None


def get_config():
    # Step 2: Set up the configuration parser
    # This function initializes the necessary command-line arguments.
    # Using command-line arguments allows users to customize various miner settings.
    parser = argparse.ArgumentParser()
    # TODO(developer): Adds your custom miner arguments to the parser.
    parser.add_argument(
        "--custom", default="my_custom_value", help="Adds a custom value to the parser."
    )
    # Adds override arguments for network and netuid.
    parser.add_argument("--netuid", type=int, default=1, help="The chain subnet uid.")
    parser.add_argument("--base_model", type=str, default="model_v4_1", help="Choose the base model you want to run (if youre not using a custom one).")
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

    # Step 3: Set up logging directory
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


def update_predictions(stream_predictions, data_generator_handler, model_chosen):
    while True:
            current_time = datetime.datetime.now()
            if current_time.second < 15:
                try:
                    bt.logging.debug(f"running update of predictions [{current_time}]")
                    # for each template
                    for stream_prediction in stream_predictions:
                        if stream_prediction.stream_type not in miner_preds:
                            bt.logging.info(
                                f"stream type doesn't exist, setting to an empty list for [{stream_prediction.stream_type}]")
                            miner_preds[stream_prediction.stream_type] = []
                        bt.logging.debug(
                            f"current predicted closes in memory [{miner_preds[stream_prediction.stream_type]}]")
                        bt.logging.info(f"setting predictions for [{stream_prediction.stream_type}]")
                        # get lookback data
                        ts_ranges = TimeUtil.convert_range_timestamps_to_millis(
                            TimeUtil.generate_range_timestamps(
                                TimeUtil.generate_start_timestamp(MinerConfig.STD_LOOKBACK), MinerConfig.STD_LOOKBACK))
                        ds = ValiUtils.get_standardized_ds()
                        for ts_range in ts_ranges:
                            data_generator_handler.data_generator_handler(stream_prediction.topic_id,
                                                                          0,
                                                                          stream_prediction.additional_details,
                                                                          ds,
                                                                          ts_range)
                        np_ds = np.array(ds)

                        # generate stream predictions
                        predicted_closes = MiningUtils.open_model_prediction_generation(np_ds, model_chosen,
                                                                                        stream_prediction.prediction_size)
                        bt.logging.debug(f"predicted closes [{predicted_closes}]")

                        # set preds in memory
                        miner_preds[stream_prediction.stream_type] = predicted_closes
                        bt.logging.info(f"done setting predictions for [{stream_prediction.stream_type}] in memory "
                                        f"with length [{len(predicted_closes)}]")
                    time.sleep(15)
                except Exception as e:
                    bt.logging.warning(f"error updating predictions, will sleep & retry: {e}")
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
        bt.logging.trace(f'Hotkey does not exist in metagraph [{hotkey}]')
        return True

    bt.logging.info(f'Valid hotkey [{hotkey}]')

    # step 2: check to see if the vali has min threshold
    uid = None
    for k, v in enumerate(metagraph.axons):
        if v.hotkey == hotkey:
            uid = k
            break

    stake = metagraph.neurons[uid].stake.tao
    bt.logging.debug(f"stake of [{hotkey}]: [{stake}]")

    if stake < MinerConfig.MIN_VALI_SIZE:
        bt.logging.info(f"Denied due to low stake. Min threshold [{MinerConfig.MIN_VALI_SIZE}]")
        return True

    # step 3: ensure the request is in the required time window
    current_time = datetime.datetime.now()

    bt.logging.debug(f"Acceptable intervals for requests [{acceptable_intervals}]")

    if current_time.minute not in acceptable_intervals:
        bt.logging.info(f"Denied due to incorrect interval [{current_time.minute}m]")
        return True

    return False


# Main takes the config and starts the miner.
def main( config ):
    base_mining_models = {
        "model_v4_1": {
            "model_dir": get_model_dir("/mining_models/model_v4_1.h5"),
            "window_size": 100,
            "id": "model2308",
            "features": BaseMiningModel.base_model_dataset,
            "rows": 601
        },
        "model_v4_2": {
            "model_dir": get_model_dir("/mining_models/model_v4_2.h5"),
            "window_size": 500,
            "id": "model3005",
            "features": BaseMiningModel.base_model_dataset,
            "rows": 601
        },
        "model_v4_3": {
            "model_dir": get_model_dir("/mining_models/model_v4_3.h5"),
            "window_size": 100,
            "id": "model3103",
            "features": BaseMiningModel.base_model_dataset,
            "rows": 601
        },
        "model_v4_4": {
            "model_dir": get_model_dir("/mining_models/model_v4_4.h5"),
            "window_size": 100,
            "id": "model3104",
            "features": BaseMiningModel.base_model_dataset,
            "rows": 601
        },
        "model_v4_5": {
            "model_dir": get_model_dir("/mining_models/model_v4_5.h5"),
            "window_size": 100,
            "id": "model3105",
            "features": BaseMiningModel.base_model_dataset,
            "rows": 601
        },
        "model_v4_6": {
            "model_dir": get_model_dir("/mining_models/model_v4_6.h5"),
            "window_size": 100,
            "id": "model3106",
            "features": BaseMiningModel.base_model_dataset,
            "rows": 601
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

    # where we'll store miner predictions in memory and reference
    miner_preds = {}
    sent_preds = {}

    base_mining_model = None
    base_model_id = config.base_model
    model_chosen = None

    if base_model_id is not None and base_model_id in base_mining_models:
        bt.logging.debug(f"using an existing base model [{config.base_model}]")

        model_chosen = base_mining_models[base_model_id]

        base_mining_model = BaseMiningModel(4) \
            .set_window_size(model_chosen["window_size"]) \
            .set_model_dir(model_chosen["model_dir"]) \
            .load_model()
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

    # def tf_blacklist_fn(synapse: template.protocol.TrainingForward) -> Tuple[bool, str]:
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
    # def tb_blacklist_fn( synapse: template.protocol.TrainingBackward ) -> Tuple[bool, str]:
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

    def lf_hash_blacklist_fn(synapse: template.protocol.LiveForwardHash) -> Tuple[bool, str]:
        _is_invalid_validator = is_invalid_validator(metagraph,
                                                     synapse.dendrite.hotkey,
                                                     MinerConfig.ACCEPTABLE_INTERVALS_HASH)
        return _is_invalid_validator, synapse.dendrite.hotkey

    def lf_hash_priority_fn(synapse: template.protocol.LiveForwardHash) -> float:
        caller_uid = metagraph.hotkeys.index( synapse.dendrite.hotkey )
        priority = float( metagraph.S[ caller_uid ] )
        bt.logging.trace(f'Prioritizing {synapse.dendrite.hotkey} with value: ', priority)
        return priority

    def live_hash_f(synapse: template.protocol.LiveForwardHash) -> template.protocol.LiveForwardHash:

        bt.logging.debug(f"received lf hash request on stream type [{synapse.stream_id}] "
                         f"by vali [{synapse.dendrite.hotkey}]")
        # Convert the string back to a list using literal_eval
        try:
            if synapse.stream_id in miner_preds:
                if synapse.stream_id not in sent_preds:
                    sent_preds[synapse.stream_id] = {}
                stream_preds = miner_preds[synapse.stream_id]
                sent_preds[synapse.stream_id][synapse.dendrite.hotkey] = stream_preds
                hashed_preds = HashingUtils.hash_predictions(wallet.hotkey.ss58_address, str(stream_preds))

                synapse.hashed_predictions = hashed_preds
                bt.logging.debug(f"sending hash lf [{stream_preds}]")
                bt.logging.debug(f'sending hash lf with length [{len(stream_preds)}]')
                return synapse
            else:
                bt.logging.error(f"miner preds not stored properly in memory")
        except Exception as e:
            bt.logging.error(f"error returning synapse to vali: {e}")

    def lf_blacklist_fn(synapse: template.protocol.LiveForward) -> Tuple[bool, str]:
        _is_invalid_validator = is_invalid_validator(metagraph,
                                                     synapse.dendrite.hotkey,
                                                     MinerConfig.ACCEPTABLE_INTERVALS_PREDICTIONS)
        return _is_invalid_validator, synapse.dendrite.hotkey

    def lf_priority_fn(synapse: template.protocol.LiveForward) -> float:
        caller_uid = metagraph.hotkeys.index( synapse.dendrite.hotkey )
        priority = float( metagraph.S[ caller_uid ] )
        bt.logging.trace(f'Prioritizing {synapse.dendrite.hotkey} with value: ', priority)
        return priority

    def live_f(synapse: template.protocol.LiveForward) -> template.protocol.LiveForward:

        bt.logging.debug(f"received lf request on stream type [{synapse.stream_id}] "
                         f"by vali [{synapse.dendrite.hotkey}]")

        # need to determine if the validator has correctly updated to the latest logic. This is
        # important for backward compatibility so miners dont lose incentive
        try:
            if synapse.stream_id in sent_preds and synapse.dendrite.hotkey in sent_preds[synapse.stream_id]:
                stream_preds = sent_preds[synapse.stream_id][synapse.dendrite.hotkey]
                synapse.predictions = bt.tensor(np.array(stream_preds))
            else:
                bt.logging.warning(f"suspecting validator has not updated to V5, sending back non-hash based preds")
                stream_preds = miner_preds[synapse.stream_id]
                synapse.predictions = bt.tensor(np.array(stream_preds))
            bt.logging.debug(f"sending lf [{stream_preds}]")
            bt.logging.debug(f'sending lf with length [{len(stream_preds)}]')
            return synapse
        except Exception as e:
            bt.logging.error(f"error returning synapse to vali: {e}")

    # def lb_blacklist_fn(synapse: template.protocol.LiveBackward) -> Tuple[bool, str]:
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

    # Step 5: Build and link miner functions to the axon.
    # The axon handles request processing, allowing validators to send this process requests.
    bt.logging.info(f"setting port [{config.axon.port}]")
    bt.logging.info(f"setting external port [{config.axon.external_port}]")
    axon = bt.axon( wallet = wallet, port=config.axon.port, external_port=config.axon.external_port)
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
        forward_fn = live_f,
        blacklist_fn = lf_blacklist_fn,
        priority_fn = lf_priority_fn,
    )
    axon.attach(
        forward_fn = live_hash_f,
        blacklist_fn = lf_hash_blacklist_fn,
        priority_fn = lf_hash_priority_fn,
    )


    # will leave live backward running from valis in case as a miner you'd like to use. Left out for majority of miners.

    # axon.attach(
    #     forward_fn = live_b,
    #     blacklist_fn = lb_blacklist_fn,
    #     priority_fn = lb_priority_fn,
    # )

    # Serve passes the axon information to the network + netuid we are hosting on.
    # This will auto-update if the axon port of external ip have changed.
    bt.logging.info(f"Serving attached axons on network:"
                    f" {config.subtensor.chain_endpoint} with netuid: {config.netuid}")
    axon.serve(netuid = config.netuid, subtensor = subtensor )

    # Start  starts the miner's axon, making it active on the network.
    bt.logging.info(f"Starting axon server on port: {config.axon.port}")
    axon.start()

    # Step 6: Keep the miner alive
    # This loop maintains the miner's operations until intentionally stopped.
    bt.logging.info(f"Starting main loop")
    step = 0

    if base_mining_model is None:
        bt.logging.error(f"base model not chosen, please pass using --base_model input arg")
        sys.exit(0)

    stream_predictions = [StreamPrediction.init_stream_prediction(request_template)
                          for request_template in RequestTemplates().templates]

    data_generator_handler = DataGeneratorHandler()

    run_update_predictions = threading.Thread(target=update_predictions, args=(stream_predictions,
                                                                        data_generator_handler,
                                                                        model_chosen))
    run_update_predictions.start()

    while True:
        try:
            # Below: Periodically update our knowledge of the network graph.
            if step % 5 == 0:
                metagraph = subtensor.metagraph(config.netuid)
                log =  (f'Step:{step} | '\
                        f'Block:{metagraph.block.item()} | '\
                        f'Stake:{metagraph.S[my_subnet_uid]} | '\
                        f'Rank:{metagraph.R[my_subnet_uid]} | '\
                        f'Trust:{metagraph.T[my_subnet_uid]} | '\
                        f'Consensus:{metagraph.C[my_subnet_uid] } | '\
                        f'Incentive:{metagraph.I[my_subnet_uid]} | '\
                        f'Emission:{metagraph.E[my_subnet_uid]}')
                bt.logging.info(log)
            step += 1
            time.sleep(1)

        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            axon.stop()
            bt.logging.success('Miner killed by keyboard interrupt.')
            break
        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception as e:
            bt.logging.error(traceback.format_exc())
            continue

    run_update_predictions.join()


# This is the main function, which runs the miner.
if __name__ == "__main__":
    main( get_config() )