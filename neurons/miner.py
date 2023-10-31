# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Taoshidev
# Copyright © 2023 Taoshi, LLC


# Step 1: Import necessary libraries and modules
import os
import random
import time
from typing import Type, Tuple

import numpy as np

import template
import argparse
import traceback
import bittensor as bt

from mining_objects.base_mining_model import BaseMiningModel


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


# Main takes the config and starts the miner.
def main( config ):

    # Activating Bittensor's logging with the set configurations.
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(
        f"Running miner for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint} with config:"
    )

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

    def tf_blacklist_fn(synapse: template.protocol.TrainingForward) -> Tuple[bool, str]:
        if synapse.dendrite.hotkey not in metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(f'Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}')
            return True, synapse.dendrite.hotkey
        bt.logging.trace(f'Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}')
        return False, synapse.dendrite.hotkey

    def tf_priority_fn(synapse: template.protocol.TrainingForward) -> float:
        caller_uid = metagraph.hotkeys.index( synapse.dendrite.hotkey ) # Get the caller index.
        prirority = float( metagraph.S[ caller_uid ] ) # Return the stake as the priority.
        bt.logging.trace(f'Prioritizing {synapse.dendrite.hotkey} with value: ', prirority)
        return prirority

    # This is the core miner function, which decides the miner's response to a valid, high-priority request.
    def training_f( synapse: template.protocol.TrainingForward ) -> template.protocol.TrainingForward:
        bt.logging.debug(f'received tf')
        predictions = np.array([random.uniform(0.499, 0.501) for i in range(0, synapse.prediction_size)])
        synapse.predictions = bt.tensor(predictions)
        bt.logging.debug(f'sending tf with length {len(predictions)}')
        return synapse

    def tb_blacklist_fn( synapse: template.protocol.TrainingBackward ) -> Tuple[bool, str]:
        if synapse.dendrite.hotkey not in metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(f'Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}')
            return True, synapse.dendrite.hotkey
        bt.logging.trace(f'Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}')
        return False, synapse.dendrite.hotkey

    def tb_priority_fn( synapse: template.protocol.TrainingBackward ) -> float:
        caller_uid = metagraph.hotkeys.index( synapse.dendrite.hotkey ) # Get the caller index.
        prirority = float( metagraph.S[ caller_uid ] ) # Return the stake as the priority.
        bt.logging.trace(f'Prioritizing {synapse.dendrite.hotkey} with value: ', prirority)
        return prirority

    # This is the core miner function, which decides the miner's response to a valid, high-priority request.
    def training_b( synapse: template.protocol.TrainingBackward ) -> template.protocol.TrainingBackward:
        bt.logging.debug(f'received lb with length {len(synapse.samples.numpy())}')
        synapse.received = True
        return synapse

    def lf_blacklist_fn(synapse: template.protocol.LiveForward) -> Tuple[bool, str]:
        bt.logging.debug("got to blacklisting lf")
        if synapse.dendrite.hotkey not in metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(f'Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}')
            return True, synapse.dendrite.hotkey
        bt.logging.trace(f'Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}')
        return False, synapse.dendrite.hotkey

    def lf_priority_fn(synapse: template.protocol.LiveForward) -> float:
        caller_uid = metagraph.hotkeys.index( synapse.dendrite.hotkey ) # Get the caller index.
        prirority = float( metagraph.S[ caller_uid ] ) # Return the stake as the priority.
        bt.logging.trace(f'Prioritizing {synapse.dendrite.hotkey} with value: ', prirority)
        return prirority

    # This is the core miner function, which decides the miner's response to a valid, high-priority request.
    def live_f(synapse: template.protocol.LiveForward) -> template.protocol.LiveForward:
        bt.logging.debug(f'received tf')
        prep_dataset = BaseMiningModel.base_model_dataset(synapse.samples.numpy())
        base_mining_model = BaseMiningModel(len(prep_dataset.T))\
            .set_model_dir('mining_models/base_model.h5')\
            .set_window_size(12)\
            .load_model()

        prep_dataset_cp = prep_dataset[:]

        predicted_closes = []
        for i in range(synapse.prediction_size):
            predictions = base_mining_model.predict(prep_dataset_cp)[0]
            prep_dataset_cp = np.concatenate((prep_dataset_cp, predictions), axis=0)
            predicted_closes.append(predictions.tolist()[0][0])

        synapse.predictions = bt.tensor(np.array(predicted_closes))
        
        bt.logging.debug(f'sending tf with length {len(predicted_closes)}')
        return synapse

    def lb_blacklist_fn(synapse: template.protocol.LiveBackward) -> Tuple[bool, str]:
        if synapse.dendrite.hotkey not in metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(f'Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}')
            return True, synapse.dendrite.hotkey
        bt.logging.trace(f'Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}')
        return False, synapse.dendrite.hotkey

    def lb_priority_fn(synapse: template.protocol.LiveBackward) -> float:
        caller_uid = metagraph.hotkeys.index( synapse.dendrite.hotkey ) # Get the caller index.
        prirority = float( metagraph.S[ caller_uid ] ) # Return the stake as the priority.
        bt.logging.trace(f'Prioritizing {synapse.dendrite.hotkey} with value: ', prirority)
        return prirority

    # This is the core miner function, which decides the miner's response to a valid, high-priority request.
    def live_b(synapse: template.protocol.LiveBackward) -> template.protocol.LiveBackward:
        bt.logging.debug(f'received lb with length {len(synapse.samples.numpy())}')
        synapse.received = True
        return synapse

    # Step 5: Build and link miner functions to the axon.
    # The axon handles request processing, allowing validators to send this process requests.
    bt.logging.info(f"setting port [{config.axon.port}]")
    bt.logging.info(f"setting external port [{config.axon.external_port}]")
    axon = bt.axon( wallet = wallet, port=config.axon.port, external_port=config.axon.external_port)
    bt.logging.info(f"Axon {axon}")

    # Attach determiners which functions are called when servicing a request.
    bt.logging.info(f"Attaching forward function to axon.")
    axon.attach(
        forward_fn = training_f,
        blacklist_fn = tf_blacklist_fn,
        priority_fn = tf_priority_fn,
    )
    axon.attach(
        forward_fn = training_b,
        blacklist_fn = tb_blacklist_fn,
        priority_fn = tb_priority_fn,
    )
    axon.attach(
        forward_fn = live_f,
        blacklist_fn = lf_blacklist_fn,
        priority_fn = lf_priority_fn,
    )
    axon.attach(
        forward_fn = live_b,
        blacklist_fn = lb_blacklist_fn,
        priority_fn = lb_priority_fn,
    )

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
    while True:
        try:
            # TODO(developer): Define any additional operations to be performed by the miner.
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


# This is the main function, which runs the miner.
if __name__ == "__main__":
    main( get_config() )