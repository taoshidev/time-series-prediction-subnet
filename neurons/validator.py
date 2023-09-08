# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Taoshi
# Copyright © 2023 TARVIS Labs, LLC

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Bittensor Validator Template:
# TODO(developer): Rewrite based on protocol defintion.

# Step 1: Import necessary libraries and modules
import os
import time
import uuid

import numpy as np
import torch

import argparse
import traceback
import bittensor as bt

# import this repo
import template

# Step 2: Set up the configuration parser
# This function is responsible for setting up and parsing command-line arguments.
from data_generator.binance_data import BinanceData
from vali_objects.exceptions import ValiMemoryCorruptDataException
from vali_objects.exceptions import ValiKeyMisalignmentException
from vali_objects.exceptions import ValiMemoryMissingException
from request_objects import PredictTrainingRequest, PredictLiveRequest
from template.protocol import Forward
from time_util.time_util import TimeUtil
from vali_objects.utils.vali_utils import ValiUtils
from vali_config import ValiConfig


def get_config():
    parser = argparse.ArgumentParser()
    # TODO(developer): Adds your custom validator arguments to the parser.
    parser.add_argument('--custom', default='my_custom_value', help='Adds a custom value to the parser.')
    # Adds override arguments for network and netuid.
    parser.add_argument('--netuid', type=int, default=1, help="The chain subnet uid.")
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
            'validator',
        )
    )
    # Ensure the logging directory exists.
    if not os.path.exists(config.full_path): os.makedirs(config.full_path, exist_ok=True)

    # Return the parsed config.
    return config


def main(config):
    # Set up logging with the provided configuration and directory.
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(
        f"Running validator for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint} with config:")
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

    # The metagraph holds the state of the network, letting us know about other miners.
    metagraph = subtensor.metagraph(config.netuid)
    bt.logging.info(f"Metagraph: {metagraph}")

    # Step 5: Connect the validator to the network
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(
            f"\nYour validator: {wallet} if not registered to chain connection: {subtensor} \nRun btcli register and try again.")
        exit()
    else:
        # Each miner gets a unique identity (UID) in the network for differentiation.
        my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
        bt.logging.info(f"Running validator on uid: {my_subnet_uid}")

    # Step 6: Set up initial scoring weights for validation
    bt.logging.info("Building validation weights.")
    alpha = 0.9
    scores = torch.ones_like(metagraph.S, dtype=torch.float32)
    bt.logging.info(f"Weights: {scores}")

    # Step 7: The Main Validation Loop
    bt.logging.info("Starting validator loop.")
    step = 0
    while True:
        try:
            # TODO(developer): Define how the validator selects a miner to query, how often, etc.
            # Broadcast a query to all miners on the network.
            responses = dendrite.query(
                # Send the query to all axons in the network.
                metagraph.axons,
                # Construct a dummy query.
                template.protocol.Dummy(dummy_input=step),  # Construct a dummy query.
                # All responses have the deserialize function called on them before returning.
                deserialize=True,
            )

            # Log the results for monitoring purposes.
            bt.logging.info(f"Received dummy responses: {responses}")

            # TODO(developer): Define how the validator scores responses.
            # Adjust the scores based on responses from miners.
            for i, resp_i in enumerate(responses):
                # Initialize the score for the current miner's response.
                score = 0

                # Check if the miner has provided the correct response by doubling the dummy input.
                # If correct, set their score for this round to 1.
                if resp_i == step * 2:
                    score = 1

                # Update the global score of the miner.
                # This score contributes to the miner's weight in the network.
                # A higher weight means that the miner has been consistently responding correctly.
                scores[i] = alpha * scores[i] + (1 - alpha) * 0

            # Periodically update the weights on the Bittensor blockchain.
            if (step + 1) % 100 == 0:
                # TODO(developer): Define how the validator normalizes scores before setting weights.
                weights = torch.nn.functional.normalize(scores, p=1.0, dim=0)
                bt.logging.info(f"Setting weights: {weights}")
                # This is a crucial step that updates the incentive mechanism on the Bittensor blockchain.
                # Miners with higher scores (or weights) receive a larger share of TAO rewards on this subnet.
                subtensor.set_weights(
                    netuid=config.netuid,  # Subnet to set weights on.
                    wallet=wallet,  # Wallet to sign set weights using hotkey.
                    uids=metagraph.uids,  # Uids of the miners to set weights for.
                    weights=weights  # Weights to set for the miners.
                )

            # End the current step and prepare for the next iteration.
            step += 1
            # Resync our local state with the latest state from the blockchain.
            metagraph = subtensor.metagraph(config.netuid)
            # Sleep for a duration equivalent to the block time (i.e., time between successive blocks).
            time.sleep(bt.__blocktime__)

        # If we encounter an unexpected error, log it for debugging.
        except RuntimeError as e:
            bt.logging.error(e)
            traceback.print_exc()

        # If the user interrupts the program, gracefully exit.
        except KeyboardInterrupt:
            bt.logging.success("Keyboard interrupt detected. Exiting validator.")
            exit()


def get_vali_records() -> dict:
    # first ensure memory and bkp align
    try:
        ValiUtils.check_memory_matches_bkp()
    except ValiMemoryMissingException:
        bt_logger.info("memory data is missing, attempting to load from bkp!")
        ValiUtils.set_memory_with_bkp()
        get_vali_records()
    except ValiKeyMisalignmentException:
        bt_logger.info("bkp and memory don't match in existing data. Reloading memory from bkp!")
        ValiUtils.set_memory_with_bkp()
        get_vali_records()
    except ValiMemoryCorruptDataException:
        bt_logger.error("bkp data in unexpected format. Please download bkp from available source of truth vali.")
        raise ValiMemoryCorruptDataException
    except Exception as e:
        bt.logging.error(e)
        traceback.print_exc()
    else:
        return ValiUtils.get_vali_memory_json()


def calculate_weighted_rmse(predictions: np, actual: np) -> float:
    predictions = np.array(predictions)
    actual = np.array(actual)

    k = 0.01

    weights = np.exp(-k * np.arange(len(predictions)))

    weighted_squared_errors = weights * (predictions - actual) ** 2
    weighted_rmse = np.sqrt(np.sum(weighted_squared_errors) / np.sum(weights))

    return weighted_rmse


def calculate_directional_accuracy(predictions: np, actual: np) -> float:
    pred_len = len(predictions)

    pred_dir = np.sign([predictions[i] - predictions[i - 1] for i in range(1, pred_len)])
    actual_dir = np.sign([actual[i] - actual[i - 1] for i in range(1, pred_len)])

    correct_directions = 0

    for i in range(0, pred_len):
        correct_directions += actual_dir[i] == pred_dir[i]

    return correct_directions / pred_len


def score_response(predictions: np, actual: np) -> float:
    if len(predictions) != len(actual):
        return 0

    rmse = calculate_weighted_rmse(predictions, actual)
    da = calculate_directional_accuracy(predictions, actual)

    # geometric mean
    return np.sqrt(rmse * da)


def count_decimal_places(number):
    number_str = str(number)

    if '.' in number_str:
        integer_part, fractional_part = number_str.split('.')
        return len(fractional_part)
    else:
        # If there's no decimal point, return 0
        return 0


def scale_values(v: np) -> (float, np):
    avg = np.mean(v)
    k = ValiConfig.SCALE_FACTOR
    return float(avg), np.array([np.tanh(k * (x - avg)) for x in v])


def scale_data_structure(ds: list[list]) -> (list[float], list[int], np):
    scaled_data_structure = []
    averages = []
    dp_decimal_places = []

    for dp in ds:
        avg, scaled_data_point = scale_values(dp)
        averages.append(avg)
        dp_decimal_places.append(count_decimal_places(dp[0]))
        scaled_data_structure.append(scaled_data_point)
    return averages, dp_decimal_places, np.array(scaled_data_structure)


def unscale_values(avg: float, decimal_places: int, v: np) -> np:
    k = ValiConfig.SCALE_FACTOR
    return np.array([np.round(avg + (1 / k) * np.arctanh(x), decimals=decimal_places) for x in v])


def unscale_data_structure(avgs: list[float], dp_decimal_places: list[int], sds: np) -> np:
    usds = []
    for i, dp in enumerate(sds):
        usds.append(unscale_values(avgs[i], dp_decimal_places[i], dp))
    return usds


# def scale_scores(scores: dict[str, float]) -> dict[str, float]:
#     avg_score = sum([score for miner_uid, score in scores.items()]) / len(scores)
#     scaled_scores_map = {}
#     for miner_uid, score in scores.items():
#         scaled_scores_map[miner_uid] = 1 - math.e ** (-1 / (score / avg_score))
#     return scaled_scores_map


def main2(config):
    # Set up logging with the provided configuration and directory.
    bt_logger.info(
        f"Running validator for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint} with config:")
    # Log the configuration for reference.
    bt_logger.info(config)

    # These are core Bittensor classes to interact with the network.
    bt_logger.info("Setting up bittensor objects.")

    # The wallet holds the cryptographic key pairs for the validator.
    wallet = bt.wallet(config=config)
    bt_logger.info(f"Wallet: {wallet}")

    # The subtensor is our connection to the Bittensor blockchain.
    subtensor = bt.subtensor(config=config)
    bt_logger.info(f"Subtensor: {subtensor}")

    # Dendrite is the RPC client; it lets us send messages to other nodes (axons) in the network.
    dendrite = bt.dendrite(wallet=wallet)
    bt_logger.info(f"Dendrite: {dendrite}")

    # The metagraph holds the state of the network, letting us know about other miners.
    metagraph = subtensor.metagraph(config.netuid)
    bt_logger.info(f"Metagraph: {metagraph}")

    # Connect the validator to the network
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt_logger.error(
            f"\nYour validator: {wallet} if not registered to chain connection: {subtensor} \nRun btcli register and try again.")
        exit()
    else:
        # Each miner gets a unique identity (UID) in the network for differentiation.
        my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
        bt_logger.info(f"Running validator on uid: {my_subnet_uid}")

    # Set up initial scoring weights for validation
    bt_logger.info("Building validation weights.")
    scores = torch.ones_like(metagraph.S, dtype=torch.float32)
    bt_logger.info(f"Weights: {scores}")

    days = ValiConfig.HISTORICAL_DATA_LOOKBACK_DAYS

    ts_ranges = TimeUtil.convert_range_timestamps_to_millis(
        TimeUtil.generate_range_timestamps(
            TimeUtil.generate_start_timestamp(days), days))

    high, low, close, volume = [], [], [], []
    data_structure = [high, low, close, volume]

    for ts_range in ts_ranges:
        BinanceData.convert_output_to_data_points(data_structure,
                                                  BinanceData().get_historical_data(start=ts_range[0],
                                                                                    end=ts_range[1]).json())

    print(data_structure)

    weighted_rmse()

    if len(requests) > 0:
        for request in requests:
            request_obj = verify_and_return_request_object(request)
            client_stream_hash = get_client_stream_hash(request_obj.client_id,
                                                        request_obj.stream_type)

            if isinstance(request_obj, PredictTrainingRequest):
                bt.logging.info("Predicting training request.")

                req_uuid = uuid.uuid4()

                # TODO - convert samples to tensor
                predict_training_proto = Forward(
                    request_uuid=req_uuid,
                    stream_type=client_stream_hash,
                    samples=request_obj.samples,
                    prediction_size=request_obj.prediction_size
                )

                try:
                    # step 1: send response to network and get responses
                    # Broadcast a query to all miners on the network.
                    responses = dendrite.query(
                        metagraph.axons,
                        predict_training_proto,  # Construct a dummy query.
                        # All responses have the deserialize function called on them before returning.
                        deserialize=True,
                    )

                    # TODO - validation that the predictions are in the same type dtype as the samples?

                    # step 2: don't score, just return
                    predictions_response = format_training_predictions_response(metagraph, responses)

                    # formatting response object
                    response_body = {
                        "request_uuid": req_uuid,
                        "predictions": predictions_response
                    }

                # If we encounter an unexpected error, log it for debugging.
                except RuntimeError as e:
                    bt.logging.error(e)
                    traceback.print_exc()

            if isinstance(request_obj, PredictLiveRequest):
                bt.logging.info("Processing training predictions.")

                # step 1: score returning results against outcome
                predictions = request_obj.predictions
                prediction_results = request_obj.prediction_results

                miner_scores = {}

                for miner_uid, miner_preds in predictions.items():
                    miner_scores[miner_uid] = score_response(miner_preds, prediction_results)

                scaled_miner_scores = scale_scores(miner_scores)

                # step 2: check and update vali records

                # step 1: get predictions on samples
                predict_live_proto = Forward(
                    request_uuid=request_obj.request_uuid,
                    stream_type=request_obj.stream_type,
                    samples=request_obj.samples,
                    prediction_size=request_obj.prediction_size
                )

                try:
                    # step 2: send response to network and get responses
                    # Broadcast a query to all miners on the network.
                    responses = dendrite.query(
                        metagraph.axons,
                        predict_training_proto,  # Construct a dummy query.
                        # All responses have the deserialize function called on them before returning.
                        deserialize=True,
                    )

                    # step 3: score responses
                    for i, resp_i in enumerate(responses):
                        if isinstance(resp_i, Forward):
                            rmse_scores[i] = score_response(used_t_s_testing_results, resp_i.predictions.tolist())

                    # step 4: scale rmse scores
                    scale_rmse_scores(rmse_scores)

                # If we encounter an unexpected error, log it for debugging.
                except RuntimeError as e:
                    bt.logging.error(e)
                    traceback.print_exc()

                try:
                    # Broadcast a query to all miners on the network.
                    responses = dendrite.query(
                        # Send the query to all axons in the network.
                        metagraph.axons,
                        # Construct a dummy query.
                        template.protocol.Dummy(dummy_input=step),  # Construct a dummy query.
                        # All responses have the deserialize function called on them before returning.
                        deserialize=True,
                    )

                    # Log the results for monitoring purposes.
                    bt.logging.info(f"Received dummy responses: {responses}")

                    # TODO(developer): Define how the validator scores responses.
                    # Adjust the scores based on responses from miners.
                    for i, resp_i in enumerate(responses):
                        # Initialize the score for the current miner's response.
                        score = 0

                        # Check if the miner has provided the correct response by doubling the dummy input.
                        # If correct, set their score for this round to 1.
                        if resp_i == step * 2:
                            score = 1

                        # Update the global score of the miner.
                        # This score contributes to the miner's weight in the network.
                        # A higher weight means that the miner has been consistently responding correctly.
                        scores[i] = alpha * scores[i] + (1 - alpha) * 0

                    # Periodically update the weights on the Bittensor blockchain.
                    if (step + 1) % 100 == 0:
                        # TODO(developer): Define how the validator normalizes scores before setting weights.
                        weights = torch.nn.functional.normalize(scores, p=1.0, dim=0)
                        bt.logging.info(f"Setting weights: {weights}")
                        # This is a crucial step that updates the incentive mechanism on the Bittensor blockchain.
                        # Miners with higher scores (or weights) receive a larger share of TAO rewards on this subnet.
                        subtensor.set_weights(
                            netuid=config.netuid,  # Subnet to set weights on.
                            wallet=wallet,  # Wallet to sign set weights using hotkey.
                            uids=metagraph.uids,  # Uids of the miners to set weights for.
                            weights=weights  # Weights to set for the miners.
                        )

                    # End the current step and prepare for the next iteration.
                    step += 1
                    # Resync our local state with the latest state from the blockchain.
                    metagraph = subtensor.metagraph(config.netuid)
                    # Sleep for a duration equivalent to the block time (i.e., time between successive blocks).
                    time.sleep(bt.__blocktime__)

                # If we encounter an unexpected error, log it for debugging.
                except RuntimeError as e:
                    bt.logging.error(e)
                    traceback.print_exc()

                # If the user interrupts the program, gracefully exit.
                except KeyboardInterrupt:
                    bt.logging.success("Keyboard interrupt detected. Exiting validator.")
                    exit()
            elif type(request_obj) == Backward_Request:
                pass
    else:
        responses = [resp.is_success for resp in bt.dendrite()(metagraph.axons, Ping())]


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    config = get_config()
    bt_logger = bt.logging(config=config, logging_dir=config.full_path)
    # Run the main function.
    main(config)
