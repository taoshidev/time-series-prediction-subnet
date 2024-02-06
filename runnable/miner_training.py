# developer: taoshi-mbrown
# Copyright © 2024 Taoshi, LLC
from features import FeatureCollector
from feature_sources import BinaryFileFeatureStorage
from keras.mixed_precision import Policy
from keras.preprocessing import timeseries_dataset_from_array
from mining_objects import BaseMiningModel
from mining_objects.streams.btcusd_5m import (
    historical_feature_ids,
    INTERVAL_MS,
    model_feature_ids,
    model_feature_scaler,
    prediction_feature_ids,
    PREDICTION_COUNT,
    PREDICTION_LENGTH,
    SAMPLE_COUNT,
    spontaneous_feature_sources,
)
import numpy as np
import tensorflow as tf
from time_util import datetime


def main():
    process_start_time = datetime.now()
    print(f"Training started at {process_start_time}")

    # Prevents main memory leak and consequent slowdown
    # Remove if this leak is fixed in later versions of TensorFlow
    tf.config.run_functions_eagerly(False)

    _DATA_PRECISION = np.float32
    _MODEL_PRECISION = Policy("mixed_float16")

    _SCENARIOS_PER_BATCH = 128
    # Adjust to a multiple of the number of GPUs
    _BATCHES_PER_CHUNK = 32
    _LAYER_UNITS = 512
    _LEARNING_RATE = 0.001
    _TRAINING_EPOCHS = 100
    _TRAINING_PATIENCE = 10

    prediction_feature_count = len(prediction_feature_ids)

    print("Opening historical data...")

    historical_feature_storage = BinaryFileFeatureStorage(
        filename="historical_financial_data/data_training.taosfs",
        mode="r",
        feature_ids=historical_feature_ids,
    )

    historical_start_time_ms = historical_feature_storage.get_start_time_ms()
    historical_sample_count = historical_feature_storage.get_sample_count()

    training_feature_sources = [
        historical_feature_storage,
        *spontaneous_feature_sources,
    ]

    feature_collector = FeatureCollector(training_feature_sources)
    feature_count = feature_collector.feature_count

    if feature_collector.feature_ids != model_feature_ids:
        raise RuntimeError("Features of historical data do not match model.")

    print("Creating model...")

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = BaseMiningModel(
            filename="mining_models/model_v5.h5",
            mode="w",
            feature_count=feature_count,
            sample_count=SAMPLE_COUNT,
            prediction_feature_count=prediction_feature_count,
            prediction_count=PREDICTION_COUNT,
            prediction_length=PREDICTION_LENGTH,
            layers=[
                [_LAYER_UNITS, 0],
                [_LAYER_UNITS, 0.2],
                [_LAYER_UNITS, 0.4],
                [_LAYER_UNITS, 0.6],
            ],
            learning_rate=_LEARNING_RATE,
            dtype=_MODEL_PRECISION,
        )

    # A chunk contains batches.
    # A batch contains scenarios.
    # A scenario contains a training sequence and targets.
    #
    # Example of chunks containing 3 batches, with 2 scenarios per batch:
    #
    # start |--------| chunk n                         |--------------------------| end
    # of    |        |training data        |           |                          | of
    # file  |        |                |target data     |                          | file
    #       |batch 1:|scenario 1      |targets    |    |                          |
    #       |         |scenario 2     .|targets    |   |                          |
    #       |batch 2:  |scenario 3    . |targets    |  |                          |
    #       |           |scenario 4   .  |targets    | |                          |
    #       |batch 3:    |scenario 5  .   |targets    ||                          |
    #       |             |scenario 6 .    |targets    |                          |
    #       |------------------------------| chunk n+1                       |----|
    # (additional chunks)
    scenarios_per_chunk = _BATCHES_PER_CHUNK * _SCENARIOS_PER_BATCH
    training_data_length = SAMPLE_COUNT + scenarios_per_chunk - 1
    target_data_length = PREDICTION_LENGTH + scenarios_per_chunk - 1
    chunk_length = training_data_length + target_data_length

    targets = np.empty(
        shape=(scenarios_per_chunk, PREDICTION_COUNT, prediction_feature_count),
        dtype=_DATA_PRECISION,
    )

    chunk_start_ms = historical_start_time_ms
    chunk_start = 0
    chunk_end = chunk_length
    while True:
        if chunk_end > historical_sample_count:
            chunk_end = historical_sample_count
            chunk_length = chunk_end - chunk_start
            scenarios_per_chunk = (
                int((chunk_length - SAMPLE_COUNT - PREDICTION_LENGTH) / 2) + 1
            )

            if scenarios_per_chunk <= 0:
                break

            training_data_length = SAMPLE_COUNT + scenarios_per_chunk - 1
            targets = np.empty(
                shape=(scenarios_per_chunk, PREDICTION_COUNT, prediction_feature_count),
                dtype=_DATA_PRECISION,
            )

        chunk_start_datetime = datetime.fromtimestamp_ms(chunk_start_ms)
        print(f"Reading historical data for {chunk_start_datetime}...")

        chunk_samples = feature_collector.get_feature_samples(
            chunk_start_ms, INTERVAL_MS, chunk_length
        )

        model_feature_scaler.scale_feature_samples(chunk_samples)

        training_data = feature_collector.feature_samples_to_array(
            chunk_samples, stop=training_data_length, dtype=_DATA_PRECISION
        )

        target_data = feature_collector.feature_samples_to_array(
            chunk_samples,
            feature_ids=prediction_feature_ids,
            start=training_data_length,
            dtype=_DATA_PRECISION,
        )

        # Shape to simplify including multiple features in predictions
        targets.shape = (
            scenarios_per_chunk,
            PREDICTION_COUNT,
            prediction_feature_count,
        )

        # Populate the training targets
        sparse_indexes = model.prediction_sparse_indexes
        for scenario_index in range(scenarios_per_chunk):
            for prediction_index in range(PREDICTION_COUNT):
                target_data_index = scenario_index + sparse_indexes[prediction_index]
                targets[scenario_index, prediction_index] = target_data[
                    target_data_index
                ]

        # Flatten to interlace features within predictions to match flat model output
        targets.shape = (
            scenarios_per_chunk,
            PREDICTION_COUNT * prediction_feature_count,
        )

        training_dataset = timeseries_dataset_from_array(
            training_data,
            targets,
            sequence_length=SAMPLE_COUNT,
            batch_size=_SCENARIOS_PER_BATCH,
            shuffle=True,
        )

        print("Training...")

        model.train(
            training_dataset, epochs=_TRAINING_EPOCHS, patience=_TRAINING_PATIENCE
        )

        chunk_start_ms += INTERVAL_MS * (chunk_length - PREDICTION_LENGTH)
        chunk_start += chunk_length
        chunk_end += chunk_length

    process_end_time = datetime.now()
    print(f"Training ended at {process_end_time}")


if __name__ == "__main__":
    main()
