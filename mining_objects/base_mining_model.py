# developer: taoshi-mbrown
# Copyright © 2023 Taoshi Inc
from keras.callbacks import Callback, EarlyStopping
from keras.layers import Dense, Dropout, Layer, LSTM
from keras.mixed_precision import Policy
from keras.models import load_model, Sequential
from keras.optimizers import Adam
import numpy as np
from numpy import ndarray
import resource
import tensorflow as tf


def _get_dataset_options():
    dataset_options = tf.data.Options()
    dataset_options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )
    return dataset_options


_DATASET_OPTIONS = _get_dataset_options()


class ResourceUsageCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        main_memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        main_memory_usage /= 1024
        gpu_memory_usage = tf.config.experimental.get_memory_info("GPU:0")["current"]
        gpu_memory_usage /= 1048576
        print(f" - Memory (MiB): {main_memory_usage:,.2f}, {gpu_memory_usage:,.2f}")


def _get_sparse_indexes(prediction_length: int, prediction_count: int):
    if prediction_count > prediction_length:
        raise ValueError("prediction_count cannot be greater than prediction_length.")

    results = [0] * prediction_count

    if prediction_count > 1:
        stride = int(prediction_length / (prediction_count - 1))
        prediction_index = stride

        for i in range(1, prediction_count - 1):
            results[i] = prediction_index
            prediction_index += stride

        results[-1] = prediction_length - 1

    return results


class BaseMiningModel:
    def __init__(
        self,
        filename: str,
        mode: str,
        feature_count: int,
        sample_count: int,
        prediction_feature_count: int,
        prediction_count: int,
        prediction_length: int,
        layers: list[list[int, float]] = None,
        learning_rate: float = 0.01,
        display_memory_usage: bool = False,
        dtype: np.dtype | Policy = Policy("float32"),
    ):
        input_shape = (None, sample_count, feature_count)
        output_length = prediction_feature_count * prediction_count
        output_shape = (None, output_length)

        self._read_only = "w" not in mode

        if "r" in mode:
            model = load_model(filename)
            first_layer: Layer = model.layers[0]
            output_layer: Layer = model.layers[-1]

            if first_layer.input_shape != input_shape:
                raise ValueError(
                    f"sample_count {sample_count} and "
                    f"feature_count {feature_count} "
                    "do not match the loaded model's "
                    f"input_shape of {first_layer.input_shape}."
                )

            if output_layer.output_shape != output_shape:
                raise ValueError(
                    f"prediction_feature_count {prediction_feature_count} and "
                    f"prediction_count {prediction_count} "
                    f"for an output_length {output_length}"
                    "does not match the loaded model's "
                    f"output_shape of {output_layer.output_shape}."
                )

        else:
            if not layers:
                raise ValueError("layers must be defined when creating a new model.")

            model = Sequential()
            last_lstm_index = len(layers) - 1
            for i, (lstm_units, dropout_rate) in enumerate(layers):
                return_sequences = i != last_lstm_index
                if i == 0:
                    first_layer = LSTM(
                        dtype=dtype,
                        units=lstm_units,
                        input_shape=(sample_count, feature_count),
                        return_sequences=return_sequences,
                    )
                    model.add(first_layer)
                else:
                    if dropout_rate != 0:
                        model.add(Dropout(dtype=dtype, rate=dropout_rate))
                    model.add(
                        LSTM(
                            dtype=dtype,
                            units=lstm_units,
                            return_sequences=return_sequences,
                        )
                    )

            model.add(Dense(dtype=dtype, units=output_length))

            optimizer = Adam(learning_rate=learning_rate)
            model.compile(
                optimizer=optimizer, loss="mean_squared_error", run_eagerly=False
            )

        self._model = model
        self._filename = filename
        self._feature_count = feature_count
        self.sample_count = sample_count
        self._prediction_feature_count = prediction_feature_count
        self.prediction_count = prediction_count
        self.prediction_length = prediction_length
        self._display_memory_usage = display_memory_usage

        self.prediction_sparse_indexes = _get_sparse_indexes(
            prediction_length, prediction_count
        )

        if prediction_count == prediction_length:
            self._interpolation_indexes = None
        else:
            self._interpolation_indexes = np.arange(prediction_length)

    def train(
        self,
        training_dataset,
        epochs: int = 100,
        patience: int = 10,
    ):
        if self._read_only:
            raise RuntimeError("Training not supported in read only mode.")

        # Prevent warnings about using data sharding for multiple GPUs
        training_dataset = training_dataset.with_options(_DATASET_OPTIONS)

        early_stopping = EarlyStopping(
            monitor="loss", patience=patience, restore_best_weights=True
        )

        callbacks = [
            early_stopping,
        ]

        if self._display_memory_usage:
            callbacks.append(ResourceUsageCallback())

        self._model.fit(
            training_dataset,
            epochs=epochs,
            callbacks=callbacks,
        )

        self._model.save(self._filename)

    def predict(self, model_input: ndarray, dtype: np.dtype = np.float32) -> ndarray:
        window = model_input[-self.sample_count :]
        window = window.reshape(1, self.sample_count, self._feature_count)

        prediction = self._model.predict(window)
        prediction.shape = (self.prediction_count, self._prediction_feature_count)

        if self._interpolation_indexes is None:
            return prediction

        else:
            sparse_prediction = prediction.T
            full_prediction = np.empty(
                shape=(self._prediction_feature_count, self.prediction_length),
                dtype=dtype,
            )

            for i in range(self._prediction_feature_count):
                full_prediction[i] = np.interp(
                    self._interpolation_indexes,
                    self.prediction_sparse_indexes,
                    sparse_prediction[i],
                )

            return full_prediction.T
