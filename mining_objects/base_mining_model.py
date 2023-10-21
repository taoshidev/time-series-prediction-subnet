# developer: Taoshidev
# Copyright © 2023 Taoshi, LLC

import numpy as np
import tensorflow
from numpy import ndarray

class BaseMiningModel:
    def __init__(self, features):
        self.neurons = 50
        self.features = features
        self.loaded_model = None
        self.window_size = 100
        self.model_dir = None
        self.batch_size = 16
        self.learning_rate = 0.01

    def set_neurons(self, neurons):
        self.neurons = neurons

    def set_window_size(self, window_size):
        self.window_size = window_size
        return self

    def set_model_dir(self, model, stream_id=None):
        if model is None and stream_id is not None:
            self.model_dir = f'mining_models/{stream_id}.keras'
        elif model is not None:
            self.model_dir = model
        else:
            raise Exception("stream_id is not provided to define model")
        return self

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        return self

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        return self

    def load_model(self):
        self.loaded_model = tensorflow.keras.models.load_model(self.model_dir)
        return self

    def train(self, data: ndarray, epochs: 100):
        try:
            model = tensorflow.keras.models.load_model(self.model_dir)
        except OSError:
            model = None

        if model is None:
            model = tensorflow.keras.models.Sequential()
            model.add(tensorflow.keras.layers.LSTM(self.neurons, input_shape=(self.window_size, self.features)))
            model.add(tensorflow.keras.layers.Dense(self.features))
            optimizer = tensorflow.keras.optimizers.Adam(learning_rate=self.learning_rate)
            model.compile(optimizer=optimizer, loss='mean_squared_error')

        X_train, Y_train = [], []

        for i in range(len(data) - self.window_size):
            input_sequence = data[i:i + self.window_size]
            target_value = data[i + self.window_size]

            X_train.append(input_sequence)
            Y_train.append(target_value)

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)

        X_train = tensorflow.convert_to_tensor(np.array(X_train, dtype=np.float32))
        Y_train = tensorflow.convert_to_tensor(np.array(Y_train, dtype=np.float32))

        model.fit(X_train, Y_train, epochs=epochs, batch_size=self.batch_size)
        model.save(self.model_dir)

    def predict(self, data: ndarray):
        predictions = []

        window_data = data[-self.window_size:]
        window_data = window_data.reshape(1, self.window_size, self.features)

        predicted_value = self.loaded_model.predict(window_data)
        predictions.append(predicted_value)

        return predictions

    @staticmethod
    def base_model_dataset(samples):
        min_cutoff = 0

        cutoff_close = samples.tolist()[1][min_cutoff:]
        cutoff_high = samples.tolist()[2][min_cutoff:]
        cutoff_low = samples.tolist()[3][min_cutoff:]
        cutoff_volume = samples.tolist()[4][min_cutoff:]

        return np.array([cutoff_close,
                                 cutoff_high,
                                 cutoff_low,
                                 cutoff_volume]).T
