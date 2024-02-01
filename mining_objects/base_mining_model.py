# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

import numpy as np
import tensorflow
from numpy import ndarray
from neuralforecast import NeuralForecast

class BaseMiningModel:
    def __init__(self, features):
        self.neurons = [[50,0]]
        self.features = features
        self.loaded_model = None
        self.window_size = 100
        self.model_dir = None
        self.batch_size = 16
        self.learning_rate = 0.01

    def set_neurons(self, neurons):
        self.neurons = neurons
        return self

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

    def train(self, data: ndarray, epochs: int = 100):
        try:
            model = tensorflow.keras.models.load_model(self.model_dir)
        except OSError:
            model = None

        output_sequence_length = 100

        if model is None:
            model = tensorflow.keras.models.Sequential()

            if len(self.neurons) > 1:
                model.add(tensorflow.keras.layers.LSTM(self.neurons[0][0],
                                                       input_shape=(self.window_size, self.features),
                                                       return_sequences=True))
                for ind, stack in enumerate(self.neurons[1:]):
                    return_sequences = True
                    if ind+1 == len(self.neurons)-1:
                        return_sequences = False
                    model.add(tensorflow.keras.layers.Dropout(stack[1]))
                    model.add(tensorflow.keras.layers.LSTM(stack[0], return_sequences=return_sequences))
            else:
                model.add(tensorflow.keras.layers.LSTM(self.neurons[0][0],
                                                       input_shape=(self.window_size, self.features)))

            model.add(tensorflow.keras.layers.Dense(1))

            optimizer = tensorflow.keras.optimizers.Adam(learning_rate=self.learning_rate)
            model.compile(optimizer=optimizer, loss='mean_squared_error')

        X_train, Y_train = [], []

        X_train_data = data
        Y_train_data = data.T[0].T

        for i in range(len(Y_train_data) - output_sequence_length - self.window_size):
            target_sequence = Y_train_data[i+self.window_size+output_sequence_length:i+self.window_size+output_sequence_length+1]
            Y_train.append(target_sequence)

        for i in range(len(X_train_data) - output_sequence_length - self.window_size):
            input_sequence = X_train_data[i:i+self.window_size]
            X_train.append(input_sequence)

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)

        X_train = tensorflow.convert_to_tensor(np.array(X_train, dtype=np.float32))
        Y_train = tensorflow.convert_to_tensor(np.array(Y_train, dtype=np.float32))

        early_stopping = tensorflow.keras.callbacks.EarlyStopping(monitor="loss", patience=10,

                                                                  restore_best_weights=True)

        model.fit(X_train, Y_train, epochs=epochs, batch_size=self.batch_size, callbacks=[early_stopping])
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



class MiningModelNHITS:
    def __init__(self):
        self.neurons = [[50,0]] # dont think I need
        self.features = 4 # dont think I need
        self.loaded_model = None
        self.window_size = 100 
        self.model_dir = None
        self.batch_size = 16 # dont think I need
        self.learning_rate = 0.01 # dont think I need

    def set_neurons(self, neurons):
        self.neurons = neurons
        return self

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

    def load_model(self):
        self.loaded_model = NeuralForecast.load(self.model_dir)

        return self

    def predict(self, df,futr):

        predictions =  self.loaded_model.predict(df,futr_df=futr.reset_index())
   
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