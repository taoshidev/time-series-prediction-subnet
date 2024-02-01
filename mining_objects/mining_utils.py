import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from mining_objects.base_mining_model import BaseMiningModel,MiningModelNHITS
from vali_config import ValiConfig
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils

import pandas as pd 

def handler_to_model_input_format(ds):  
    '''Prepares the data for the model input
    '''
    df = pd.concat([pd.DataFrame(i) for i in ds],axis=1)
    df.columns = [
        'ds',
        'close',
        'high',
        'low',
        'volume',
    ]
    df.ds= pd.to_datetime(df.ds, unit='ms')
    timestamp = pd.to_datetime(df.ds, unit='ms')
    df['year'] = timestamp.dt.year
    df['month'] = timestamp.dt.month
    df['hour'] = timestamp.dt.hour
    df['minute'] = timestamp.dt.minute
    df['day'] = timestamp.dt.day
    df['dayofweek'] = timestamp.dt.isocalendar().day
    df['weekofyear'] = timestamp.dt.isocalendar().week
    df['quarter'] = timestamp.dt.quarter
    # Add more features as reuired
    df['unique_id'] ='BTCUSD'
    df['y'] = df['close']
    df = df.drop('close',axis=1)
    return df 




import pandas as pd
import numpy as np
from pandas.tseries.offsets import Minute

def prepare_futr_datset(df, num_rows=100):
    # Check if the DataFrame is empty or the 'ds' column doesn't exist
    if df.empty or 'ds' not in df.columns:
        raise ValueError("DataFrame is empty or 'ds' column is not present.")

    # Get the last datetime from the 'ds' column
    last_datetime = df['ds'].iloc[-1]

    # Generate a date range starting from the last datetime plus 5 minutes
    datetimes = pd.date_range(start=last_datetime + Minute(5), periods=num_rows, freq='5T')

    # Create a DataFrame with the new datetimes
    expected_futr_df = pd.DataFrame({'ds': datetimes})
    timestamp = pd.to_datetime(expected_futr_df.ds, unit='ms')
    expected_futr_df['year'] = timestamp.dt.year
    expected_futr_df['month'] = timestamp.dt.month
    expected_futr_df['hour'] = timestamp.dt.hour
    expected_futr_df['minute'] = timestamp.dt.minute
    expected_futr_df['day'] = timestamp.dt.day
    expected_futr_df['dayofweek'] = timestamp.dt.isocalendar().day
    expected_futr_df['weekofyear'] = timestamp.dt.isocalendar().week
    expected_futr_df['quarter'] = timestamp.dt.quarter
    expected_futr_df['unique_id'] = df['unique_id'][0]
    # Set other columns to NaN


    # Append the new DataFrame to the original one
    return expected_futr_df





class MiningUtils:

    @staticmethod
    def write_file(dir: str, file_name: str, vali_data: dict | object, is_pickle: bool = False) -> None:
        # will concat dir and file name
        ValiBkpUtils.make_dir(ValiConfig.BASE_DIR + dir)
        ValiBkpUtils.write_to_vali_dir(ValiConfig.BASE_DIR + dir + file_name, vali_data, is_pickle)

    @staticmethod
    def get_file(file, is_pickle: bool = True) -> dict | object:
        with open(ValiConfig.BASE_DIR + file, ValiBkpUtils.get_read_type(is_pickle)) as f:
            return pickle.load(f) if is_pickle else f.read()

    @staticmethod
    def scale_values(data_structure, scale_min, scale_max):
        s_ndarray = data_structure.T
        scaler = MinMaxScaler(feature_range=(scale_min, scale_max))

        if s_ndarray.ndim == 1:
            s_ndarray = s_ndarray.reshape(-1, 1)
        scaled_data = scaler.fit_transform(s_ndarray)
        scaled_data = scaled_data.T
        return scaled_data

    @staticmethod
    def scale_ds_0_100(data_structure):
        return MiningUtils.scale_values(np.array(data_structure), 0, 100)

    @staticmethod
    def trim_min_size(features):
        min_ind = 0
        min_length = 0
        for feature in features:
            for ind, value in enumerate(feature):
                if value is None or value == "":
                    if ind > min_ind:
                        min_ind = ind

    @staticmethod
    def plt_features(features, feature_length):
        NUM_COLORS = len(features)
        cm = plt.get_cmap('gist_rainbow')
        colors = [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)]
        color_chosen = 0

        for label, feature in features.items():
            plt.plot(feature_length, feature, label=label, color=colors[color_chosen])
            color_chosen += 1
        plt.legend()
        plt.show()

    @staticmethod
    def open_model_prediction_generation(samples, mining_details, prediction_size):
        model_samples = ValiUtils.get_standardized_ds()
        # trim the samples to the number of rows thats supposed to be for the current model
        for i in range(len(samples)):
            model_samples[i] = samples[i][-mining_details["rows"]:]

        model_samples = np.array(model_samples)
        prep_dataset = mining_details["features"](model_samples)
        # leverage base mining model class to generate predictions
        base_mining_model = BaseMiningModel(len(prep_dataset.T)) \
            .set_window_size(mining_details["window_size"]) \
            .set_model_dir(mining_details["model_dir"]) \
            .load_model()

        # scale the data to 0 - 1
        sds_ndarray = samples.T
        scaler = MinMaxScaler(feature_range=(0, 1))

        scaled_data = scaler.fit_transform(sds_ndarray)
        scaled_data = scaled_data.T
        prep_dataset_cp = BaseMiningModel.base_model_dataset(scaled_data)

        # generate equally sloped line between last close and predicted end close
        last_close = prep_dataset_cp.T[0][len(prep_dataset_cp) - 1]
        predicted_close = base_mining_model.predict(prep_dataset_cp, )[0].tolist()[0]
        total_movement = predicted_close - last_close
        total_movement_increment = total_movement / prediction_size

        predicted_closes = []
        curr_price = last_close
        for x in range(prediction_size):
            curr_price += total_movement_increment
            predicted_closes.append(curr_price[0])

        close_column = samples[1].reshape(-1, 1)

        scaler = MinMaxScaler()
        scaler.fit(close_column)

        # inverse the scale back to raw closing price scale
        reshaped_predicted_closes = np.array(predicted_closes).reshape(-1, 1)
        predicted_closes = scaler.inverse_transform(reshaped_predicted_closes)

        predicted_closes = predicted_closes.T.tolist()[0]
        return predicted_closes
    
    
    @staticmethod
    def open_model_prediction_generation_nhits(samples, mining_details, prediction_size):
   
        input = handler_to_model_input_format(samples)
        futr = prepare_futr_datset(input)

        # leverage base mining model class to generate predictions
        base_mining_model = MiningModelNHITS() \
            .set_model_dir(mining_details["model_dir"]) \
            .load_model()

        predicted_closes = MiningModelNHITS.predict(df=input,futr=futr)

        predicted_closes = predicted_closes['NHITS'].tolist()[0]

        return predicted_closes # needs to be a list
