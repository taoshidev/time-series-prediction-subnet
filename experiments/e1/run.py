# developer: taoshi-tdougherty
# Copyright © 2024 Taoshi Inc
import numpy as np
import pandas as pd
import os
from prettytable import PrettyTable

from sklearn.pipeline import Pipeline
from sklearn import set_config
set_config(transform_output = "pandas") # want to keep things as pandas dataframes

from typing import Union

import argparse

from metrics.rmse import rmse
from metrics.mae import mae
from metrics.mape import mape
from metrics.weighted_rmse import weighted_rmse
from metrics.classification import accuracy, precision, recall, f1_score

from functional.crossvalidation import cv
from preprocessors import (
    resample_transformer,
    missing_values_transformer,
    interpolate_transformer,
    normalize_transformer,
    indicators_transformer,
    lags_transformer,
)

from e1.models import baseline, hand_strategy, tuned_strategy
import numpy as np

config = {
    "output_size": 100,
}

models = {
    "baseline": baseline.Model(**config),
    "hand_strategy": hand_strategy.Model(**config),
    "tuned_strategy": tuned_strategy.Model(**config),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the first experiment.")
    parser.add_argument("--data", type=str, help="The path to the data file.")
    parser.add_argument("--model", type=str, default="baseline", help="The model to use for the experiment.")
    parser.add_argument("--train_percentage", type=float, default=0.8, help="The percentage of the data to use for training.")
    args = parser.parse_args()

    assert args.train_percentage > 0 and args.train_percentage < 1, "The train percentage must be between 0 and 1."
    assert args.data is not None, "The data file must be provided."
    assert os.path.exists(args.data), "The data file does not exist."
    assert args.model in models, "The model must be one of the following: baseline, hand_strategy, tuned_strategy."
    
    model = models[args.model]

    data = pd.read_csv(args.data)
    data["Date"] = pd.to_datetime(data["Open Time"])
    data.drop("Open Time", axis=1, inplace=True)
    data = data.set_index("Date")

    ohlcv_columns = ["Open", "High", "Low", "Close", "Volume"]
    assert all([col in data.columns for col in ohlcv_columns]), "The data must contain the following columns: Open, High, Low, Close, Volume."

    # initially only keep the OHLCV data
    ohlcv_data = data[ohlcv_columns]

    ## preprocessing steps
    preprocessing_pipeline = Pipeline(steps=[
        ('resample', resample_transformer),
        ('remove_missing', missing_values_transformer),
        ('interpolate', interpolate_transformer),
        ('indicators', indicators_transformer),
        ('lags', lags_transformer),
        ('normalize', normalize_transformer),
    ])

    print(f"Preprocessing data with {len(ohlcv_data)} rows.")

    # preprocess the data
    ohlcv_preprocessed = preprocessing_pipeline.fit_transform(ohlcv_data)
    print(f"Data Columns: {ohlcv_preprocessed.columns}")

    # split the data into training and testing
    split_index = int(len(ohlcv_preprocessed) * args.train_percentage)

    # split for full validation set
    train = ohlcv_preprocessed.iloc[:split_index]
    test = ohlcv_preprocessed.iloc[split_index:]

    print(f"Splitting testing at date: {ohlcv_preprocessed.index[split_index]}")
    
    folds = cv(len(train))
    rmse_values = []
    weighted_rmse_values = []
    mae_values = []
    mape_values = []
    accuracy_values = []
    precision_values = []
    recall_values = []
    f1_score_values = []

    for train_index, test_index in folds:
        train_subset = train.iloc[train_index]
        test_subset = train.iloc[test_index] # just want to keep the close price

        model.fit(train_subset.drop(columns=["Close"]), train_subset[['Close']])
        pred = model.predict(test_subset.drop(columns=["Close"]))

        gt = np.array(test_subset['Close'])
        prior_gt = np.array(test_subset['Close_lag_1'])
        predicted = np.array(pred)

        real_growth = gt > prior_gt
        predicted_growth = predicted > prior_gt

        rmse_values.append(rmse(gt, predicted))
        weighted_rmse_values.append(weighted_rmse(gt, predicted))
        mae_values.append(mae(gt, predicted))
        mape_values.append(mape(gt, predicted))
        accuracy_values.append(accuracy(real_growth, predicted_growth))
        precision_values.append(precision(real_growth, predicted_growth))
        recall_values.append(recall(real_growth, predicted_growth))
        f1_score_values.append(f1_score(real_growth, predicted_growth))

    # Assuming rmse_values, mae_values, etc., are defined
    metrics = [
        ["RMSE", np.mean(rmse_values), np.std(rmse_values)],
        ["WRMSE", np.mean(weighted_rmse_values), np.std(weighted_rmse_values)],
        ["MAE", np.mean(mae_values), np.std(mae_values)],
        ["MAPE", np.mean(mape_values), np.std(mape_values)],
        ["Accuracy", np.mean(accuracy_values), np.std(accuracy_values)],
        ["Precision", np.mean(precision_values), np.std(precision_values)],
        ["Recall", np.mean(recall_values), np.std(recall_values)],
        ["F1 Score", np.mean(f1_score_values), np.std(f1_score_values)],
    ]

    # Create a PrettyTable with field names
    table = PrettyTable(["Metric", "Mean", "Std Dev"])

    # Format each row's numeric values to scientific notation and add to the table
    for metric_name, mean, std_dev in metrics:
        formatted_mean = f"{mean:.3f}"  # Format to scientific notation with 2 decimal places
        formatted_std_dev = f"{std_dev:.3f}"  # Format to scientific notation with 2 decimal places
        table.add_row([metric_name, formatted_mean, formatted_std_dev])

    # Print the table
    print(table)

