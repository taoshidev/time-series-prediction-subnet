# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

import numpy as np
from scipy.stats import norm
from mining_objects.mining_utils import MiningUtils
from time_util.time_util import TimeUtil
from vali_objects.dataclasses.client_request import ClientRequest

import matplotlib.pyplot as plt
import statistics

if __name__ == "__main__":

    curr_iter = 100000

    values_dict = {}
    values_list = []
    while True:

        client_request = ClientRequest(
            client_uuid="test_client_uuid",
            stream_type="BTCUSD-5m",
            topic_id=1,
            schema_id=1,
            feature_ids=[0.001, 0.002, 0.003, 0.004],
            prediction_size=100,
            additional_details={
                "tf": 5,
                "trade_pair": "BTCUSD"
            }
        )

        iter_add = 100

        data_structure = MiningUtils.get_file(
            "/runnable/historical_financial_data/data_training.pickle", True)
        data_structure = [data_structure[0][curr_iter:curr_iter + iter_add],
                          data_structure[1][curr_iter:curr_iter + iter_add],
                          data_structure[2][curr_iter:curr_iter + iter_add],
                          data_structure[3][curr_iter:curr_iter + iter_add],
                          data_structure[4][curr_iter:curr_iter + iter_add]]

        start_dt = TimeUtil.millis_to_timestamp(data_structure[0][0])
        end_dt = TimeUtil.millis_to_timestamp(data_structure[0][len(data_structure[0]) - 1])
        curr_iter += iter_add
        data_structure = np.array(data_structure)
        samples = data_structure

        if max(data_structure[1]) / min(data_structure[1]) < 1.05:
            standard_deviation = statistics.stdev(data_structure[1])
            values_list.append(standard_deviation)

        if curr_iter >= 340000:
            break

    mean = np.mean(values_list)
    std_dev_value = np.std(values_list)

    plt.hist(values_list, bins=1000, density=True, alpha=0.7, color='blue', edgecolor='black')

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = np.exp(-0.5 * ((x - mean) / std_dev_value) ** 2) / (std_dev_value * np.sqrt(2 * np.pi))
    plt.plot(x, p, 'k', linewidth=2, label='Standard Normal Distribution')

    q25 = norm.ppf(0.25, loc=mean, scale=std_dev_value)
    q50 = norm.ppf(0.50, loc=mean, scale=std_dev_value)
    q75 = norm.ppf(0.75, loc=mean, scale=std_dev_value)

    plt.axvline(q25, color='red', linestyle='--', label='25th Percentile (Q1)')
    plt.axvline(q50, color='green', linestyle='--', label='50th Percentile (Median)')
    plt.axvline(q75, color='purple', linestyle='--', label='75th Percentile (Q3)')

    plt.title("Histogram and Standard Normal Distribution with Quartiles")
    plt.xlabel("Values")
    plt.ylabel("Density")
    plt.legend()

    plt.show()

    print("25th Percentile (Q1):", q25)
    print("50th Percentile (Median):", q50)
    print("75th Percentile (Q3):", q75)

    percentiles = np.arange(1, 101, 1)
    values_at_percentiles = np.percentile(values_list, percentiles)

    for p, value_at_p in zip(percentiles, values_at_percentiles):
        plt.axvline(value_at_p, color='red', linestyle='--', alpha=0.3)

    plt.title("Histogram and Standard Normal Distribution with Linear Percentiles")
    plt.xlabel("Values")
    plt.ylabel("Density")
    plt.legend()

    plt.show()

    print(percentiles)
    print(values_at_percentiles)

    # Print values at every 1% step
    for p, value_at_p in zip(percentiles, values_at_percentiles):
        print(f"{p}th Percentile: {value_at_p}")






