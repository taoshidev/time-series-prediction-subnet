# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Taoshi
# Copyright © 2023 TARVIS Labs, LLC

import numpy as np

from vali_config import ValiConfig


class Scaling:

    @staticmethod
    def count_decimal_places(number):
        number_str = str(number)

        if '.' in number_str:
            integer_part, fractional_part = number_str.split('.')
            return len(fractional_part)
        else:
            # If there's no decimal point, return 0
            return 0

    @staticmethod
    def scale_values(v: np) -> (float, np):
        avg = np.mean(v)
        k = ValiConfig.SCALE_FACTOR
        return float(avg), np.array([np.tanh(k * (x - avg)) for x in v])

    @staticmethod
    def scale_data_structure(ds: list[list]) -> (list[float], list[int], np):
        scaled_data_structure = []
        averages = []
        dp_decimal_places = []

        for dp in ds:
            avg, scaled_data_point = Scaling.scale_values(dp)
            averages.append(avg)
            dp_decimal_places.append(Scaling.count_decimal_places(dp[0]))
            scaled_data_structure.append(scaled_data_point)
        return averages, dp_decimal_places, np.array(scaled_data_structure)

    @staticmethod
    def unscale_values(avg: float, decimal_places: int, v: np) -> np:
        k = ValiConfig.SCALE_FACTOR
        return np.array([np.round(avg + (1 / k) * np.arctanh(x), decimals=decimal_places) for x in v])

    @staticmethod
    def unscale_data_structure(avgs: list[float], dp_decimal_places: list[int], sds: np) -> np:
        usds = []
        for i, dp in enumerate(sds):
            usds.append(Scaling.unscale_values(avgs[i], dp_decimal_places[i], dp))
        return usds
