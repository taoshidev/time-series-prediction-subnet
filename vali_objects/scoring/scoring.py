# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Taoshi
# Copyright © 2023 TARVIS Labs, LLC

import math
import numpy as np


class Scoring:

    @staticmethod
    def calculate_weighted_rmse(predictions: np, actual: np) -> float:
        predictions = np.array(predictions)
        actual = np.array(actual)

        k = 0.01

        weights = np.exp(-k * np.arange(len(predictions)))

        weighted_squared_errors = weights * (predictions - actual) ** 2
        weighted_rmse = np.sqrt(np.sum(weighted_squared_errors) / np.sum(weights))

        return weighted_rmse

    # @staticmethod
    # def calculate_directional_accuracy(predictions: np, actual: np) -> float:
    #     pred_len = len(predictions)
    #
    #     pred_dir = np.sign([predictions[i] - predictions[i - 1] for i in range(1, pred_len)])
    #     actual_dir = np.sign([actual[i] - actual[i - 1] for i in range(1, pred_len)])
    #
    #     correct_directions = pred_len-1
    #     for i in range(0, pred_len-1):
    #         correct_directions += actual_dir[i] == pred_dir[i]
    #
    #     return correct_directions / pred_len

    @staticmethod
    def score_response(predictions: np, actual: np) -> float:
        if len(predictions) != len(actual):
            return 0

        rmse = Scoring.calculate_weighted_rmse(predictions, actual)

        # remove da for now
        # da = Scoring.calculate_directional_accuracy(predictions, actual)
        # geometric mean
        # return np.sqrt(rmse * da)

        return rmse

    @staticmethod
    def scale_scores(scores: dict[str, float]) -> dict[str, float]:
        avg_score = sum([score for miner_uid, score in scores.items()]) / len(scores)
        scaled_scores_map = {}
        for miner_uid, score in scores.items():
            scaled_scores_map[miner_uid] = 1 - math.e ** (-1 / (score / avg_score))
        return scaled_scores_map

