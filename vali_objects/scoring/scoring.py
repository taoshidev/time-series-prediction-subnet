# developer: Taoshidev
# Copyright © 2023 Taoshi, LLC

import math
from typing import List, Tuple, Dict

import numpy as np

from vali_config import ValiConfig
from vali_objects.exceptions.incorrect_prediction_size_error import IncorrectPredictionSizeError
from vali_objects.exceptions.min_responses_exception import MinResponsesException
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils


class Scoring:

    @staticmethod
    def calculate_weighted_rmse(predictions: np, actual: np) -> float:
        predictions = np.array(predictions)
        actual = np.array(actual)

        k = ValiConfig.RMSE_WEIGHT

        weights = np.exp(-k * np.arange(len(predictions)))
        weighted_squared_errors = weights * (predictions - actual) ** 2
        weighted_rmse = np.sqrt(np.sum(weighted_squared_errors) / np.sum(weights))

        return weighted_rmse

    @staticmethod
    def calculate_directional_accuracy(predictions: np, actual: np) -> float:
        pred_len = len(predictions)

        pred_dir = np.sign([predictions[i] - predictions[i - 1] for i in range(1, pred_len)])
        actual_dir = np.sign([actual[i] - actual[i - 1] for i in range(1, pred_len)])

        correct_directions = 0
        for i in range(0, pred_len-1):
            correct_directions += actual_dir[i] == pred_dir[i]

        return correct_directions / (pred_len-1)

    @staticmethod
    def score_response(predictions: np, actual: np) -> float:
        if len(predictions) != len(actual) or len(predictions) == 0 or len(actual) < 2:
            raise IncorrectPredictionSizeError(f"the number of predictions or the number of responses "
                                               f"needed are incorrect: preds: '{len(predictions)}',"
                                               f" results: '{len(actual)}'")

        rmse = Scoring.calculate_weighted_rmse(predictions, actual)

        return rmse

    @staticmethod
    def scale_scores(scores: Dict[str, float]) -> Dict[str, float]:
        avg_score = sum([score for miner_uid, score in scores.items()]) / len(scores)
        scaled_scores_map = {}
        for miner_uid, score in scores.items():
            # handle case of a perfect score
            if score == 0:
                score = 0.00000001
            scaled_scores_map[miner_uid] = 1 - math.e ** (-1 / (score / avg_score))
        return scaled_scores_map

    @staticmethod
    def weigh_miner_scores(scores: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        if len(scores) == 1:
            return [(scores[0][0], 1.0)]

        min_score = min(score for _, score in scores)
        max_score = max(score for _, score in scores)

        normalized_scores = [(name, (score - min_score) / (max_score - min_score)) for name, score in scores]
        total_normalized_score = sum(score for _, score in normalized_scores)

        normalized_scores = [(name, round(score / total_normalized_score, 4)) for name, score in normalized_scores]

        return normalized_scores

    @staticmethod
    def simple_scale_scores(scores: Dict[str, float]) -> Dict[str, float]:
        if len(scores) <= 1:
            raise MinResponsesException("not enough responses")
        score_values = [score for miner_uid, score in scores.items()]
        min_score = min(score_values)
        max_score = max(score_values)

        return {miner_uid:  1 - ((score - min_score) / (max_score - min_score)) for miner_uid, score in scores.items()}

    @staticmethod
    def history_of_values() -> None | Dict[str, float]:
        # attempt to rebuild state using cmw objects
        pass

    @staticmethod
    def get_percentile(value, percentiles):
        for ind, range_value in enumerate(percentiles):
            percentile = (ind + 1) / 100
            if value < range_value:
                return percentile
        return 1

    @staticmethod
    def get_geometric_mean_of_percentile(ds: List[List[float]]):
        min_max_ranges_percentiled = ValiConfig.MIN_MAX_RANGES_PERCENTILED
        std_dev_ranges_percentiled = ValiConfig.STD_DEV_RANGES_PERCENTILED

        results_min_max = max(ds[1]) / min(ds[1])
        results_std_dev = np.std(ds[1])

        min_max_percentile = Scoring.get_percentile(results_min_max, min_max_ranges_percentiled)
        std_dev_percentile = Scoring.get_percentile(results_std_dev, std_dev_ranges_percentiled)

        return math.sqrt(min_max_percentile * std_dev_percentile)

    @staticmethod
    def update_weights_using_historical_distributions(scores: List[Tuple[str, float]], ds: List[List[float]]):

        vweights = ValiUtils.get_vali_weights_json()
        geometric_mean_of_percentile = Scoring.get_geometric_mean_of_percentile(ds)

        for score in scores:
            if score[0] in vweights:
                vweights[score[0]] = Scoring.basic_ema((vweights[score[0]] +
                                                        (score[1] * geometric_mean_of_percentile))
                                                       / (1 + geometric_mean_of_percentile), vweights[score[0]])
            else:
                vweights[score[0]] = score[1] * geometric_mean_of_percentile

        ValiUtils.set_vali_weights_bkp(vweights)
        return vweights, geometric_mean_of_percentile

    @staticmethod
    def basic_ema(current_value, previous_ema, length=24):
        alpha = 2 / (length + 1)
        ema = alpha * current_value + (1 - alpha) * previous_ema
        return ema
