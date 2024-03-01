# developer: Taoshidev
# Copyright © 2023 Taoshi Inc
import math
import numpy as np
from numpy import ndarray
from vali_config import ValiConfig, ValiStream
from vali_objects.exceptions.incorrect_prediction_size_error import (
    IncorrectPredictionSizeError,
)
from vali_objects.exceptions.min_responses_exception import MinResponsesException
from vali_objects.utils.vali_utils import ValiUtils


class Scoring:
    @staticmethod
    def calculate_weighted_rmse(predictions, actual) -> float:
        predictions = np.array(predictions)
        actual = np.array(actual)

        k = ValiConfig.RMSE_WEIGHT

        weights = np.exp(-k * np.arange(len(predictions)))
        weighted_squared_errors = weights * (predictions - actual) ** 2
        weighted_rmse = np.sqrt(np.sum(weighted_squared_errors) / np.sum(weights))

        return weighted_rmse

    @staticmethod
    def calculate_directional_accuracy(predictions, actual) -> float:
        pred_len = len(predictions)

        pred_dir = np.sign(
            [predictions[i] - predictions[i - 1] for i in range(1, pred_len)]
        )
        actual_dir = np.sign([actual[i] - actual[i - 1] for i in range(1, pred_len)])

        correct_directions = 0
        for i in range(0, pred_len - 1):
            correct_directions += actual_dir[i] == pred_dir[i]

        return correct_directions / (pred_len - 1)

    @staticmethod
    def score_response(predictions, actual) -> float:
        if len(predictions) != len(actual) or len(predictions) == 0 or len(actual) < 2:
            raise IncorrectPredictionSizeError(
                f"the number of predictions or the number of responses "
                f"needed are incorrect: preds: '{len(predictions)}',"
                f" results: '{len(actual)}'"
            )

        rmse = Scoring.calculate_weighted_rmse(predictions, actual)

        return rmse

    @staticmethod
    def scale_scores(scores: dict[str, float]) -> dict[str, float]:
        avg_score = sum([score for miner_uid, score in scores.items()]) / len(scores)
        scaled_scores_map = {}
        for miner_uid, score in scores.items():
            # handle case of a perfect score
            if score == 0:
                score = 0.00000001
            scaled_scores_map[miner_uid] = 1 - math.e ** (-1 / (score / avg_score))
        return scaled_scores_map

    @staticmethod
    def weigh_miner_scores(scores: list[tuple[str, float]]) -> list[tuple[str, float]]:
        if len(scores) == 1:
            return [(scores[0][0], 1.0)]

        min_score = min(score for _, score in scores)
        max_score = max(score for _, score in scores)

        normalized_scores = [
            (name, (score - min_score) / (max_score - min_score))
            for name, score in scores
        ]
        total_normalized_score = sum(score for _, score in normalized_scores)

        normalized_scores = [
            (name, round(score / total_normalized_score, 4))
            for name, score in normalized_scores
        ]

        return normalized_scores

    @staticmethod
    def simple_scale_scores(scores: dict[str, float]) -> dict[str, float]:
        if len(scores) <= 1:
            raise MinResponsesException("not enough responses")
        score_values = [score for miner_uid, score in scores.items()]
        min_score = min(score_values)
        max_score = max(score_values)

        return {
            miner_uid: 1 - ((score - min_score) / (max_score - min_score))
            for miner_uid, score in scores.items()
        }

    @staticmethod
    def history_of_values() -> None | dict[str, float]:
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
    def get_geometric_mean_of_percentile(ds: ndarray):
        min_max_ranges_percentiled = ValiConfig.MIN_MAX_RANGES_PERCENTILED
        std_dev_ranges_percentiled = ValiConfig.STD_DEV_RANGES_PERCENTILED

        results_min_max = max(ds) / min(ds)
        results_std_dev = np.std(ds)

        min_max_percentile = Scoring.get_percentile(
            results_min_max, min_max_ranges_percentiled
        )
        std_dev_percentile = Scoring.get_percentile(
            results_std_dev, std_dev_ranges_percentiled
        )

        return math.sqrt(min_max_percentile * std_dev_percentile)

    @staticmethod
    def update_weights_using_historical_distributions(
        scores: list[tuple[str, float]], ds: ndarray, stream_id: str
    ):
        vweights = ValiUtils.get_vali_weights_json(stream_id)
        geometric_mean_of_percentile = Scoring.get_geometric_mean_of_percentile(ds)

        score_miner_uids = [score[0] for score in scores]

        if len(vweights) != 0:
            vweight_avg = sum(vweights.values()) / len(vweights)
        else:
            vweight_avg = 0

        for key, value in vweights.items():
            if key not in score_miner_uids:
                vweights[key] = Scoring.basic_ema(
                    (vweights[key] + (0 * geometric_mean_of_percentile))
                    / (1 + geometric_mean_of_percentile),
                    vweights[key],
                )

        for score in scores:
            if score[0] in vweights:
                previous_ema = vweights[score[0]]
            else:
                previous_ema = vweight_avg
            vweights[score[0]] = Scoring.basic_ema(
                (previous_ema + (score[1] * geometric_mean_of_percentile))
                / (1 + geometric_mean_of_percentile),
                previous_ema,
            )

        for k, v in vweights.items():
            if math.isnan(v):
                print(f"bad data provided.")
                print(f"geometric mean of percentile [{geometric_mean_of_percentile}]")
                print(f"scores [{scores}]")
                raise ValueError("nan set in vweights file")

        ValiUtils.set_vali_weights_bkp(vweights)
        return vweights, geometric_mean_of_percentile

    @staticmethod
    def update_weights_remove_deregistrations(miner_uids: list[str]):
        for vali_stream in ValiStream:
            stream_id = vali_stream.stream_id
            vweights = ValiUtils.get_vali_weights_json(stream_id)
            for miner_uid in miner_uids:
                if miner_uid in vweights:
                    del vweights[miner_uid]
            ValiUtils.set_vali_weights_bkp(vweights, stream_id)

    @staticmethod
    def basic_ema(current_value, previous_ema, length=48):
        alpha = 2 / (length + 1)
        ema = alpha * current_value + (1 - alpha) * previous_ema
        return ema
