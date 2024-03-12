# developer: Taoshidev
# Copyright © 2023 Taoshi Inc
import math
import numpy as np
from numpy import ndarray
from scipy.stats import norm

from vali_config import ValiConfig
from vali_objects.exceptions.incorrect_prediction_size_error import (
    IncorrectPredictionSizeError,
)
from vali_objects.exceptions.min_responses_exception import MinResponsesException
from vali_objects.utils.vali_utils import ValiUtils


class Scoring:
    @staticmethod
    def weigh_miner_scores(scores: list[tuple[str, float]]) -> list[tuple[str, float]]:
        ## Assign weights to the scores based on their relative position
        if len(scores) == 0:
            return [ scores[0][0], 1.0 ]

        n_miners = len(scores)

        miner_names = [ x[0] for x in scores ]
        miner_scores = [ x[1] for x in scores ]
        score_arrangement = np.argsort(miner_scores)

        top_miner_benefit = ValiConfig.TOP_MINER_BENEFIT
        top_miner_percent = ValiConfig.TOP_MINER_PERCENT

        def exponential_decay_scores(
                scale: int, 
                a_percent: float = 0.8,
                b_percent: float = 0.2
                ) -> np.ndarray:
            """
            Args:
                scale: int - the number of miners
                a_percent: float - % benefit to the top % miners
                b_percent: float - top % of miners
            """
            a_percent = np.clip(a_percent, a_min=0, a_max=1)
            b_percent = np.clip(b_percent, a_min=0.00000001, a_max=1)
            scale = np.clip(scale, a_min=1, a_max=None)

            if scale == 1:
                # base case, if there is only one miner
                return np.array([1])
            
            k = -np.log(1 - a_percent) / (b_percent)
            xdecay = np.linspace(0, scale-1, scale)
            decayed_scores = np.exp((-k / scale) * xdecay)
            
            # Normalize the decayed_scores so that they sum up to 1
            return decayed_scores / np.sum(decayed_scores)
        
        decayed_scores = exponential_decay_scores(
            n_miners, 
            top_miner_benefit, 
            top_miner_percent
        )
        miner_decay_scores = decayed_scores[score_arrangement]

        return list(zip(miner_names, miner_decay_scores))
    
    @staticmethod
    def update_weights_using_historical_distributions(
        prior_weights: dict[str, float],
        weighted_scores: list[tuple[str, float]], 
        validation_array: ndarray
    ) -> dict[str, float]:
        """
        Args:
            prior_weights: dict[str, float] - the historical weights
            weighted_scores: list[tuple[str, float]] - the scores for the current round
            validation_array: ndarray - the validation array
        """
        # this parameter will define how much weight we want to put on the new data compared to the eda
        difficulty = Scoring.difficulty(validation_array)
        difficulty_modifier = Scoring.difficulty_transform(difficulty)

        # set the proportion of value to current and historical
        ema_history_weight = 1 - difficulty_modifier
        current_weight = difficulty_modifier

        # (ema * ema_history_weight) + (current * current_weight) -> this is the formula for the weighted average
        score_miner_uids = [ score[0] for score in weighted_scores ]

        if len(prior_weights) != 0:
            vweight_avg = sum(prior_weights.values()) / len(prior_weights)
        else:
            vweight_avg = 0

        # if we have a history of activity but nothing on the current round
        for key in prior_weights.keys():
            if key not in score_miner_uids:
                prior_weights[key] = prior_weights[key] * ema_history_weight

        # iterate through a history of activity and update the weights
        for score in weighted_scores:
            previous_ema = prior_weights.get(score[0], vweight_avg)
            current_score = score[1]

            prior_weights[score[0]] = ema_history_weight * previous_ema + current_weight * current_score

        # normalize the weights to sum to 1
        total_weight = sum(prior_weights.values())
        for k, v in prior_weights.items():
            prior_weights[k] = v / total_weight

        return prior_weights, difficulty_modifier
    
    @staticmethod
    def difficulty_transform(difficulty: float):
        """
        Run difficulty through a sigmoid, for influence on the EMA.
        Args:
            difficulty: float - the difficulty score
            alpha: float - the alpha parameter for the sigmoid. Intensity of the spike - how fast do we shift from historical EMA to current value.
            beta: float - the beta parameter for the sigmoid - what is the middle point at which we consider with 50/50 weight.
        """
        # recommended values:
        # beta = 0.01 (estimated min 1e-4, max 0.2)
        # alpha = 450 (estimated min 15, max 600)
        alpha = ValiConfig.DIFFICULTY_EMA_INTENSITY
        beta = ValiConfig.DIFFICULTY_TYPICAL

        return (1 / (1 + math.exp(-alpha * (difficulty - beta))))

    @staticmethod
    def generate_weighting_array(num_predictions: int) -> np.ndarray:
        # these are the parameters for the std dev function
        a = ValiConfig.WEIGHTED_SCALING
        b = ValiConfig.WEIGHTED_XDISPLACEMENT
        c = ValiConfig.WEIGHTED_YDISPLACEMENT
        k = ValiConfig.WEIGHTED_EXPONENT

        # might want to also track the historical mean here

        x = np.arange(1, num_predictions) # drop zero, as it has no value
        weights = a * (x+b)**(-k) + c # this is more in line with historical percentage changes
        weights = np.clip(weights, a_min=0.00000001, a_max=None)
        return weights

    @staticmethod
    def calculate_weighted_rmse(predictions, actual) -> float:
        predictions = np.array(predictions)
        actual = np.array(actual)

        weights = Scoring.generate_weighting_array(len(predictions))

        # mathematically, we might not need this. But this is included to aid in our interpretation of results.
        normalized_weights = weights / np.max(weights) # max weight here is 1
        # weights_norm = weights / np.max(weights) # max weight here is 1

        return float(sum(normalized_weights * ((predictions[1:] - actual[1:]) ** 2)))

    @staticmethod
    def score_response(predictions, actual) -> float:
        if len(predictions) != len(actual) or len(predictions) == 0 or len(actual) < 2:
            raise IncorrectPredictionSizeError(
                f"the number of predictions or the number of responses "
                f"needed are incorrect: preds: '{len(predictions)}',"
                f" results: '{len(actual)}'"
            )

        return Scoring.calculate_weighted_rmse(predictions, actual)

    @staticmethod
    def get_z_scores(value, std_dev, means) -> ndarray:
        return (value - means) / std_dev

    @staticmethod
    def difficulty(validation_array: ndarray) -> float:
        """Compute the difficulty of the prediction interval based on prior history."""
        validation_start = max(validation_array[0], 0.00000001)

        if len(validation_array) < 2:
            ## this is the default value for typical difficulty
            return np.array([ValiConfig.DIFFICULTY_TYPICAL])
        
        validation_percent_changes = ( validation_array - validation_start ) / validation_start
        
        ## taking from the first element because percentage change from 0 won't be useful
        validation_percent_changes = validation_percent_changes[1:]

        # get the std dev scores
        weighting_array = Scoring.generate_weighting_array(len(validation_array))
        weights = np.clip(weighting_array, a_min=0.00000001, a_max=None)
        # we might want to raise an error here if the std dev is 0, something is wrong

        stddevs = (1 / weights)
        means = np.zeros(len(validation_percent_changes))
        zscores = Scoring.get_z_scores(validation_percent_changes, stddevs, means)

        if len(zscores) == 0:
            return 0
        
        if len(zscores) == 1:
            return abs(zscores[0])
        
        mean_score = np.mean(zscores)

        # two tailed distribution percentile - difficulty is (1 - percentile)
        difficulty_score = 1 - (2 * (1 - norm.cdf(abs(mean_score))))
        return difficulty_score

    @staticmethod
    def update_weights_remove_deregistrations(
        prior_weights: dict[str, float],
        miner_uids: list[str]):

        for miner_uid in miner_uids:
            if miner_uid in prior_weights:
                del prior_weights[miner_uid]

        return prior_weights
