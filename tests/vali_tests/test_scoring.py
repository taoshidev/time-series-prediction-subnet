# developer: Taoshidev
# Copyright © 2023 Taoshi Inc
import math
import numpy as np
import os
from tests.vali_tests.samples.testing_data import TestingData
from tests.vali_tests.base_objects.test_base import TestBase
import unittest
from vali_config import ValiConfig
from vali_objects.exceptions.incorrect_prediction_size_error import (
    IncorrectPredictionSizeError,
)
from vali_objects.exceptions.min_responses_exception import MinResponsesException
from vali_objects.scoring.scoring import Scoring
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils


class TestScoring(TestBase):
    def test_calculate_weighted_rmse(self):
        predictions = []
        actual = []

        for x in range(0, 100):
            predictions.append(x)
            actual.append(x - 0.05)

        weighted_rmse = Scoring.calculate_weighted_rmse(predictions, actual)
        self.assertEqual(weighted_rmse, 0.04999999999999829)

    def test_score_response_rmse(self):
        predictions = []
        actual = []

        for x in range(0, 100):
            predictions.append(x)
            actual.append(x - 0.05)

        weighted_rmse = Scoring.score_response(predictions, actual)
        self.assertEqual(weighted_rmse, 0.04999999999999829)

    def test_score_response_rmse_exs(self):
        predictions1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]
        actual1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        weighted_rmse1 = Scoring.score_response(predictions1, actual1)

        predictions2 = [2, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        actual2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        weighted_rmse2 = Scoring.score_response(predictions2, actual2)

        predictions3 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11]
        actual3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        weighted_rmse3 = Scoring.score_response(predictions3, actual3)

        predictions4 = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
        actual4 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        weighted_rmse4 = Scoring.score_response(predictions4, actual4)

        self.assertTrue(weighted_rmse3 > weighted_rmse4)

    def test_scale_scores(self):
        scaled_scores = Scoring.scale_scores(TestingData.SCORES)
        self.assertEqual(scaled_scores, TestingData.SCALED_SCORES)

    def test_simple_scale_scores(self):
        try:
            sample_scores = {
                "5F7GYJfDNRccc2ZTFXqjWVEQ96Vjv2yEsa1wzr3ULijD8Qhd": 34.54618500386289
            }
            scaled_scores = Scoring.simple_scale_scores(sample_scores)
        except Exception as e:
            self.assertIsInstance(e, MinResponsesException)
        sample_scores = {
            "5F7GYJfDNRccc2ZTFXqjWVEQ96Vjv2yEsa1wzr3ULijD8Qhd": 33.54618500386289,
            "5F7GYJfDNRccc2ZTFXqjWVEQ96Vjv2yEsa1wzr3ULijD8Qhe": 34.54618500386289,
            "5F7GYJfDNRccc2ZTFXqjWVEQ96Vjv2yEsa1wzr3ULijD8Qhf": 35.54618500386289,
            "5F7GYJfDNRccc2ZTFXqjWVEQ96Vjv2yEsa1wzr3ULijD8Qhg": 36.54618500386289,
        }
        scaled_scores_results = {
            "5F7GYJfDNRccc2ZTFXqjWVEQ96Vjv2yEsa1wzr3ULijD8Qhd": 1.0,
            "5F7GYJfDNRccc2ZTFXqjWVEQ96Vjv2yEsa1wzr3ULijD8Qhe": 0.6666666666666667,
            "5F7GYJfDNRccc2ZTFXqjWVEQ96Vjv2yEsa1wzr3ULijD8Qhf": 0.33333333333333337,
            "5F7GYJfDNRccc2ZTFXqjWVEQ96Vjv2yEsa1wzr3ULijD8Qhg": 0.0,
        }

        scaled_scores = Scoring.simple_scale_scores(sample_scores)
        self.assertEqual(scaled_scores_results, scaled_scores)

    def test_get_percentile(self):
        self.assertEqual(
            0.01, Scoring.get_percentile(0, ValiConfig.MIN_MAX_RANGES_PERCENTILED)
        )
        self.assertEqual(
            0.02, Scoring.get_percentile(1.004, ValiConfig.MIN_MAX_RANGES_PERCENTILED)
        )
        self.assertEqual(
            0.14, Scoring.get_percentile(1.01, ValiConfig.MIN_MAX_RANGES_PERCENTILED)
        )
        self.assertEqual(
            1, Scoring.get_percentile(1.05, ValiConfig.MIN_MAX_RANGES_PERCENTILED)
        )

        self.assertEqual(
            0.01, Scoring.get_percentile(0, ValiConfig.STD_DEV_RANGES_PERCENTILED)
        )
        self.assertEqual(
            0.11, Scoring.get_percentile(45, ValiConfig.STD_DEV_RANGES_PERCENTILED)
        )
        self.assertEqual(
            0.29, Scoring.get_percentile(100, ValiConfig.STD_DEV_RANGES_PERCENTILED)
        )
        self.assertEqual(
            1, Scoring.get_percentile(1000, ValiConfig.STD_DEV_RANGES_PERCENTILED)
        )

    def test_get_geometric_mean_of_percentile(self):
        ds = [[], [30000, 30100, 30150, 30200]]

        mean = np.mean(ds[1])
        squared_diff = np.sum((ds[1] - mean) ** 2)
        std_dev = np.sqrt(squared_diff / len(ds[1]))

        min_max = 1.006666666666667

        min_max_percentage = Scoring.get_percentile(
            min_max, ValiConfig.MIN_MAX_RANGES_PERCENTILED
        )
        std_dev_percentage = Scoring.get_percentile(
            std_dev, ValiConfig.STD_DEV_RANGES_PERCENTILED
        )

        geometric_mean_of_percentiles = math.sqrt(
            min_max_percentage * std_dev_percentage
        )

        self.assertEqual(
            geometric_mean_of_percentiles, Scoring.get_geometric_mean_of_percentile(ds)
        )

    def test_update_weights_using_historical_distributions(self):
        try:
            os.remove(
                ValiBkpUtils.get_vali_weights_dir()
                + ValiBkpUtils.get_vali_weights_file()
            )
        except:
            pass

        scores = [("miner1", 0.1), ("miner2", 0.2), ("miner3", 0.3), ("miner4", 0.4)]
        ds = [[], [30000, 30100, 30150, 30200]]

        (
            vweights,
            geometric_mean_of_percentile,
        ) = Scoring.update_weights_using_historical_distributions(scores, ds)

        gmop = 0.11224972160321824

        self.assertEqual(gmop, geometric_mean_of_percentile)

        self.assertEqual(
            {
                "miner1": 0.0004119237974115415,
                "miner2": 0.000823847594823083,
                "miner3": 0.0012357713922346242,
                "miner4": 0.001647695189646166,
            },
            vweights,
        )

        set_vweights = ValiUtils.get_vali_weights_json()
        self.assertEqual(vweights, set_vweights)

        scores.pop()
        ds = [[], [30000, 30100, 30150]]

        # testing if we remove a miner their score begins to drop

        (
            vweights,
            geometric_mean_of_percentile,
        ) = Scoring.update_weights_using_historical_distributions(scores, ds)

        self.assertEqual(
            {
                "miner1": 0.0006828611697138493,
                "miner2": 0.0013657223394276986,
                "miner3": 0.002048583509141548,
                "miner4": 0.0016432125023400094,
            },
            vweights,
        )

        # testing adding a new miner that they fit to the average even if the scores
        # have a larger magnitude move

        ds = [[], [30000, 3100, 32000]]
        scores.append(("miner5", 0.5))

        (
            vweights,
            geometric_mean_of_percentile,
        ) = Scoring.update_weights_using_historical_distributions(scores, ds)

        self.assertEqual(
            {
                "miner1": 0.002709741554005403,
                "miner2": 0.005419483108010806,
                "miner3": 0.00812922466201621,
                "miner4": 0.0016096775533126623,
                "miner5": 0.011609888862193414,
            },
            vweights,
        )

        os.remove(
            ValiBkpUtils.get_vali_weights_dir() + ValiBkpUtils.get_vali_weights_file()
        )

    def test_update_weights_remove_deregistrations(self):
        try:
            os.remove(
                ValiBkpUtils.get_vali_weights_dir()
                + ValiBkpUtils.get_vali_weights_file()
            )
        except:
            pass

        scores = [("miner1", 0.1), ("miner2", 0.2), ("miner3", 0.3), ("miner4", 0.4)]
        ds = [[], [30000, 30100, 30150, 30200]]

        (
            vweights,
            geometric_mean_of_percentile,
        ) = Scoring.update_weights_using_historical_distributions(scores, ds)

        gmop = 0.11224972160321824

        self.assertEqual(gmop, geometric_mean_of_percentile)

        deregistered_mineruids = []
        deregistered_mineruids.append("miner4")

        Scoring.update_weights_remove_deregistrations(deregistered_mineruids)

        set_vweights = ValiUtils.get_vali_weights_json()
        del vweights["miner4"]

        self.assertEqual(set_vweights, vweights)

        os.remove(
            ValiBkpUtils.get_vali_weights_dir() + ValiBkpUtils.get_vali_weights_file()
        )

    def test_update_weights_using_historical_distributions_with_dummy_data(self):
        scores = [("miner1", 0.1), ("miner2", 0.2), ("miner3", 0.3), ("miner4", 0.4)]
        data = [[], [10, 20, 30, 40]]
        updated_scores = Scoring.update_weights_using_historical_distributions(
            scores, data
        )

    def test_calculate_directional_accuracy(self):
        predictions = [1, 2, 1, 3, 4, 5, 6, 7, 6, 7, 8]
        actual = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        da = Scoring.calculate_directional_accuracy(predictions, actual)
        self.assertEqual(da, 0.8)

    def test_score_response(self):
        def assert_preds_size_error(a, p):
            try:
                Scoring.score_response(a, p)
            except Exception as e:
                self.assertIsInstance(e, IncorrectPredictionSizeError)

        actual = []
        predictions = [1, 2]

        assert_preds_size_error(actual, predictions)

        actual = [1, 2]
        predictions = []

        assert_preds_size_error(actual, predictions)

        actual = []
        predictions = []

        assert_preds_size_error(actual, predictions)

        actual = [1]
        predictions = [1, 2]

        assert_preds_size_error(actual, predictions)

        actual = [1, 2]
        predictions = [1]

        assert_preds_size_error(actual, predictions)

        predictions = []
        actual = []

        for x in range(0, 100):
            predictions.append(x)
            actual.append(x - 0.05)

        score = Scoring.score_response(predictions, actual)
        self.assertEqual(score, 0.04999999999999829)


if __name__ == "__main__":
    unittest.main()
