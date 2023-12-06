import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from vali_config import ValiConfig
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils


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
    def scale_ds(data_structure, scale_min, scale_max):
        sds_ndarray = data_structure.T
        scaler = MinMaxScaler(feature_range=(scale_min, scale_max))

        scaled_data = scaler.fit_transform(sds_ndarray)
        scaled_data = scaled_data.T
        return scaled_data

    @staticmethod
    def scale_ds_0_100(data_structure):
        return MiningUtils.scale_ds(np.array(data_structure), 0, 100)

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