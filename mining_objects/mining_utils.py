import pickle

from vali_config import ValiConfig
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils


class MiningUtils:

    @staticmethod
    def write_file(dir: str, file_name: str, vali_data: dict | object, is_pickle: bool = False) -> None:
        # will concat dir and file name
        ValiBkpUtils.make_dir(ValiConfig.BASE_DIR + dir)
        ValiBkpUtils.write_to_vali_dir(ValiConfig.BASE_DIR + dir + file_name, vali_data, is_pickle)

    @staticmethod
    def get_file(file, is_pickle: bool = False) -> dict | object:
        with open(ValiConfig.BASE_DIR + file, ValiBkpUtils.get_read_type(is_pickle)) as f:
            return pickle.load(f) if is_pickle else f.read()