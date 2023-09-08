import json
import os
import pickle
from datetime import datetime

from vali_config import ValiConfig


class ValiBkpUtils:

    @staticmethod
    def get_vali_bkp_dir():
        return ValiConfig.BASE_DIR + '/validation/backups/'

    @staticmethod
    def get_vali_data_file():
        return 'valirecords.json'

    @staticmethod
    def get_vali_responses_dir():
        return ValiConfig.BASE_DIR + '/validation/predictions/'

    @staticmethod
    def get_response_filename(request_uuid: str):
        return str(request_uuid) + ".pickle"

    @staticmethod
    def make_dir(vali_dir: str):
        if not os.path.exists(vali_dir):
            os.makedirs(vali_dir)

    @staticmethod
    def get_write_type(is_pickle: bool):
        if is_pickle:
            write_type = 'wb'
        else:
            write_type = 'w'
        return write_type

    @staticmethod
    def get_read_type(is_pickle: bool):
        if is_pickle:
            write_type = 'rb'
        else:
            write_type = 'r'
        return write_type

    @staticmethod
    def write_to_vali_dir(vali_file: str, vali_data: dict | object, is_pickle: bool = False):
        with open(vali_file, ValiBkpUtils.get_write_type(is_pickle)) as f:
            if is_pickle:
                pickle.dump(vali_data, f)
            else:
                print(vali_data)
                f.write(json.dumps(vali_data))
        f.close()

    @staticmethod
    def write_vali_file(vali_dir: str, file_name: str, vali_data: dict | object, is_pickle: bool = False):
        # will concat dir and file name
        ValiBkpUtils.make_dir(vali_dir)
        ValiBkpUtils.write_to_vali_dir(vali_dir + file_name, vali_data, is_pickle)

    @staticmethod
    def get_vali_file(vali_file, is_pickle: bool = False):
        with open(vali_file, ValiBkpUtils.get_read_type(is_pickle)) as f:
            if is_pickle:
                return pickle.load(f)
            else:
                return f.read()

    @staticmethod
    def get_all_files_in_dir(vali_dir: str):
        all_files = []
        for filename in os.listdir(vali_dir):
            if os.path.isfile(os.path.join(vali_dir, filename)):
                all_files.append(vali_dir + filename)
        return all_files

    @staticmethod
    def delete_stale_files(vali_dir: str):
        current_date = datetime.now()
        for filename in os.listdir(vali_dir):
            file_path = os.path.join(vali_dir, filename)
            if os.path.isfile(file_path):
                creation_timestamp = os.path.getctime(file_path)
                creation_date = datetime.fromtimestamp(creation_timestamp)
                age_in_days = (current_date - creation_date).days
                if age_in_days > ValiConfig.DELETE_STALE_DATA:
                    os.remove(file_path)



