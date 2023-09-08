# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Taoshi
# Copyright © 2023 TARVIS Labs, LLC

import json
import traceback
from io import UnsupportedOperation
from json import JSONDecodeError

from vali_objects.cmw.cmw_objects.cmw import CMW
from vali_objects.cmw.cmw_util import CMWUtil
from vali_objects.exceptions.corrupt_data_exception import ValiBkpCorruptDataException
from vali_objects.exceptions.vali_bkp_file_missing_exception import ValiFileMissingException
from vali_objects.exceptions.vali_records_misalignment_exception import ValiRecordsMisalignmentException
from vali_objects.exceptions.vali_memory_missing_exception import ValiMemoryMissingException
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_memory_utils import ValiMemoryUtils


class ValiUtils:
    @staticmethod
    def get_vali_records() -> CMW:
        try:
            return ValiUtils.get_vali_memory_json()
        except ValiMemoryMissingException:
            ValiUtils.set_memory_with_bkp()
            return ValiUtils.get_vali_records()

    @staticmethod
    def load_json(data):
        # try:
        return json.loads(data)
        # except JSONDecodeError:
        #     raise ValiMemoryCorruptDataException("data is not json formatting")

    @staticmethod
    def get_vali_bkp_json():
        # wrapping here to allow simpler error handling & original for other error handling
        try:
            vbkp = ValiBkpUtils.get_vali_file(ValiBkpUtils.get_vali_bkp_dir()
                                              + ValiBkpUtils.get_vali_data_file())
        except FileNotFoundError:
            init_cmw = CMWUtil.initialize_cmw()
            ValiUtils.set_vali_bkp(init_cmw)
            return init_cmw
        else:
            return ValiUtils.load_json(vbkp)

    @staticmethod
    def get_vali_memory_json():
        # wrapping here to allow simpler error handling & original for other error handling
        try:
            vm = ValiMemoryUtils.get_vali_memory()
        except KeyError:
            raise ValiMemoryMissingException("loading vali memory failed")
        else:
            if vm is None:
                raise ValiMemoryMissingException("vm is none")
            return CMWUtil.unload_cmw(ValiUtils.load_json(vm))

    @staticmethod
    def check_memory_matches_bkp():
        vbkp = ValiUtils.get_vali_bkp_json()
        vm = ValiUtils.get_vali_memory_json()
        if vbkp != vm:
            return ValiRecordsMisalignmentException("misalignment of bkp and memory")

    @staticmethod
    def set_memory_with_bkp():
        print("setting from bkp")
        ValiMemoryUtils.set_vali_memory(json.dumps(ValiUtils.get_vali_bkp_json()))

    @staticmethod
    def get_vali_predictions(file):
        # wrapping here to allow simpler error handling & original for other error handling
        try:
            preds = ValiBkpUtils.get_vali_file(file, True)
        except FileNotFoundError:
            raise ValiFileMissingException("loading predictions file failed")
        try:
            return preds
        except UnsupportedOperation:
            raise ValiBkpCorruptDataException("prediction data is not pickled")

    @staticmethod
    def save_predictions_request(request_uuid: str, content: dict | object):
        ValiBkpUtils.write_vali_file(ValiBkpUtils.get_vali_responses_dir(),
                                     ValiBkpUtils.get_response_filename(request_uuid),
                                     content, True)

    @staticmethod
    def set_vali_memory_and_bkp(vali_records: dict):
        ValiMemoryUtils.set_vali_memory(json.dumps(vali_records))
        ValiUtils.set_vali_bkp(vali_records)

    @staticmethod
    def set_vali_bkp(vali_records: dict):
        ValiBkpUtils.write_vali_file(ValiBkpUtils.get_vali_bkp_dir(),
                                     ValiBkpUtils.get_vali_data_file(),
                                     vali_records)

