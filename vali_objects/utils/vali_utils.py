# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Taoshi
# Copyright © 2023 TARVIS Labs, LLC

import json
from pickle import UnpicklingError
from typing import Dict

from vali_objects.cmw.cmw_objects.cmw import CMW
from vali_objects.cmw.cmw_util import CMWUtil
from vali_objects.exceptions.corrupt_data_exception import ValiBkpCorruptDataException
from vali_objects.exceptions.vali_bkp_file_missing_exception import ValiFileMissingException
from vali_objects.exceptions.vali_records_misalignment_exception import ValiRecordsMisalignmentException
from vali_objects.exceptions.vali_memory_missing_exception import ValiMemoryMissingException
from vali_objects.dataclasses.prediction_output import PredictionOutput
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
    def get_vali_bkp_json() -> Dict:
        # wrapping here to allow simpler error handling & original for other error handling
        try:
            vbkp = ValiBkpUtils.get_vali_file(ValiBkpUtils.get_vali_bkp_dir()
                                              + ValiBkpUtils.get_vali_data_file())
        except FileNotFoundError:
            init_cmw = CMWUtil.initialize_cmw()
            ValiUtils.set_vali_bkp(init_cmw)
            return init_cmw
        else:
            return json.loads(vbkp)

    @staticmethod
    def get_vali_memory_json() -> CMW:
        # wrapping here to allow simpler error handling & original for other error handling
        try:
            vm = ValiMemoryUtils.get_vali_memory()
        except KeyError:
            raise ValiMemoryMissingException("loading vali memory failed")
        else:
            if vm is None:
                raise ValiMemoryMissingException("vm is none")
            return CMWUtil.load_cmw(json.loads(vm))

    @staticmethod
    def check_memory_matches_bkp() -> bool:
        vbkp = ValiUtils.get_vali_bkp_json()
        vm = CMWUtil.dump_cmw(ValiUtils.get_vali_memory_json())
        if vbkp != vm:
            raise ValiRecordsMisalignmentException("misalignment of bkp and memory")
        else:
            return True

    @staticmethod
    def set_memory_with_bkp() -> None:
        ValiMemoryUtils.set_vali_memory(json.dumps(ValiUtils.get_vali_bkp_json()))

    @staticmethod
    def get_vali_predictions(file) -> PredictionOutput:
        # wrapping here to allow simpler error handling & original for other error handling
        try:
            return ValiBkpUtils.get_vali_file(file, True)
        except FileNotFoundError:
            raise ValiFileMissingException("Vali predictions file is missing")
        except UnpicklingError:
            raise ValiBkpCorruptDataException("prediction data is not pickled")

    @staticmethod
    def save_predictions_request(request_uuid: str, content: Dict | object):
        ValiBkpUtils.write_vali_file(ValiBkpUtils.get_vali_predictions_dir(),
                                     ValiBkpUtils.get_response_filename(request_uuid),
                                     content, True)

    @staticmethod
    def set_vali_memory_and_bkp(vali_records: Dict):
        ValiMemoryUtils.set_vali_memory(json.dumps(vali_records))
        ValiUtils.set_vali_bkp(vali_records)

    @staticmethod
    def set_vali_bkp(vali_records: Dict):
        ValiBkpUtils.write_vali_file(ValiBkpUtils.get_vali_bkp_dir(),
                                     ValiBkpUtils.get_vali_data_file(),
                                     vali_records)

