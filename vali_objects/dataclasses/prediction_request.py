# developer: Taoshidev
# Copyright © 2023 Taoshi, LLC

from dataclasses import dataclass
from typing import Optional

import numpy as np

from vali_objects.dataclasses.base_objects.base_request_dataclass import BaseRequestDataClass
from vali_objects.dataclasses.prediction_data_file import PredictionDataFile


@dataclass
class PredictionRequest(BaseRequestDataClass):
    request_uuid: str
    df: PredictionDataFile
    files: list[str]
    predictions: dict[str, np]
    samples: Optional[list[float]] = None

    def __eq__(self, other):
        return self.equal_base_class_check(other)

    def schema_integrity_check(self):
        pass
