# developer: Taoshidev
# Copyright © 2023 Taoshi, LLC

from dataclasses import dataclass

import numpy as np

from vali_objects.dataclasses.base_objects.base_dataclass import BaseDataClass


@dataclass
class PredictionDataFile(BaseDataClass):
    client_uuid: str
    stream_type: str
    stream_id: str
    topic_id: int
    request_uuid: str
    miner_uid: str
    start: int
    end: int
    vmins: list[float]
    vmaxs: list[float]
    decimal_places: list[int]
    predictions: np
    prediction_size: int
    additional_details: dict

    def __eq__(self, other):
        return self.equal_base_class_check(other)