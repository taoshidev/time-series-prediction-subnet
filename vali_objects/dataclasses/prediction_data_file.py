# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Taoshi
# Copyright © 2023 Taoshi, LLC

from dataclasses import dataclass

import numpy as np

from vali_objects.dataclasses.base_objects.base_dataclass import BaseDataClass


@dataclass
class PredictionDataFile(BaseDataClass):
    client_uuid: str
    stream_type: str
    stream_id: int
    topic_id: int
    request_uuid: str
    miner_uid: str
    start: int
    end: int
    vmins: list[float]
    vmaxs: list[float]
    decimal_places: list[int]
    predictions: np

    def __eq__(self, other):
        return self.equal_base_class_check(other)