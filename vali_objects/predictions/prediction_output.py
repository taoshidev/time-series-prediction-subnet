from dataclasses import dataclass

import numpy as np

from vali_objects.base_objects.base_dataclass import BaseDataClass


@dataclass
class PredictionOutput(BaseDataClass):
    client_uuid: str
    stream_type: int
    request_uuid: str
    miner_uid: str
    start: int
    end: int
    avgs: list[float]
    decimal_places: list[int]
    predictions: np
