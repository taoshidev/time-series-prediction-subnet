# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

from dataclasses import dataclass
from typing import Optional

from vali_objects.dataclasses.base_objects.base_request_dataclass import BaseRequestDataClass


@dataclass
class NewRequestDataClass(BaseRequestDataClass):
    stream_ids: list[str]
    topic_ids: list[int]
    schema_ids: list[int]
    feature_ids: list[float]
    prediction_size: int
    additional_details: list[dict]
    client_uuid: Optional[str] = None
