from dataclasses import dataclass
from typing import Optional

from vali_objects.dataclasses.base_objects.base_request_dataclass import BaseRequestDataClass


@dataclass
class NewRequestDataClass(BaseRequestDataClass):
    stream_type: str
    topic_id: int
    schema_id: int
    feature_ids: list[float]
    prediction_size: int
    client_uuid: Optional[str] = None
