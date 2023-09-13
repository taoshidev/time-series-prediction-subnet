from dataclasses import dataclass

from vali_objects.dataclasses.base_objects.base_dataclass import BaseDataClass


@dataclass
class ClientRequest(BaseDataClass):
    client_uuid: str
    stream_type: int
    topic_id: int
    schema_id: int
    stream_type: int
    feature_ids: list[float]
    prediction_size: int

    def __eq__(self, other):
        return self.equal_base_class_check(other)