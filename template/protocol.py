# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Taoshidev
# Copyright © 2023 Taoshi, LLC

import typing
import bittensor as bt
from pydantic import Field

from typing import List


class BaseProtocol(bt.Synapse):
    request_uuid: str = Field(..., allow_mutation=False)
    stream_id: str = Field(..., allow_mutation=False)
    samples: bt.Tensor
    topic_id: typing.Optional[int] = Field(..., allow_mutation=False)


class Forward(BaseProtocol):
    feature_ids: List[float]
    prediction_size: int = Field(..., allow_mutation=False)
    schema_id: typing.Optional[int] = Field(..., allow_mutation=False)
    predictions: bt.Tensor = None

    def deserialize(self) -> bt.Tensor:
        return self.predictions


class Backward(BaseProtocol):
    received: bool = None

    def deserialize(self) -> bool:
        return self.received


class TrainingForward(Forward):
    pass


class LiveForward(Forward):
    pass


class TrainingBackward(Backward):
    pass


class LiveBackward(Backward):
    pass


# class LiveForward(bt.Synapse):
#     requestuuid: str = Field(..., allow_mutation=False)
#     stream_id: str = Field(..., allow_mutation=False)
#     # samples: bt.Tensor = Field(..., allow_mutation=False)
#     # topic_id: typing.Optional[int] = Field(..., allow_mutation=False)
#     # feature_ids: List[float] = Field(..., allow_mutation=False)
#     predictionsize: int = Field(..., allow_mutation=False)
#     # schema_id: typing.Optional[int] = Field(..., allow_mutation=False)
#     predictions: bt.Tensor = None
#
#     requestuuid_hash: str = None
#     stream_id_hash: str = None
#     # samples_hash: str = None
#     # topic_id_hash: str = None
#     # feature_ids_hash: str = None
#     predictionsize_hash: str = None
#     # schema_id_hash: str = None
#
#     def deserialize(self) -> bt.Tensor:
#         return self.predictions
#


