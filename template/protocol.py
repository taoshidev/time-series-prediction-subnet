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
    stream_id: int = Field(..., allow_mutation=False)
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


class LiveForwardTwo( bt.Synapse ):
    """
    A simple dummy protocol representation which uses bt.Synapse as its base.
    This protocol helps in handling dummy request and response communication between
    the miner and the validator.

    Attributes:
    - dummy_input: An integer value representing the input request sent by the validator.
    - dummy_output: An optional integer value which, when filled, represents the response from the miner.
    """

    # Required request input, filled by sending dendrite caller.
    # request_uuid: str = Field(..., allow_mutation=False)
    # stream_id: int = Field(..., allow_mutation=False)
    # samples: bt.Tensor
    # topic_id: typing.Optional[int] = Field(..., allow_mutation=False)
    # feature_ids: List[float] = Field(default_factory=list)
    # schema_id: typing.Optional[int] = Field(..., allow_mutation=False)
    # prediction_size: int = Field(32, allow_mutation=False)
    # predictions: bt.Tensor = None
    # Optional request output, filled by recieving axon.
    request_uuid: str = Field(..., allow_mutation=False)
    stream_id: int = Field(..., allow_mutation=False)
    samples: bt.Tensor
    topic_id: typing.Optional[int] = Field(..., allow_mutation=False)
    feature_ids: List[float]
    schema_id: typing.Optional[int] = Field(..., allow_mutation=False)
    prediction_size: int = Field(..., allow_mutation=False)
    predictions: bt.Tensor = None

    def deserialize(self) -> bt.Tensor:
        return self.predictions

