# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Taoshi
# Copyright © 2023 TARVIS Labs, LLC

import typing
import bittensor as bt
from pydantic import Field

from typing import List


class BaseProtocol(bt.Synapse):
    request_uuid: str = Field(..., allow_mutation=False)
    stream_type: int = Field(..., allow_mutation=False)
    feature_ids: List[float] = Field(default_factory=list)
    samples: bt.Tensor
    topic_id: typing.Optional[int] = Field(..., allow_mutation=False)
    schema_id: typing.Optional[int] = Field(..., allow_mutation=False)


class Forward(BaseProtocol):
    prediction_size: int = Field(32, allow_mutation=False)
    predictions: bt.Tensor = None


class Backward(BaseProtocol):
    pass
