# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Taoshi
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
    feature_ids: List[float] = Field(default_factory=list)
    schema_id: typing.Optional[int] = Field(..., allow_mutation=False)
    prediction_size: int = Field(32, allow_mutation=False)
    predictions: bt.Tensor = None


class Backward(BaseProtocol):
    pass
