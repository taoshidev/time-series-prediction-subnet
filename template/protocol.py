# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Taoshidev
# Copyright © 2023 Taoshi Inc
import bittensor as bt
from pydantic import Field
from typing import List, Optional


class BaseProtocol(bt.Synapse):
    request_uuid: str = Field(..., allow_mutation=False)
    stream_id: str = Field(..., allow_mutation=False)
    samples: Optional[bt.Tensor] = None
    topic_id: Optional[int] = Field(..., allow_mutation=False)


class Forward(BaseProtocol):
    feature_ids: List[float]
    prediction_size: int = Field(..., allow_mutation=False)
    schema_id: Optional[int] = Field(..., allow_mutation=False)


class ForwardPrediction(Forward):
    predictions: Optional[bt.Tensor] = None


class ForwardHash(Forward):
    hashed_predictions: Optional[str] = None


class Backward(BaseProtocol):
    received: bool = None


class TrainingForward(ForwardPrediction):
    pass


class LiveForward(ForwardPrediction):
    pass


class LiveForwardHash(ForwardHash):
    pass


class TrainingBackward(Backward):
    pass


class LiveBackward(Backward):
    pass