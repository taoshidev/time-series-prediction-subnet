# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Taoshidev
# Copyright © 2023 Taoshi Inc
import bittensor as bt
from pydantic import Field
from typing import List, Optional, Dict

from vali_config import ValiStream

# keeping old protos for backward compat
# will deprecate once all valis are upgraded to V7
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


# new stream based protos
class BaseProtocolStreams(bt.Synapse):
    request_uuid: str = Field(..., allow_mutation=False)


class ForwardStreams(BaseProtocolStreams):
    vali_streams: List[Dict]


class ForwardPredictionStreams(ForwardStreams):
    predictions: Optional[bt.Tensor] = None


class ForwardHashStreams(ForwardStreams):
    hashed_predictions: Optional[List[str]] = None


class LiveForwardHash(ForwardHash):
    pass


class LiveForwardStreams(ForwardPredictionStreams):
    pass


class LiveForwardHashStreams(ForwardHashStreams):
    pass


class TrainingBackward(Backward):
    pass


class LiveBackward(Backward):
    pass