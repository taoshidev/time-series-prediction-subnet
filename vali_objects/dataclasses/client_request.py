# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

from dataclasses import dataclass
from typing import List, Optional

from vali_config import ValiStream
from vali_objects.dataclasses.base_objects.new_request_dataclass import NewRequestDataClass


@dataclass
class ClientRequest:
    vali_streams: List[ValiStream]
    client_uuid: Optional[str] = None

    # @classmethod
    # def init_client_request(cls, client_request_dict):
    #     return cls(**vars(client_request_dict))
