# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

from dataclasses import dataclass
from vali_objects.dataclasses.base_objects.new_request_dataclass import NewRequestDataClass


@dataclass
class ClientRequest(NewRequestDataClass):
    pass

    @classmethod
    def init_client_request(cls, new_request_data_class_obj):
        return cls(**vars(new_request_data_class_obj))

    def __eq__(self, other):
        return self.equal_base_class_check(other)