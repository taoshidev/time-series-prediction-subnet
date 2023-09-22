# developer: Taoshidev
# Copyright © 2023 Taoshi, LLC

from dataclasses import dataclass
from vali_objects.dataclasses.base_objects.new_request_dataclass import NewRequestDataClass


@dataclass
class ClientRequest(NewRequestDataClass):
    pass

    def __eq__(self, other):
        return self.equal_base_class_check(other)