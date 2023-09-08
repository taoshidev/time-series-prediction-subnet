# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Taoshi
# Copyright © 2023 TARVIS Labs, LLC

import os


class ValiMemoryUtils:

    @staticmethod
    def get_vali_memory():
        return os.getenv("vm")

    @staticmethod
    def set_vali_memory(vm):
        os.environ["vm"] = vm
