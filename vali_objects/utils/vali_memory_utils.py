import json
import os


class ValiMemoryUtils:

    @staticmethod
    def get_vali_memory():
        return os.getenv("vm")

    @staticmethod
    def set_vali_memory(vm):
        os.environ["vm"] = vm
