# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Taoshi
# Copyright © 2023 TARVIS Labs, LLC

import json

from vali_objects.cmw.cmw_objects.cmw import CMW
from vali_objects.cmw.cmw_objects.cmw_client import CMWClient
from vali_objects.cmw.cmw_objects.cmw_miner import CMWMiner
from vali_objects.cmw.cmw_objects.cmw_stream_type import CMWStreamType


class CMWUtil:

    @staticmethod
    def unload_cmw(vr):
        cmw = CMW()
        for client in vr["clients"]:
            cmw_client = CMWClient().set_client_uuid(client["client_uuid"])
            for stream in client["streams"]:
                cmw_stream = CMWStreamType().set_stream_type(stream["stream_type"]).set_topic_id(stream["topic_id"])
                for miner in stream["miners"]:
                    cmw_stream.add_miner(CMWMiner(miner["miner_id"], miner["wins"], miner["o_wins"], miner["scores"]))
                cmw_client.add_stream(cmw_stream)
            cmw.add_client(cmw_client)
        return cmw

    @staticmethod
    def load_cmw(cmw: CMW):
        return json.loads(json.dumps(cmw, default=lambda o: o.__dict__))

    @staticmethod
    def initialize_cmw():
        return {
                "clients": {}
            }