from vali_objects.cmw.cmw_objects.cmw_stream_type import CMWStreamType


class CMWClient:
    def __init__(self):
        self.client_uuid = None
        self.streams = []

    def set_client_uuid(self, client_uuid):
        self.client_uuid = client_uuid
        return self

    def add_stream(self, cmw_stream_type: CMWStreamType):
        self.streams.append(cmw_stream_type)

    def get_stream(self, stream_type: int):
        for stream in self.streams:
            if stream.stream_type == stream_type:
                return stream
        return None

