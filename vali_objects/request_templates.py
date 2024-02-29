from vali_objects.dataclasses.base_objects.new_request_dataclass import NewRequestDataClass


class RequestTemplates:

    def __init__(self):
        self.templates = [
            NewRequestDataClass(
                stream_ids="BTCUSD-5m",
                topic_ids=1,
                schema_ids=1,
                feature_ids=[1, 2, 3, 4, 5],
                prediction_size=100,
                additional_details={
                    "tf": 5,
                    "trade_pair": "BTCUSD"
                }
            )
        ]
