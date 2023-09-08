from datetime import datetime, timedelta


class TimeUtil:

    @staticmethod
    def generate_range_timestamps(start_date: datetime, end_date_days: int) -> list[tuple[datetime, datetime]]:
        end_date = start_date + timedelta(days=end_date_days)

        timestamps = []

        current_date = start_date
        while current_date <= end_date:
            start_timestamp = current_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_timestamp = current_date.replace(hour=23, minute=59, second=59, microsecond=999999)
            timestamps.append((start_timestamp, end_timestamp))
            current_date += timedelta(days=1)

        return timestamps

    @staticmethod
    def generate_start_timestamp(days: int):
        return datetime.now() - timedelta(days=days)

    @staticmethod
    def convert_range_timestamps_to_millis(timestamps: list[tuple[datetime, datetime]]):
        return [(int(row[0].timestamp() * 1000), int(row[1].timestamp() * 1000)) for row in timestamps]

    @staticmethod
    def now_in_millis():
        return int(datetime.now().timestamp() * 1000)

    @staticmethod
    def minute_in_millis(minutes: int):
        return minutes * 60000

    @staticmethod
    def convert_millis_to_timestamp(millis: int):
        return datetime.fromtimestamp(millis / 1000)