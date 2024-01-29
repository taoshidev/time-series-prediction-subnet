# developer: Taoshidev
# Copyright © 2024 Taoshi, LLC
import datetime as native_datetime
import pendulum
from pendulum.tz.timezone import Timezone
from pendulum.tz import local_timezone
from pendulum.tz import UTC
import time as native_time
from typing import Optional, Union


def time_span_ms(
    weeks: int | float = 0,
    days: int | float = 0,
    hours: int | float = 0,
    minutes: int | float = 0,
    seconds: int | float = 0,
    milliseconds: int | float = 0,
) -> int:
    return int(
        (
            ((((((((weeks * 7) + days) * 24) + hours) * 60) + minutes) * 60) + seconds)
            * 1000
        )
        + milliseconds
    )


def parse_time_interval_ms(interval: str) -> float:
    reversed_terms = reversed(interval.split(":"))
    return sum(float(term) * 60**i for i, term in enumerate(reversed_terms))


def current_interval_ms(timestamp_ms: int, interval_ms: int) -> int:
    return interval_ms * int(timestamp_ms / interval_ms)


def next_interval_ms(timestamp_ms: int, interval_ms: int) -> int:
    return current_interval_ms(timestamp_ms + interval_ms, interval_ms)


def previous_interval_ms(timestamp_ms: int, interval_ms: int) -> int:
    return current_interval_ms(timestamp_ms - interval_ms, interval_ms)


def closest_interval_ms(timestamp_ms: int, interval_ms: int) -> int:
    return interval_ms * round(timestamp_ms / interval_ms)


# noinspection PyPep8Naming
class datetime(pendulum.DateTime):
    def __new__(
        cls,
        year,
        month=None,
        day=None,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
        tzinfo=UTC,
        *,
        fold=0
    ):
        return super().__new__(
            cls, year, month, day, hour, minute, second, microsecond, tzinfo, fold=fold
        )

    @classmethod
    def _clone(cls, dt) -> "datetime":
        return cls(
            dt.year,
            dt.month,
            dt.day,
            dt.hour,
            dt.minute,
            dt.second,
            dt.microsecond,
            tzinfo=dt.tzinfo,
            fold=dt.fold,
        )

    # The assumption should always be UTC, not local time
    @classmethod
    def fromtimestamp(cls, t, tz=UTC) -> "datetime":
        return cls._clone(native_datetime.datetime.fromtimestamp(t, tz=tz))

    @classmethod
    def fromtimestamp_ms(cls, ms, tz=UTC) -> "datetime":
        t = int(ms / 1000)
        us = (ms % 1000) * 1000
        result = cls.fromtimestamp(t, tz)
        result = result.replace(microsecond=us)
        return result

    # The assumption should always be UTC, not local time
    @classmethod
    def now(cls, tz: Optional[Union[str, Timezone]] = UTC) -> "datetime":
        """
        Returns an instance for the current date and time
        (in UTC by default, unlike Pendulum).
        """
        dt = cls.fromtimestamp(native_time.time(), UTC)

        if tz is None or tz == "local":
            dt = dt.in_timezone(local_timezone())
        elif tz is UTC or tz == "UTC":
            pass
        else:
            dt = dt.in_timezone(tz)

        return cls._clone(dt)

    @classmethod
    def parse(cls, text, **options) -> "datetime":
        dt = pendulum.parse(text, **options)
        return cls._clone(dt)

    def __str__(self):
        return self.to_iso8601_string()

    def timestamp_ms(self) -> int:
        t = self.timestamp()
        return (int(t) * 1000) + int(self.microsecond / 1000)

    def current_interval_ms(self, interval_ms: int) -> "datetime":
        return datetime.fromtimestamp_ms(
            current_interval_ms(self.timestamp_ms(), interval_ms), self.tzinfo
        )

    def next_interval_ms(self, interval_ms: int) -> "datetime":
        return datetime.fromtimestamp_ms(
            next_interval_ms(self.timestamp_ms(), interval_ms), self.tzinfo
        )

    def previous_interval_ms(self, interval_ms: int) -> "datetime":
        return datetime.fromtimestamp_ms(
            previous_interval_ms(self.timestamp_ms(), interval_ms), self.tzinfo
        )

    def closest_interval_ms(self, interval_ms: int) -> "datetime":
        return datetime.fromtimestamp_ms(
            closest_interval_ms(self.timestamp_ms(), interval_ms), self.tzinfo
        )


def sleep_until(wake_time: float) -> None:
    delay = wake_time - native_time.time()
    if delay > 0:
        native_time.sleep(delay)


def sleep_until_ms(wake_time_ms: int) -> None:
    sleep_until(wake_time_ms / 1000)


def sleep_until_datetime(wake_datetime: native_datetime) -> None:
    sleep_until(wake_datetime.float_timestamp())
