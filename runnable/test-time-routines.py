from miner_config import MinerConfig
from time_util import datetime
from time_util.time_util import TimeUtil

now = datetime.now()
print(now)
print(now.timestamp() * 1000)
print()

ts_ranges = TimeUtil.convert_range_timestamps_to_millis(
    TimeUtil.generate_range_timestamps(
        TimeUtil.generate_start_timestamp(MinerConfig.STD_LOOKBACK),
        MinerConfig.STD_LOOKBACK,
    )
)

for ts_range in ts_ranges:
    print(ts_range)
    print(
        f"{datetime.fromtimestamp_ms(ts_range[0])} {datetime.fromtimestamp_ms(ts_range[1])}"
    )
    print()
