# developer: taoshi-mbrown
# Copyright © 2023 Taoshi, LLC
from features import FeatureCollector, FeatureSource
from feature_sources import BinaryFileFeatureStorage
from streams.btcusd_5m import (
    historical_sources,
    historical_feature_ids,
    INTERVAL_MS,
)
from time_util import datetime, time_span_ms, previous_interval_ms
from vali_config import ValiConfig

SAMPLE_COUNT_MAX = 1000


def generate_historical_data(
    feature_source: FeatureSource,
    start_time_ms: int,
    end_time_ms: int,
    filename: str,
) -> None:
    print("Creating historical data file...")
    data_filename = (
        ValiConfig.BASE_DIR + "/runnable/historical_financial_data/" + filename
    )
    historical_feature_storage = BinaryFileFeatureStorage(
        filename=data_filename,
        mode="w",
        feature_ids=historical_feature_ids,
    )

    while start_time_ms < end_time_ms:
        sample_count = int((end_time_ms - start_time_ms) / INTERVAL_MS)
        sample_count = min(sample_count, SAMPLE_COUNT_MAX)

        start_time_datetime = datetime.fromtimestamp_ms(start_time_ms)
        print(f"Requesting historical data for {start_time_datetime}...")

        samples = feature_source.get_feature_samples(
            start_time_ms, INTERVAL_MS, sample_count
        )

        print("Storing...")

        historical_feature_storage.set_feature_samples(
            start_time_ms, INTERVAL_MS, samples
        )

        start_time_ms += INTERVAL_MS * sample_count

    historical_feature_storage.close()


def main() -> None:
    _TRAINING_LOOKBACK_DAYS = 400
    _TESTING_LOOKBACK_DAYS = 30

    now = datetime.now()
    now_time_ms = now.timestamp_ms()
    now_time_ms = previous_interval_ms(now_time_ms, INTERVAL_MS)

    training_start_time_ms = now_time_ms - time_span_ms(days=_TRAINING_LOOKBACK_DAYS)
    testing_start_time_ms = now_time_ms + time_span_ms(days=_TESTING_LOOKBACK_DAYS)
    training_end_time_ms = testing_start_time_ms

    historical_feature_collector = FeatureCollector(
        sources=historical_sources,
        feature_ids=historical_feature_ids,
        cache_results=False,
    )

    generate_historical_data(
        historical_feature_collector,
        training_start_time_ms,
        training_end_time_ms,
        "data_training.taosfs",
    )

    print("Done.")


if __name__ == "__main__":
    main()
