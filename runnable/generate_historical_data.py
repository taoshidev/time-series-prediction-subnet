# developer: taoshi-mbrown
# Copyright © 2023 Taoshi, LLC
from features import FeatureCollector
from feature_sources import BinaryFileFeatureStorage
from streams.btcusd_5m import (
    historical_sources,
    historical_feature_ids,
    INTERVAL_MS,
    model_feature_sources,
    model_feature_ids,
)
from time_util import closest_interval_ms, datetime, time_span_ms
from vali_config import ValiConfig


def main():
    # choose the range of days to look back
    # number of days back start
    _DAYS_BACK_START = 600
    # number of days forward since end day
    # for example start from 100 days ago and get 70 days from 100 days ago
    # (100 days ago, 99 days ago, 98 days ago, ..., up to 30 days ago)
    _DAYS = 599

    sample_count_max = int(time_span_ms(days=1) / INTERVAL_MS)

    now = datetime.now()
    now_time_ms = now.timestamp_ms()
    start_time_ms = now_time_ms - time_span_ms(days=_DAYS_BACK_START)
    end_time_ms = start_time_ms + time_span_ms(days=_DAYS)

    start_time_ms = closest_interval_ms(start_time_ms, INTERVAL_MS)
    end_time_ms = closest_interval_ms(end_time_ms, INTERVAL_MS)
    
    print(f'hist sources { historical_sources}')
    print(f'Running from model ids: { model_feature_ids}')

    historical_feature_collector = FeatureCollector(
        sources=model_feature_sources,
        feature_ids= model_feature_ids,
        cache_results=False,
    )

    print("Opening historical data...")

    data_filename = (
        ValiConfig.BASE_DIR + "/runnable/historical_financial_data/data2.taosfs"
    )
    historical_feature_storage = BinaryFileFeatureStorage(
        filename=data_filename,
        mode="w",
        feature_ids= model_feature_ids,
    )

    while start_time_ms < end_time_ms:
        sample_count = int((end_time_ms - start_time_ms) / INTERVAL_MS)
        sample_count = min(sample_count, sample_count_max)

        start_time_datetime = datetime.fromtimestamp_ms(start_time_ms)
        print(f"Requesting historical data for {start_time_datetime}...")

        samples = historical_feature_collector.get_feature_samples(
            start_time_ms, INTERVAL_MS, sample_count
        )

        print("Storing...")

        historical_feature_storage.set_feature_samples(
            start_time_ms, INTERVAL_MS, samples
        )

        start_time_ms += INTERVAL_MS * sample_count

    historical_feature_storage.close()

    print("Done.")


if __name__ == "__main__":
    main()
