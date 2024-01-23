# developer: Taoshi
# Copyright © 2023 Taoshi, LLC
from features import FeatureCollector
from feature_sources import BinaryFileFeatureStorage
from mining_objects.streams.btcusd_5m import (
    historical_sources,
    historical_feature_ids,
    INTERVAL_MS,
)
from time_util import closest_interval_ms, datetime, time_span_ms

if __name__ == "__main__":
    # choose the range of days to look back
    # number of days back start
    days_back_start = 200
    # number of days forward since end day
    # for example start from 100 days ago and get 70 days from 100 days ago
    # (100 days ago, 99 days ago, 98 days ago, etc.)
    days_back_end = 199

    sample_count_max = int(time_span_ms(days=1) / INTERVAL_MS)

    now = datetime.now()
    now_time_ms = now.timestamp_ms()
    start_time_ms = now_time_ms - time_span_ms(days=days_back_start)
    end_time_ms = start_time_ms + time_span_ms(days=days_back_end)

    start_time_ms = closest_interval_ms(start_time_ms, INTERVAL_MS)
    end_time_ms = closest_interval_ms(end_time_ms, INTERVAL_MS)

    historical_feature_collector = FeatureCollector(
        sources=historical_sources,
        feature_ids=historical_feature_ids,
        cache_results=False,
    )

    historical_feature_storage = BinaryFileFeatureStorage(
        filename="historical_financial_data/data.taosfs",
        mode="w",
        feature_ids=historical_feature_ids,
    )

    while start_time_ms < end_time_ms:
        sample_count = int((end_time_ms - start_time_ms) / INTERVAL_MS)
        sample_count = min(sample_count, sample_count_max)

        samples = historical_feature_collector.get_feature_samples(
            start_time_ms, INTERVAL_MS, sample_count
        )

        historical_feature_storage.set_feature_samples(
            start_time_ms, INTERVAL_MS, samples
        )

        start_time_ms += INTERVAL_MS * sample_count

    historical_feature_storage.close()
