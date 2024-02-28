# developer: taoshi-tdougherty
# Copyright © 2023 Taoshi Inc

import pandas as pd
import requests
from datetime import datetime, timezone
import concurrent.futures
import sys

BINANCE_COLUMNS = [
    "Open Time",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Close Time",
    "Quote Asset Volume",
    "Number of Trades",
    "Taker Buy Base Asset Volume",
    "Taker Buy Quote Asset Volume",
    "Ignore",
]


def convert_to_ms(timestamp_str):
    """Convert datetime string to milliseconds since epoch."""
    dt_obj = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S")
    timestamp_ms = int(dt_obj.replace(tzinfo=timezone.utc).timestamp() * 1000)
    return timestamp_ms


def generate_api_urls(
    start_time, end_time, interval_ms=60000, limit=1000, symbol="BTCUSDT"
):
    """Generate API URLs for fetching data from Binance."""
    urls = []
    current_time = start_time
    while current_time < end_time:
        urls.append(
            f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&startTime={current_time}&endTime={min(current_time + interval_ms * limit, end_time)}&limit={limit}"
        )
        current_time += interval_ms * limit

    print(f"Generated urls: {urls}")
    return urls


def fetch_and_append_data(url, file_path):
    print(f"Fetching data from {url}")
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(
        data,
        columns=BINANCE_COLUMNS,
    )
    df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
    df["Close Time"] = pd.to_datetime(df["Close Time"], unit="ms")
    df.to_csv(file_path, mode="a", header=False, index=False)


def fetch_all_data_parallel(start_time, end_time, file_path, symbol="BTCUSDT"):
    urls = generate_api_urls(start_time, end_time, symbol=symbol)
    print(f"Urls: {urls}")
    # Ensure the file is empty with headers set
    pd.DataFrame(columns=BINANCE_COLUMNS).to_csv(file_path, index=False)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(fetch_and_append_data, urls, [file_path] * len(urls))
    # Sort and overwrite the CSV file
    df = pd.read_csv(output_file)
    df.sort_values(by="Open Time", ascending=False, inplace=True)
    df.to_csv(file_path, index=False)


if __name__ == "__main__":
    start_time = sys.argv[1]
    end_time = sys.argv[2]
    output_file = sys.argv[3]

    print(f"Fetching data from {start_time} to {end_time}")

    start = convert_to_ms(start_time)
    end = convert_to_ms(end_time)

    assert start < end
    fetch_all_data_parallel(start, end, output_file)
