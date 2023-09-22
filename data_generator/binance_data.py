# developer: Taoshidev
# Copyright © 2023 Taoshi, LLC

from datetime import datetime
from typing import List, Tuple

import requests
from requests import Response
import time

from vali_config import ValiConfig


class BinanceData:

    @staticmethod
    def get_historical_data(symbol='BTCUSDT',
                            interval=ValiConfig.STANDARD_TF_BINANCE,
                            start=None,
                            end=None,
                            limit=1000,
                            retries=0) -> Response:
        if start is None:
            # minute, minutes, hours, days, weeks
            start = str(int(datetime.now().timestamp() * 1000) - 60000 * 60 * 24 * 7 * 2)
            # start = str(int(datetime.now().timestamp() * 1000) - 60000 * 1250)
        if end is None:
            end = str(int(datetime.now().timestamp() * 1000))

        params = "symbol={symbol}&" \
                 "interval={interval}&" \
                 "startTime={startTime}&" \
                 "endTime={endTime}&" \
                 "limit={limit}".format(symbol=symbol,
                                        interval=interval,
                                        startTime=start,
                                        endTime=end,
                                        limit=limit)

        url = 'https://api.binance.com/api/v3/klines?' + params

        try:
            return requests.get(url)
        except Exception:
            if retries < 5:
                time.sleep(retries)
                retries += 1
                print("retrying getting historical binance data")
                BinanceData.get_historical_data(symbol, interval, start, end, limit, retries)
            else:
                raise ConnectionError("max number of retries exceeded trying to get binance data")

    @staticmethod
    def convert_output_to_data_points(data_structure: List[List], days_data: List[List]):
        """
        return open, high, low, vol
        close ind 4
        high ind 2
        low ind 3
        vol ind 5
        """
        for tf_row in days_data:
            data_structure[0].append(float(tf_row[4]))
            data_structure[1].append(float(tf_row[2]))
            data_structure[2].append(float(tf_row[3]))
            data_structure[3].append(float(tf_row[5]))

    @staticmethod
    def get_data_and_structure_data_points(symbol: str, data_structure: List[List], ts_range: Tuple[int, int]):
        bd = BinanceData().get_historical_data(symbol=symbol,
                                               start=ts_range[0],
                                               end=ts_range[1]).json()
        if "msg" in bd:
            raise Exception("error occurred getting Binance data, please review", bd["msg"])
        else:
            BinanceData.convert_output_to_data_points(data_structure,
                                                      BinanceData().get_historical_data(start=ts_range[0],
                                                                                        end=ts_range[1]).json())
