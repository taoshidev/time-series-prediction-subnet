# developer: Taoshidev
# Copyright © 2023 Taoshi, LLC

from datetime import datetime
from typing import List, Tuple

import requests
from requests import Response
import time

from data_generator.financial_markets_generator.base_financial_markets_generator.base_financial_markets_generator import \
    BaseFinancialMarketsGenerator
from time_util.time_util import TimeUtil
from vali_config import ValiConfig


class BinanceData(BaseFinancialMarketsGenerator):
    def __init__(self):
        super().__init__()
        self._symbols = {
            "BTCUSD": "BTCUSDT"
        }
        self._tf = {
            5: "5m",
            1: "1m"
        }

    def get_data(self,
                 symbol='BTCUSDT',
                 interval=ValiConfig.STANDARD_TF,
                 start=None,
                 end=None,
                 limit=9999,
                 retries=0) -> Response:

        if type(interval) == int:
            binance_interval = self._tf[interval]
        else:
            raise Exception("no mapping for binance interval")

        if start is None:
            # minute, minutes, hours, days, weeks
            start = str(int(datetime.now().timestamp() * 1000) - 60000 * 60 * 24 * 7 * 2)
            # start = str(int(datetime.now().timestamp() * 1000) - 60000 * 1250)
        if end is None:
            end = str(int(datetime.now().timestamp() * 1000))

        if symbol != "BTCUSDT":
            symbol = self._symbols[symbol]

        url = f'https://api.binance.com/api/v3/klines?symbol={symbol}' \
              f'&interval={binance_interval}&startTime={start}&endTime={end}&limit={limit}'

        response = requests.get(url)

        try:
            if response.status_code == 200:
                return response
            else:
                raise Exception("received error status code")
        except Exception:
            if retries < 5:
                time.sleep(retries)
                retries += 1
                # print("retrying getting historical binance data")
                self.get_data(symbol, interval, start, end, limit, retries)
            else:
                raise ConnectionError("max number of retries exceeded trying to get binance data")

    def get_data_and_structure_data_points(self, symbol: str, tf: int, data_structure: List[List], ts_range: Tuple[int, int]):

        # no guarantee that the amount of data is correct returned from Binance. Historical data can be patchy
        # therefore best to assume we need a certain number of iterations for data gathering based on inter
        iters = {
            1: 2,
            5: 1
        }

        d = []
        for i in range(iters[tf]):
            if len(d) == 0:
                last_row_gathered = ts_range[0]
            else:
                last_row_gathered = d[len(d)-1][0]
            curr_request = self.get_data(symbol=symbol, interval=tf, start=last_row_gathered, end=ts_range[1]).json()
            if "msg" in curr_request:
                raise Exception("error occurred getting Binance data, please review", curr_request["msg"])
            d.extend(curr_request)

        print(TimeUtil.millis_to_timestamp(d[0][0]))
        print(TimeUtil.millis_to_timestamp(d[len(d)-1][0]))
        self.convert_output_to_data_points(data_structure,
                                           d,
                                           [0, 4, 2, 3, 5]
                                           )
