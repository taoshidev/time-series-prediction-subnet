import time

from requests import ReadTimeout
from twelvedata import TDClient


class TwelveDataService:
    def __init__(self, api_key=None):
        # self._api_key = api_key
        self._api_key = "xxxx"

    def _fetch_data(self, symbols, interval, output_size):
        td = TDClient(apikey=self._api_key)

        ts = td.time_series(symbol=symbols, interval=interval, outputsize=output_size)

        response = ts.as_json()
        return response

    def get_data(
        self,
        trade_pair: str,
        output_size: int = 100,
        interval: str = "5min",
        retries: int = 5,
    ):
        try:
            data = self._fetch_data(trade_pair, interval, output_size)
            return data
        except ReadTimeout:
            time.sleep(5)
            retries -= 1
            if retries > 0:
                self.get_data(trade_pair)

    @staticmethod
    def get_closes(data):
        closes = [float(row["close"]) for row in data]
        closes.reverse()
        return closes


if __name__ == "__main__":
    twelve_data = TwelveDataService()
    btcusd_data = twelve_data.get_closes(twelve_data.get_data("BTC/USD"))
