# developer: Taoshidev
# Copyright © 2023 Taoshi, LLC

from typing import List
import pandas as pd


class FinancialMarketIndicators:

    @staticmethod
    def calculate_rsi(ds: List[List[float]],
                      period=14):
        closes = ds[0]
        if len(closes) < period:
            raise ValueError("Input list of closes is too short for the given period.")

        changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = [change if change > 0 else 0 for change in changes]
        losses = [-change if change < 0 else 0 for change in changes]

        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        rsi_values = [None] * (period - 1)

        for i in range(period - 1, len(closes)):
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

            rsi_values.append(rsi)

            if i < len(closes) - 1:
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        return rsi_values

    @staticmethod
    def calculate_macd(ds: List[List[float]],
                       short_period=12,
                       long_period=26,
                       signal_period=9):
        df = pd.DataFrame({'Close': ds[0]})

        df['ShortEMA'] = df['Close'].ewm(span=short_period, adjust=False).mean()
        df['LongEMA'] = df['Close'].ewm(span=long_period, adjust=False).mean()

        df['MACD'] = df['ShortEMA'] - df['LongEMA']

        df['Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()

        return df['MACD'].tolist(), df['Signal'].tolist()

    @staticmethod
    def calculate_bollinger_bands(ds: List[List[float]],
                                  window=20,
                                  num_std_dev=2):
        df = pd.DataFrame({'Close': ds[0]})

        df['Middle'] = df['Close'].rolling(window=window).mean()

        rolling_std = df['Close'].rolling(window=window).std()

        df['Upper'] = df['Middle'] + (num_std_dev * rolling_std)
        df['Lower'] = df['Middle'] - (num_std_dev * rolling_std)

        return df['Middle'].tolist(), df['Upper'].tolist(), df['Lower'].tolist()

    @staticmethod
    def calculate_ema(ds: List[List[float]], length = 9):
        closes = ds[0]

        alpha = 2 / (length + 1)

        ema = []

        for a in range(0, length):
            ema.append(None)

        ema.append(sum(closes[:length]) / length)

        for i in range(length+1, len(closes)):
            ema_value = alpha * closes[i] + (1 - alpha) * ema[i - 1]
            ema.append(ema_value)

        return ema
