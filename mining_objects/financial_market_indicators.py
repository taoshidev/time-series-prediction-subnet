# developer: Taoshidev
# Copyright © 2023 Taoshi, LLC

from typing import List

import numpy as np
import pandas as pd


class FinancialMarketIndicators:


    @staticmethod
    def calculate_rsi(closes: List[float],
                      period=14):
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
    def calculate_macd(closes: List[float], short_window=12, long_window=26, signal_window=9):
        macd_lines = []
        signal_lines = []

        for i in range(len(closes)):
            if i < max(long_window, signal_window) - 1:
                macd_lines.append(None)
                signal_lines.append(None)
            else:
                short_ema = np.convolve(closes[:i + 1], np.ones(short_window) / short_window, mode='valid')
                long_ema = np.convolve(closes[:i + 1], np.ones(long_window) / long_window, mode='valid')

                if len(short_ema) < len(long_ema):
                    short_ema = np.concatenate((np.full(len(long_ema) - len(short_ema), np.nan), short_ema))
                elif len(long_ema) < len(short_ema):
                    long_ema = np.concatenate((np.full(len(short_ema) - len(long_ema), np.nan), long_ema))

                macd_line = short_ema - long_ema
                signal_line = np.convolve(macd_line, np.ones(signal_window) / signal_window, mode='valid')

                macd_lines.append(macd_line[-1])
                signal_lines.append(signal_line[-1])

        return macd_lines, signal_lines

    @staticmethod
    def calculate_bollinger_bands(closes: List[float], window=20, num_std=2):

        middle_bands = []
        upper_bands = []
        lower_bands = []

        for i in range(len(closes)):
            if i < window - 1:
                middle_bands.append(None)
                upper_bands.append(None)
                lower_bands.append(None)
            else:
                middle_band = np.mean(closes[i - window + 1:i + 1])
                rolling_std = np.std(closes[i - window + 1:i + 1], ddof=0)

                upper_band = middle_band + (num_std * rolling_std)
                lower_band = middle_band - (num_std * rolling_std)

                middle_bands.append(middle_band)
                upper_bands.append(upper_band)
                lower_bands.append(lower_band)

        return middle_bands, upper_bands, lower_bands

    @staticmethod
    def calculate_ema(closes: List[float], window=9):
        alpha = 2 / (window + 1)

        ema = np.full(window, np.nan)
        ema = np.concatenate((ema, [np.mean(closes[:window])]))

        for i in range(window + 1, len(closes)):
            ema_value = alpha * closes[i] + (1 - alpha) * ema[i - 1]
            ema = np.append(ema, ema_value)

        return ema

    @staticmethod
    def calculate_sma(closes: List[float], window=9):
        sma_values = []

        for i in range(len(closes)):
            if i < window - 1:
                sma_values.append(None)
            else:
                sma_value = np.mean(closes[i - window + 1:i + 1])
                sma_values.append(sma_value)

        return sma_values

    @staticmethod
    def calculate_stochastic_rsi(closes: List[float], window=14, smooth_k=3, smooth_d=3):
        rsi_values = FinancialMarketIndicators.calculate_rsi(closes, period=window)
        if len(rsi_values) < window:
            return [None] * len(rsi_values)

        valid_rsi_values = [value for value in rsi_values if value is not None]
        rsi_values_np = np.array(valid_rsi_values)

        normalized_rsi = (rsi_values_np - np.min(rsi_values_np)) / (np.max(rsi_values_np) - np.min(rsi_values_np)) * 100

        stoch_rsi_k = np.convolve(normalized_rsi, np.ones(smooth_k) / smooth_k, mode='valid')
        stoch_rsi_d = np.convolve(stoch_rsi_k, np.ones(smooth_d) / smooth_d, mode='valid')

        stoch_rsi_k = np.concatenate((np.full(window + smooth_k - 2, np.nan), stoch_rsi_k))
        stoch_rsi_d = np.concatenate((np.full(window + smooth_k + smooth_d - 3, np.nan), stoch_rsi_d))

        return stoch_rsi_k, stoch_rsi_d

    @staticmethod
    def calculate_vwap(closes: List[float], volumes: List[float]) -> float:
        if len(closes) != len(volumes):
            raise ValueError("Lengths of prices and volumes must be the same.")

        cumulative_price_volume = sum(price * volume for price, volume in zip(closes, volumes))
        cumulative_volume = sum(volumes)

        vwap = cumulative_price_volume / cumulative_volume

        return vwap

    @staticmethod
    def calculate_vwap_interval(closes: List[float], volumes: List[float], number_of_rows: int, fill_values: bool):
        last_ind = 0
        vwaps = []
        for ind in range(len(closes)):
            if (ind+1) % number_of_rows == 0:
                curr_closes = closes[last_ind:ind]
                curr_volumes = volumes[last_ind:ind]
                curr_vwap = FinancialMarketIndicators.calculate_vwap(curr_closes, curr_volumes)
                if fill_values:
                    vwaps.extend([curr_vwap for i in range(number_of_rows)])
                else:
                    vwaps.append(curr_vwap)
                last_ind = ind
        return vwaps

    @staticmethod
    def calculate_vrvp(closes: List[float], volumes: List[float], visible_range: int) -> List[float]:
        if len(closes) != len(volumes) or visible_range <= 0:
            raise ValueError("Invalid inputs.")

        vrvp_values = []

        for i in range(len(closes)):
            start_index = max(0, i - visible_range + 1)
            end_index = i + 1

            close_slice = closes[start_index:end_index]
            volume_slice = volumes[start_index:end_index]

            vrvp = sum(volume * (close - min(close_slice)) for close, volume in zip(close_slice, volume_slice))
            vrvp_values.append(vrvp)

        return vrvp_values

    @staticmethod
    def calculate_sum_vrvp_per_close(closes: List[float], volumes: List[float], visible_range: int) -> dict:
        if len(closes) != len(volumes) or visible_range <= 0:
            raise ValueError("Invalid inputs.")

        vrvp = FinancialMarketIndicators.calculate_vrvp(closes, volumes, visible_range)

        close_to_vrvp = {}
        for ind, row in enumerate(vrvp):
            curr_close = closes[ind]
            if curr_close not in close_to_vrvp:
                close_to_vrvp[curr_close] = 0
            close_to_vrvp[curr_close] += row
        return close_to_vrvp

    @staticmethod
    def calculate_sum_vrvp_per_grouping_size(closes: List[float], volumes: List[float], grouping_size: int, visible_range: int) -> dict:
        if len(closes) != len(volumes) or visible_range <= 0:
            raise ValueError("Invalid inputs.")

        vrvp = FinancialMarketIndicators.calculate_vrvp(closes, volumes, visible_range)

        close_to_vrvp = {}
        for ind, row in enumerate(vrvp):
            curr_close = closes[ind]
            if curr_close not in close_to_vrvp:
                close_to_vrvp[curr_close] = 0
            close_to_vrvp[curr_close] += row

        closes_min = min(closes)
        closes_max = max(closes)

        closes_groupings = {}

        curr_grouping_value = closes_min
        while True:
            if curr_grouping_value + grouping_size < closes_max:
                closes_groupings[curr_grouping_value] = 0
                curr_grouping_value += grouping_size
            else:
                break

        for key, value in close_to_vrvp.items():
            for close_grouping in closes_groupings.keys():
                if close_grouping < key < close_grouping + grouping_size:
                    closes_groupings[close_grouping] += value
        return closes_groupings









