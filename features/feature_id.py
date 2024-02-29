# developer: taoshi-mbrown
# Copyright © 2024 Taoshi Inc
from enum import IntEnum

# TODO: Remove feature_ids property from protocols


class FeatureID(IntEnum):
    EPOCH_TIMESTAMP_MS = 0

    TIME_OF_DAY = 100
    TIME_OF_WEEK = 101
    TIME_OF_MONTH = 102
    TIME_OF_YEAR = 103

    CRYPTO_SOCIAL_POSTS_CREATED = 20000
    CRYPTO_SOCIAL_POSTS_ACTIVE = 20001
    CRYPTO_SOCIAL_INTERACTIONS = 20002
    CRYPTO_SOCIAL_CONTRIBUTORS_CREATED = 20003
    CRYPTO_SOCIAL_CONTRIBUTORS_ACTIVE = 20004
    CRYPTO_SOCIAL_SENTIMENT = 20005
    CRYPTO_SOCIAL_SPAM = 20006

    BTC_HASH_RATE = 100000

    BTC_USD_MARKET_CAP = 100010
    BTC_CIRCULATING_SUPPLY = 100011

    BTC_ADDR_COUNT_100K_USD = 100105
    BTC_ADDR_COUNT_1M_USD = 100106
    BTC_ADDR_COUNT_10M_USD = 100107

    BTC_MCTC = 100200
    BTC_MCRC = 100201
    BTC_MOMR = 100202

    BTC_GALAXY_SCORE = 100210
    BTC_ALT_RANK = 100211

    BTC_USD_OPEN = 101000
    BTC_USD_CLOSE = 101001
    BTC_USD_HIGH = 101002
    BTC_USD_LOW = 101003
    BTC_USD_VOLUME = 101004
    BTC_USD_TRADE_COUNT = 101005

    BTC_USD_VOLATILITY = 101100
    BTC_USD_SPREAD = 101101

    BTC_USD_IV_BID = 101200
    BTC_USD_IV_ASK = 101201
    BTC_USD_IV_MARK = 101202

    BTC_USD_FUTURES_FUNDING_RATE = 102000

    BTC_USD_FUTURES_OPEN_CONTRACTS = 102100

    BTC_USD_FUTURES_LIQUIDATIONS_BUY = 102200
    BTC_USD_FUTURES_LIQUIDATIONS_BUY_USD = 102201
    BTC_USD_FUTURES_LIQUIDATIONS_SELL = 102202
    BTC_USD_FUTURES_LIQUIDATIONS_SELL_USD = 102203

    BTC_SOCIAL_POSTS_CREATED = 151000
    BTC_SOCIAL_POSTS_ACTIVE = 151001
    BTC_SOCIAL_INTERACTIONS = 151002
    BTC_SOCIAL_CONTRIBUTORS_CREATED = 151003
    BTC_SOCIAL_CONTRIBUTORS_ACTIVE = 151004
    BTC_SOCIAL_SENTIMENT = 151005
    BTC_SOCIAL_SPAM = 151006
    BTC_SOCIAL_DOMINANCE = 151007

    NVDA_USD_OPEN = 701000
    NVDA_USD_CLOSE = 701001
    NVDA_USD_HIGH = 701002
    NVDA_USD_LOW = 701003
    NVDA_USD_VOLUME = 701004
