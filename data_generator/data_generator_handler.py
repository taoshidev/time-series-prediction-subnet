from data_generator.financial_markets_generator.binance_data import BinanceData
from data_generator.financial_markets_generator.bybit_data import ByBitData


class DataGeneratorHandler:

    def _get_financial_markets_data(self, exchange_list_order_ind: int = 0, expected_length: int = 0, *args):
        exchange_list_order = [BinanceData(), ByBitData()]

        # if expected_length == 0:
        #     print("not going to compare to expected length")

        try:
            args = args[0]

            symbol = args[0]
            ds = args[1]
            ts_range = args[2]

            exchange_list_order[exchange_list_order_ind]\
                .get_data_and_structure_data_points(symbol=symbol, data_structure=ds, ts_range=ts_range)
            if expected_length != 0 and len(ds[0]) != expected_length:
                raise Exception(f"not expected length for results [{len(ds[0])}], "
                                f"expected [{expected_length}]")
        except Exception as e:
            exchange_list_order_ind += 1
            if exchange_list_order_ind > len(exchange_list_order)-1:
                raise Exception("could not get financial markets data from available exchanges, make sure you "
                                "are not in a restricted region and have network connectivity.")
            else:
                print("trying next exchange", exchange_list_order[exchange_list_order_ind])
                self._get_financial_markets_data(exchange_list_order_ind, expected_length, args)

    def data_generator_handler(self, topic_id: int, expected_length: int = 0, *args):
        topic_method_map = {
            1: self._get_financial_markets_data
        }
        return topic_method_map[topic_id](0, expected_length, args)


