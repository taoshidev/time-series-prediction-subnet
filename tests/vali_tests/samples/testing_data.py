# developer: Taoshidev
# Copyright © 2023 Taoshi, LLC

import datetime

from vali_objects.dataclasses.prediction_data_file import PredictionDataFile
import numpy as np


class TestingData:
    SCORES = {'0': 0, '1': 90, '2': 56, '3': 25, '4': 71, '5': 17, '6': 65, '7': 54, '8': 38, '9': 54, '10': 16, '11': 91, '12': 23, '13': 85, '14': 54, '15': 2, '16': 30, '17': 21, '18': 96, '19': 52, '20': 46, '21': 22, '22': 42, '23': 24, '24': 65, '25': 22, '26': 1, '27': 81, '28': 92, '29': 72, '30': 12, '31': 43, '32': 85, '33': 86, '34': 78, '35': 24, '36': 19, '37': 94, '38': 65, '39': 46, '40': 38, '41': 90, '42': 4, '43': 92, '44': 84, '45': 74, '46': 58, '47': 14, '48': 64, '49': 66, '50': 34, '51': 1, '52': 46, '53': 43, '54': 56, '55': 100, '56': 10, '57': 82, '58': 13, '59': 61, '60': 15, '61': 96, '62': 61, '63': 58, '64': 89, '65': 47, '66': 29, '67': 70, '68': 15, '69': 64, '70': 88, '71': 73, '72': 11, '73': 54, '74': 49, '75': 63, '76': 55, '77': 44, '78': 16, '79': 82, '80': 27, '81': 28, '82': 75, '83': 68, '84': 63, '85': 1, '86': 72, '87': 48, '88': 6, '89': 78, '90': 15, '91': 56, '92': 11, '93': 99, '94': 76, '95': 85, '96': 54, '97': 44, '98': 8, '99': 43}
    SCALED_SCORES = {'0': 1.0, '1': 0.4278381272579169, '2': 0.5923398545651801, '3': 0.866011325331195, '4': 0.5072466226424431, '5': 0.9479672709164924, '6': 0.5384093977519866, '7': 0.6056654252575165, '8': 0.733496638261556, '9': 0.6056654252575165, '10': 0.9567442453948123, '11': 0.42431681877076943, '12': 0.8874978120064971, '13': 0.4463244815451354, '14': 0.6056654252575165, '15': 0.9999999999877439, '16': 0.812691820518043, '17': 0.9086317415765133, '18': 0.4075196078866321, '19': 0.6195293172195785, '20': 0.6645865417227526, '21': 0.8981333739509296, '22': 0.6977281713035653, '23': 0.8767758223527625, '24': 0.5384093977519866, '25': 0.8981333739509296, '26': 1.0, '27': 0.4622547641944188, '28': 0.4208510914477629, '29': 0.5023790650845147, '30': 0.9848158020431621, '31': 0.6891996863013663, '32': 0.4463244815451354, '33': 0.44250532406252185, '34': 0.47493371605714263, '35': 0.8767758223527625, '36': 0.928975958182108, '37': 0.4140814358474151, '38': 0.5384093977519866, '39': 0.6645865417227526, '40': 0.733496638261556, '41': 0.4278381272579169, '42': 0.9999964991333296, '43': 0.4208510914477629, '44': 0.45020746758760477, '45': 0.4929035486089264, '46': 0.5795288077943455, '47': 0.9723819494415242, '48': 0.5439515613501873, '49': 0.5329708653197618, '50': 0.7718931630057799, '51': 1.0, '52': 0.6645865417227526, '53': 0.6891996863013663, '54': 0.5923398545651801, '55': 0.39498377310685673, '56': 0.9934284135050704, '57': 0.4581710284846512, '58': 0.9790451386751108, '59': 0.5612258626461133, '60': 0.964915645899155, '61': 0.4075196078866321, '62': 0.5612258626461133, '63': 0.5795288077943455, '64': 0.43141629687036287, '65': 0.6566994361813732, '66': 0.8232039765251555, '67': 0.5122035851494248, '68': 0.964915645899155, '69': 0.5439515613501873, '70': 0.4350526439292187, '71': 0.4975987442774704, '72': 0.9896231904973789, '73': 0.6056654252575165, '74': 0.6413865473242829, '75': 0.5495999303962555, '76': 0.5989368423231203, '77': 0.6808344848686338, '78': 0.9567442453948123, '79': 0.4581710284846512, '80': 0.8445002431626647, '81': 0.8338132058240615, '82': 0.48829142221345745, '83': 0.5223946849183732, '84': 0.5495999303962555, '85': 1.0, '86': 0.5023790650845147, '87': 0.6489669849611899, '88': 0.9997694401324075, '89': 0.47493371605714263, '90': 0.964915645899155, '91': 0.5923398545651801, '92': 0.9896231904973789, '93': 0.3980469083616047, '94': 0.4837603640377426, '95': 0.4463244815451354, '96': 0.6056654252575165, '97': 0.6808344848686338, '98': 0.9981289396935358, '99': 0.6891996863013663}
    po = PredictionDataFile(
        client_uuid="test",
        stream_type="TEST",
        stream_id="1",
        topic_id=1,
        request_uuid="testuid",
        miner_uid="test",
        start=1234,
        end=12345,
        vmins=[0.1, 0.2, 0.3],
        vmaxs=[10, 20, 30],
        decimal_places=[1, 2, 3],
        predictions=np.array([1, 2, 3]),
        prediction_size=10
    )
    test_start_time = datetime.datetime(2023, 9, 11, 0, 0)
    test_generated_timestamps = [(datetime.datetime(2023, 9, 11, 0, 0), datetime.datetime(2023, 9, 11, 23, 59, 59, 999999)), (datetime.datetime(2023, 9, 12, 0, 0), datetime.datetime(2023, 9, 12, 23, 59, 59, 999999)), (datetime.datetime(2023, 9, 13, 0, 0), datetime.datetime(2023, 9, 13, 23, 59, 59, 999999)), (datetime.datetime(2023, 9, 14, 0, 0), datetime.datetime(2023, 9, 14, 23, 59, 59, 999999)), (datetime.datetime(2023, 9, 15, 0, 0), datetime.datetime(2023, 9, 15, 23, 59, 59, 999999)), (datetime.datetime(2023, 9, 16, 0, 0), datetime.datetime(2023, 9, 16, 0, 0, 0, 0))]
