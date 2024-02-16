
# developer: Taoshi
# Copyright © 2024 Taoshi Inc

import hashlib


class HashingUtils:

    @staticmethod
    def hash_predictions(hotkey_address, predictions):
        hash_predictions = hotkey_address + "-" + str(predictions)
        hash_object = hashlib.sha256(hash_predictions.encode())
        return hash_object.hexdigest()
