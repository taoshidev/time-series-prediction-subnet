# developer: Taoshidev
# Copyright © 2023 Taoshi, LLC

class IncorrectPredictionSizeError(Exception):
    def __init__(self, message):
        super().__init__(self, message)