# developer: Taoshidev
# Copyright © 2023 Taoshi, LLC

import os
import unittest


class TestBase(unittest.TestCase):

    def setUp(self) -> None:
        if "vm" in os.environ:
            del os.environ["vm"]
