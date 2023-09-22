# developer: Taoshidev
# Copyright © 2023 Taoshi, LLC

import unittest

# Create a test loader
from vali_config import ValiConfig

if __name__ == '__main__':
    loader = unittest.TestLoader()

    # Discover all test files in the specified directory
    start_dir = ValiConfig.BASE_DIR + "/tests/vali_tests/" # Replace with the actual directory path
    suite = loader.discover(start_dir, pattern='test_*.py')

    # Create a test runner
    runner = unittest.TextTestRunner()

    # Run the tests
    result = runner.run(suite)