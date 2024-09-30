import unittest
import h5py
import os
import numpy as np
import tests.setup_random
from pbdl.loader import *


class TestLocalDatasets(unittest.TestCase):
    def setUp(self):
        tests.setup_random.setup()

    def tearDown(self):
        tests.setup_random.teardown()

    def test_local_dir_scan(self):
        try:
            Dataloader("random", 1, local_datasets_dir="./tests/datasets/")
        except SystemExit:
            self.fail("Dataloader terminated (probably because the local data set could not be found)!")


