import unittest
import numpy as np
import numpy.testing as npt
import tests.setup_random
import h5py
import os

from pbdl.loader import *
from pbdl.normalization import StdNorm, MeanStdNorm, MinMaxNorm


class TestNormalization(unittest.TestCase):

    def setUp(self):
        tests.setup_random.setup()
        self.rand_dset = h5py.File("tests/datasets/random.hdf5", "r+")

    def tearDown(self):
        self.rand_dset.close()
        tests.setup_random.teardown()

    def test_std_norm(self):
        loader = Dataloader(
            "transonic-cylinder-flow-tiny",
            time_steps=10,
            normalize_data="std",
            normalize_const="std",
            batch_size=1,
        )

        std_input = [0] * 3  # 3 fields (1 vector field, 2 scalar fields)
        std_target = [0] * 3
        const = [] # one constant (Mach number)

        for input, target in loader:
            input = input[0]
            target = target[0]

            # calculate vector norm for velocity
            input[0] = np.linalg.norm(input[0:2], axis=0)
            input = np.delete(input, 1, axis=0)

            # calculate vector norm for velocity
            target[0] = np.linalg.norm(target[0:2], axis=0)
            target = np.delete(target, 1, axis=0)

            # calculating std over spatial dims
            for f in range(3):
                std_input[f] += np.std(input[f])
                std_target[f] += np.std(target[f])
            
            const.append(input[3])

        for f in range(3):
            std_input[f] /= len(loader)
            std_target[f] /= len(loader)

            self.assertAlmostEqual(std_input[f], 1, places=2)
            self.assertAlmostEqual(std_target[f], 1, places=2)
        
        std_const = np.std(const)
        self.assertAlmostEqual(std_const, 1, places=2)


    def test_std_norm_rev(self):
        norm = StdNorm()
        norm.prepare(self.rand_dset, sel_const=None)

        arr = np.random.rand(4, 128, 64)

        arr_norm = norm.normalize(arr)
        arr_rev = norm.normalize_rev(arr_norm)

        npt.assert_array_almost_equal(arr_rev, arr)

    def test_mean_std_norm(self):
        loader = Dataloader(
            "transonic-cylinder-flow-tiny",
            time_steps=1,
            normalize_data="mean-std",
            normalize_const="mean-std",
            batch_size=1,
        )

        mean_input = [0] * 4  # 4 scalar fields
        std_input = [0] * 4
        const = [] # one constant (Mach number)

        mean_target = [0] * 4
        std_target = [0] * 4

        for input, target in loader:
            input = input[0]
            target = target[0]

            for sf in range(4):
                mean_input[sf] += np.sum(input[sf])
                mean_target[sf] += np.sum(target[sf])
                std_input[sf] += np.std(input[sf])
                std_target[sf] += np.std(target[sf])
            
            const.append(input[4])

        for sf in range(4):
            mean_input[sf] /= len(loader)
            mean_target[sf] /= len(loader)
            std_input[sf] /= len(loader)
            std_target[sf] /= len(loader)

            self.assertAlmostEqual(
                mean_input[sf], 0, places=0
            )  # TODO precision, dataset too small
            self.assertAlmostEqual(mean_target[sf], 0, places=0)
            self.assertAlmostEqual(std_input[sf], 1, places=2)
            self.assertAlmostEqual(std_target[sf], 1, places=2)
        
        const_mean = np.mean(const)
        const_std = np.std(const)
        self.assertAlmostEqual(const_mean, 0, places=2)
        self.assertAlmostEqual(const_std, 1, places=2)


    def test_mean_std_norm_rev(self):
        norm = MeanStdNorm()
        norm.prepare(self.rand_dset, sel_const=None)

        arr = np.random.rand(4, 128, 64)

        arr_norm = norm.normalize(arr)
        arr_rev = norm.normalize_rev(arr_norm)

        npt.assert_array_almost_equal(arr_rev, arr)

    def test_min_max_norm(self):
        loader = Dataloader(
            "transonic-cylinder-flow-tiny",
            time_steps=10,
            normalize_data="minus-one-to-one",
            normalize_const="minus-one-to-one",
            batch_size=1,
        )

        min_input = [float("inf")] * 5  # 4 scalar fields + 1 constant
        max_input = [-float("inf")] * 5
        min_target = [float("inf")] * 4 # 4 scalar fields
        max_target = [-float("inf")] * 4

        for input, target in loader:
            input = input[0]
            target = target[0]

            for sf in range(5):
                min_input[sf] = min(min_input[sf], np.min(input[sf]))
                max_input[sf] = max(max_input[sf], np.max(input[sf]))
            
            for sf in range(4):
                min_target[sf] = min(min_target[sf], np.min(target[sf]))
                max_target[sf] = max(max_target[sf], np.max(target[sf]))

        for min_val in min_input:
            self.assertAlmostEqual(min_val, -1, places=5)

        for max_val in max_input:
            self.assertAlmostEqual(max_val, 1, places=5)

        for min_val in min_target:
            self.assertAlmostEqual(min_val, -1, places=5)

        for max_val in max_target:
            self.assertAlmostEqual(max_val, 1, places=5)

    def test_min_max_norm_rev(self):
        norm = MinMaxNorm(-1, 1)
        norm.prepare(self.rand_dset, sel_const=None)

        arr = np.random.rand(4, 128, 64)

        arr_norm = norm.normalize(arr)
        arr_rev = norm.normalize_rev(arr_norm)

        npt.assert_array_almost_equal(arr_rev, arr)
