import unittest
import torch
import numpy as np
import pbdl.loader as pbdl_numpy
import pbdl.torch.loader as pbdl_torch
import pbdl.torch.phi.loader as pbdl_phi
from phi.torch.flow import *

import tests.setup_random


class TestLoader(unittest.TestCase):
    def setUp(self):
        tests.setup_random.setup()

    def tearDown(self):
        tests.setup_random.teardown()

    def test_sample_shape_with_intermediate_time_steps(self):
        loader = pbdl_phi.Dataloader(
            "solver-in-the-loop-wake-flow",
            batch_size=10,
            time_steps=5,
            sel_sims=[0, 1, 2, 3, 4, 5],
            intermediate_time_steps=True,
            shuffle=True,
        )

        for input_cpu, targets_cpu in loader:
            input = input_cpu.clone().detach()
            targets = targets_cpu.clone().detach()

            self.assertEqual(input.shape, (10, 4, 65, 32))
            self.assertEqual(targets.shape, (10, 5, 3, 65, 32))

    def test_all_time_steps(self):
        loader = pbdl_numpy.Dataloader(
            "random", all_time_steps=True, local_datasets_dir="./tests/datasets/"
        )

        for inputs, targets in loader:
            self.assertEqual(inputs.shape, (1, 5, 128, 64))  # first frame
            self.assertEqual(
                targets.shape, (1, 999, 4, 128, 64)
            )  # all remaining sim frames

    def test_split(self):
        loader_train, loader_val, loader_test = pbdl_torch.Dataloader.new_split(
            [2000, 500, 3 * 999 - 2000 - 500],
            "random",
            time_steps=1,
            batch_size=10,
            local_datasets_dir="./tests/datasets/",
        )
        self.assertEqual(len(loader_train), 2000 // 10)
        self.assertEqual(len(loader_val), 500 // 10)
        self.assertEqual(len(loader_test), (3 * 999 - 2000 - 500 + 9) // 10)
