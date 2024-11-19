import unittest

import torch
import numpy as np

import tests.nets.tcf.net_small as net_small
from tests.nets.ks.ks_networks import ConvResNet1D
from tests.nets.ks.ks_solver import DifferentiableKS

import pbdl.torch.loader as pbdl_torch
import pbdl.torch.phi.loader as pbdl_phi


class TestTraining(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_torch_training(self):
        loader = pbdl_torch.Dataloader(
            "transonic-cylinder-flow-tiny",
            time_steps=10,
            sel_sims=[0, 1],
            step_size=3,
            normalize_data="std",
            batch_size=3,
            shuffle=True,
        )

        net = net_small.NetworkSmall()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.0)

        for epoch in range(3):
            for input, target in loader:

                net.zero_grad()
                output = net(input)

                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        self.assertLess(loss.item(), 0.5)

    def test_torch_training_with_solver(self):
        domain_size_base = 8
        predhorz = 5

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        diff_ks = DifferentiableKS(resolution=48, dt=0.5)

        loader = pbdl_phi.Dataloader(
            "ks-dataset",
            time_steps=predhorz,
            step_size=20,
            intermediate_time_steps=True,
            batch_size=16,
            batch_by_const=[0],
            ret_batch_const=True,
        )

        net = ConvResNet1D(16, 3, device=device)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
        criterion = torch.nn.MSELoss()

        for epoch in range(3):
            for input, targets, const in loader:

                input = input.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                domain_size = const[0]

                inputs = [input]
                outputs = []

                for _ in range(predhorz):
                    output_solver = diff_ks.etd1(
                        loader.to_phiflow(inputs[-1]), domain_size_base * domain_size
                    )

                    correction = diff_ks.dt * net(inputs[-1])
                    output_combined = loader.from_phiflow(output_solver) + correction

                    outputs.append(output_combined)
                    inputs.append(loader.cat_constants(outputs[-1], inputs[0]))

                outputs = torch.stack(outputs, axis=1)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        self.assertLess(loss.item(), 2)
