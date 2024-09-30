import torch
import torch.nn as nn

class NetworkSmall(nn.Module):
    def __init__(self):
        super(NetworkSmall, self).__init__()
        self.block_0 = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=5, padding=2),
            nn.LeakyReLU()
        )
        self.block_1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.LeakyReLU()
        )
        self.output = nn.Conv2d(32, 4, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.block_0(x)
        x = self.block_1(x)
        x = self.output(x)
        return x
