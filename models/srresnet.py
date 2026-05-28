import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, channels, scale):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * scale * scale, 3, 1, 1)
        self.ps = nn.PixelShuffle(scale)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.ps(self.conv(x)))


class SRResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, channels=64, num_blocks=16, scale=4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, channels, 9, 1, 4),
            nn.PReLU(),
        )

        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.body_tail = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )

        upsample_layers = []
        if scale in [2, 4, 8]:
            for _ in range(int(torch.log2(torch.tensor(scale)))):
                upsample_layers.append(UpsampleBlock(channels, 2))
        elif scale == 3:
            upsample_layers.append(UpsampleBlock(channels, 3))
        else:
            raise ValueError("scale must be one of [2, 3, 4, 8]")

        self.upsample = nn.Sequential(*upsample_layers)
        self.tail = nn.Conv2d(channels, out_channels, 9, 1, 4)

    def forward(self, x):
        x = self.head(x)
        res = x
        x = self.body(x)
        x = self.body_tail(x)
        x = x + res
        x = self.upsample(x)
        x = self.tail(x)
        return torch.clamp(x, 0.0, 1.0)
