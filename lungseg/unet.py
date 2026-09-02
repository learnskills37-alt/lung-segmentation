"""U-Net for binary lung-field segmentation (Ronneberger et al., 2015) with
batch-norm blocks, configurable depth/width and optional bilinear upsampling."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, bilinear: bool, dropout: float):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            up_channels = in_channels
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            up_channels = in_channels // 2
        self.conv = DoubleConv(up_channels + skip_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dy = skip.shape[-2] - x.shape[-2]
        dx = skip.shape[-1] - x.shape[-1]
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        bilinear: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = depth

        channels = [base_channels * 2**i for i in range(depth + 1)]
        self.inc = DoubleConv(in_channels, channels[0])
        self.downs = nn.ModuleList(
            nn.Sequential(nn.MaxPool2d(2), DoubleConv(channels[i], channels[i + 1], dropout))
            for i in range(depth)
        )
        self.ups = nn.ModuleList(
            Up(channels[i + 1], channels[i], channels[i], bilinear, dropout)
            for i in reversed(range(depth))
        )
        self.outc = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        x = skips[-1]
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip)
        return self.outc(x)

    @property
    def size_divisor(self) -> int:
        return 2**self.depth


def build_unet(**kwargs) -> UNet:
    return UNet(**kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
