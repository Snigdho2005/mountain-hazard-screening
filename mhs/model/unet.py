"""Compact U-Net (Ronneberger et al., 2015) for six-class surface mapping."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout, dropout=0.0):
        layers = [
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout2d(dropout))
        super().__init__(*layers)


class UNet(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, base: int = 32, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.down = nn.ModuleList([DoubleConv(in_channels, chs[0])])
        for i in range(depth):
            self.down.append(DoubleConv(chs[i], chs[i + 1], dropout if i == depth - 1 else 0.0))
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in reversed(range(depth)):
            self.up.append(nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2))
            self.dec.append(DoubleConv(chs[i] * 2, chs[i]))
        self.head = nn.Conv2d(chs[0], n_classes, 1)
        self.depth = depth

    def forward(self, x):
        h, w = x.shape[-2:]
        m = 2 ** self.depth
        ph, pw = (-h) % m, (-w) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        skips = []
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.up, self.dec):
            x = up(x)
            x = dec(torch.cat([skips.pop(), x], dim=1))
        return self.head(x)[..., :h, :w]
