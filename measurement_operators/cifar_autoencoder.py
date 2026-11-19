import torch
import torch.nn as nn
from typing import Sequence


class CifarEncoder(nn.Module):
    """
    Conv encoder with len(hidden_channels) stride-2 stages, followed by a 1x1 conv
    that projects to the bottleneck.

    Input:  (B, in_channels, 32, 32)
    Output: (B, embedding_channels, 32 / 2**n, 32 / 2**n)   with n = len(hidden_channels)

    Default (3 stages): (B, 16, 4, 4) -> 256 numbers per image (12x compression of 3072).
    The original 5-stage design would give (B, C, 1, 1) on 32x32 inputs.
    """

    def __init__(self,
                 in_channels: int = 3,
                 embedding_channels: int = 16,
                 hidden_channels: Sequence[int] = (64, 128, 256),
                 dropout_rate: float = 0.1):
        super().__init__()
        layers = []
        prev = in_channels
        for i, ch in enumerate(hidden_channels):
            layers += [
                nn.Conv2d(prev, ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            ]
            if i > 0:  # no dropout right after the first conv, as in the original
                layers.append(nn.Dropout(p=dropout_rate))
            prev = ch

        # bottleneck projection (no spatial change)
        layers += [
            nn.Conv2d(prev, embedding_channels, kernel_size=1),
            nn.BatchNorm2d(embedding_channels),
            nn.ReLU(inplace=True),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CifarDecoder(nn.Module):
    """
    Mirror of CifarEncoder: 1x1 conv back up, then len(hidden_channels) x
    (nearest upsample -> 3x3 conv), then a final 3x3 conv + Sigmoid so outputs
    live in [0, 1] like the ToTensor() targets.

    Pass the same hidden_channels you gave the encoder.
    """

    def __init__(self,
                 out_channels: int = 3,
                 embedding_channels: int = 16,
                 hidden_channels: Sequence[int] = (64, 128, 256),
                 dropout_rate: float = 0.1):
        super().__init__()
        rev = list(hidden_channels)[::-1]            # e.g. [256, 128, 64]
        stage_out = rev[1:] + [rev[-1]]               # e.g. [128, 64, 64]

        layers = [
            nn.Conv2d(embedding_channels, rev[0], kernel_size=1),
            nn.BatchNorm2d(rev[0]),
            nn.ReLU(inplace=True),
        ]
        prev = rev[0]
        for ch in stage_out:
            layers += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(prev, ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
            ]
            prev = ch

        # No BatchNorm on the output layer: it would force batch-normalized
        # outputs even though the targets are fixed pixel values in [0, 1].
        layers += [
            nn.Conv2d(prev, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


if __name__ == "__main__":
    enc, dec = CifarEncoder(), CifarDecoder()
    x = torch.rand(8, 3, 32, 32)
    z = enc(x)
    y = dec(z)
    print("embedding:", tuple(z.shape), "-> dim", z[0].numel())
    print("recon:    ", tuple(y.shape))
    assert y.shape == x.shape
