"""
model.py — U-Net multi-task architecture.

Single encoder, 5 output heads (one per flood depth).

Why multi-task over 5 separate models:
  - Shared encoder = shared representations = better cross-region transfer
  - 5x fewer parameters than 5 separate U-Nets
  - Implicit regularisation: encoder must learn features useful across depths
  - Faster training

The architecture is deliberately small (~3-5M parameters):
  - Hackathon dataset is small (few thousand independent patches)
  - Larger networks would overfit Severn and fail on Northumbria
  - Lightweight = fast iteration on Colab
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive Conv-BN-ReLU blocks."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Downsampling: maxpool + DoubleConv."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, dropout)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upsampling: bilinear upsample + concat skip + DoubleConv."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        # Bilinear upsample (parameter-free) + 1x1 to reduce channels
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
        )
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if shapes don't exactly match (rare with size 256 / stride 2)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetMultiTask(nn.Module):
    """
    Multi-task U-Net for flood risk segmentation.

    Architecture (with base_channels=32, depth=4):
      Encoder:
        E0:  in -> 32
        E1: 32 -> 64    (1/2)
        E2: 64 -> 128   (1/4)
        E3: 128 -> 256  (1/8)
      Bottleneck:
        B:  256 -> 512  (1/16)
      Decoder:
        D3: 512 + 256 -> 256
        D2: 256 + 128 -> 128
        D1: 128 + 64  -> 64
        D0:  64 + 32  -> 32
      Heads (per depth):
        Conv1x1: 32 -> n_classes

    Output: 5 tensors of shape (B, n_classes, H, W).
    """
    def __init__(
        self,
        in_channels: int,
        n_classes: int = 5,
        base_channels: int = 32,
        depth: int = 4,
        n_tasks: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.depth = depth
        self.n_tasks = n_tasks
        self.n_classes = n_classes

        # Encoder path
        chs = [base_channels * (2 ** i) for i in range(depth + 1)]
        # e.g. depth=4 -> [32, 64, 128, 256, 512]
        self.input_conv = DoubleConv(in_channels, chs[0], dropout=0.0)

        self.downs = nn.ModuleList()
        for i in range(depth):
            self.downs.append(Down(chs[i], chs[i + 1], dropout=dropout if i >= 2 else 0.0))

        # Decoder path
        self.ups = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.ups.append(Up(in_ch=chs[i], skip_ch=chs[i - 1], out_ch=chs[i - 1], dropout=0.0))

        # One head per task
        self.heads = nn.ModuleList([
            nn.Conv2d(chs[0], n_classes, kernel_size=1)
            for _ in range(n_tasks)
        ])

    def forward(self, x):
        # Encoder
        skips = [self.input_conv(x)]
        h = skips[0]
        for down in self.downs:
            h = down(h)
            skips.append(h)
        # skips[-1] is the bottleneck

        # Decoder
        out = skips[-1]
        for i, up in enumerate(self.ups):
            skip = skips[-2 - i]
            out = up(out, skip)

        # 5 heads
        outputs = [head(out) for head in self.heads]
        # Each output: (B, n_classes, H, W)
        return outputs

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(config: dict, in_channels: int) -> UNetMultiTask:
    """Construct U-Net from config dict."""
    m = config["model"]
    model = UNetMultiTask(
        in_channels=in_channels,
        n_classes=config["n_classes"],
        base_channels=m["base_channels"],
        depth=m["depth"],
        n_tasks=len(config["targets"]),
        dropout=m["dropout"],
    )
    return model
