from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

## self defined CNN block
class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

## self defined ResNet block
class ResidualBlock(nn.Module):
    """
    A small ResNet-style block (2 convs) adapted for MNIST.
    - If shape changes (stride!=1 or channels differ), uses a 1x1 projection on the skip.
    """

    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(in_ch, out_ch, kernel_size=3, stride=stride, act=True)
        self.drop = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.conv2 = ConvBNAct(out_ch, out_ch, kernel_size=3, stride=1, act=False)

        if stride != 1 or in_ch != out_ch:
            self.skip = ConvBNAct(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, act=False)
        else:
            self.skip = nn.Identity()

        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.drop(out)
        out = self.conv2(out)
        out = out + identity
        return self.out_act(out)

## save the hyperparameters of the model
@dataclass(frozen=True)
class MNISTNetConfig:
    stem_channels: int = 32
    stage2_channels: int = 64
    stage3_channels: int = 128
    blocks_stage1: int = 2
    blocks_stage2: int = 2
    blocks_stage3: int = 2
    block_dropout_p: float = 0.05
    head_dropout_p: float = 0.15

## build the whole CNN model
class MNISTNet(nn.Module):
    """
    Custom CNN for MNIST (not LeNet): Conv stem + residual stages + global average pooling.
    Input:  (B, 1, 28, 28)
    Output: (B, 10) logits
    """

    def __init__(self, cfg: MNISTNetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or MNISTNetConfig()

        c1 = self.cfg.stem_channels
        c2 = self.cfg.stage2_channels
        c3 = self.cfg.stage3_channels

        self.stem = nn.Sequential(
            ConvBNAct(1, c1, kernel_size=3, stride=1),
            ConvBNAct(c1, c1, kernel_size=3, stride=1),
        )

        self.stage1 = self._make_stage(c1, c1, num_blocks=self.cfg.blocks_stage1, stride=1)
        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28 -> 14

        self.stage2 = self._make_stage(c1, c2, num_blocks=self.cfg.blocks_stage2, stride=1)
        self.down2 = ConvBNAct(c2, c2, kernel_size=3, stride=2)  # 14 -> 7

        self.stage3 = self._make_stage(c2, c3, num_blocks=self.cfg.blocks_stage3, stride=1)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=self.cfg.head_dropout_p),
            nn.Linear(c3, 10),
        )

        self._init_weights()

    def _make_stage(self, in_ch: int, out_ch: int, *, num_blocks: int, stride: int) -> nn.Sequential:
        blocks: list[nn.Module] = []
        blocks.append(ResidualBlock(in_ch, out_ch, stride=stride, dropout_p=self.cfg.block_dropout_p))
        for _ in range(num_blocks - 1):
            blocks.append(ResidualBlock(out_ch, out_ch, stride=1, dropout_p=self.cfg.block_dropout_p))
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.pool(x)
        return self.head(x)


def create_model() -> MNISTNet:
    return MNISTNet()