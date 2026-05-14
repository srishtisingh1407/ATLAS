from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetSmall(nn.Module):
    """
    Small UNet for binary pixel-level change detection.

    Inputs: stacked pre/post channels (e.g. 3-band pre + 1-band post = 4 channels).
    Output: single-channel logit map; apply sigmoid + threshold to get binary change mask.

    Dropout layers are present but inactive (eval mode) during normal inference.
    Call model.enable_mc_dropout() after model.eval() to activate them for MC Dropout.
    """

    def __init__(self, in_channels: int, base: int = 32, dropout_p: float = 0.3) -> None:
        super().__init__()

        self.enc1 = ConvBlock(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base * 4, base * 8)
        # Dropout after bottleneck and decoder blocks — used for MC Dropout uncertainty.
        # nn.Dropout2d has no parameters so state_dict keys are unchanged vs old checkpoints.
        self.drop_b = nn.Dropout2d(p=dropout_p)
        self.drop_d3 = nn.Dropout2d(p=dropout_p * 0.5)
        self.drop_d2 = nn.Dropout2d(p=dropout_p * 0.5)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)

        self.head = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.drop_b(self.bottleneck(self.pool3(e3)))

        d3 = self.drop_d3(self.dec3(torch.cat([self.up3(b), e3], dim=1)))
        d2 = self.drop_d2(self.dec2(torch.cat([self.up2(d3), e2], dim=1)))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)

    def enable_mc_dropout(self) -> None:
        """Set only Dropout layers to train mode, keeping BN in eval mode.
        Call this after model.eval() to enable stochastic inference for MC Dropout."""
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()
