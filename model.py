"""Depth estimation models: lightweight UNet baseline + pretrained encoder UNet."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBlock(nn.Module):
    """Conv-BN-ReLU block (stable training vs bare ReLU conv stacks)."""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Bilinear upsample + skip concat + conv refinement."""

    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        self.conv = ConvBlock(in_c + skip_c, out_c)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DoubleConv(nn.Module):
    """Legacy block kept for backward-compatible tiny UNet."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """Small from-scratch UNet (no pretrained weights). Kept for smoke tests."""

    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.middle = DoubleConv(128, 256)
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        mid = self.middle(p2)
        u1 = self.up1(mid)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.conv1(u1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.conv2(u2)
        return torch.sigmoid(self.out(u2))


class ResNetEncoderUNet(nn.Module):
    """
    UNet decoder on a pretrained ResNet encoder.
    Skip connections at 4 scales preserve edges and thin structures.
    """

    def __init__(self, encoder: str = "resnet50", pretrained: bool = True):
        super().__init__()
        self.encoder_name = encoder

        weights = None
        if pretrained:
            if encoder == "resnet34":
                weights = models.ResNet34_Weights.IMAGENET1K_V1
                backbone = models.resnet34(weights=weights)
                channels = [64, 64, 128, 256, 512]
            elif encoder == "resnet50":
                weights = models.ResNet50_Weights.IMAGENET1K_V1
                backbone = models.resnet50(weights=weights)
                channels = [64, 256, 512, 1024, 2048]
            else:
                raise ValueError(f"Unknown encoder: {encoder}")
        else:
            if encoder == "resnet34":
                backbone = models.resnet34(weights=None)
                channels = [64, 64, 128, 256, 512]
            elif encoder == "resnet50":
                backbone = models.resnet50(weights=None)
                channels = [64, 256, 512, 1024, 2048]
            else:
                raise ValueError(f"Unknown encoder: {encoder}")

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        c1, c2, c3, c4, c5 = channels
        self.up4 = UpBlock(c5, c4, 256)
        self.up3 = UpBlock(256, c3, 128)
        self.up2 = UpBlock(128, c2, 64)
        self.up1 = UpBlock(64, c1, 32)

        self.head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x):
        h, w = x.shape[2:]
        x0 = self.stem(x)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        d = self.up4(x4, x3)
        d = self.up3(d, x2)
        d = self.up2(d, x1)
        d = self.up1(d, x0)
        d = F.interpolate(d, size=(h, w), mode="bilinear", align_corners=False)
        return torch.sigmoid(self.head(d))


def build_model(name: str = "resnet50", pretrained: bool = True) -> nn.Module:
    """Factory: 'unet' | 'resnet34' | 'resnet50'."""
    if name == "unet":
        return UNet()
    if name in ("resnet34", "resnet50"):
        return ResNetEncoderUNet(encoder=name, pretrained=pretrained)
    raise ValueError(f"Unknown model: {name}. Use unet, resnet34, or resnet50.")
