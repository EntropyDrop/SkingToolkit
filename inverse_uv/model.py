import torch
import torch.nn as nn
import torch.nn.functional as F


def norm_groups(channels):
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.block = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        return self.block(self.down(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.block = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class InverseUVNet(nn.Module):
    def __init__(self, input_channels=6, base_channels=64, output_channels=4):
        super().__init__()
        c = base_channels
        self.stem = ConvBlock(input_channels, c)
        self.down1 = DownBlock(c, c * 2)
        self.down2 = DownBlock(c * 2, c * 4)
        self.down3 = DownBlock(c * 4, c * 8)
        self.down4 = DownBlock(c * 8, c * 8)
        self.mid = ConvBlock(c * 8, c * 8)
        self.up3 = UpBlock(c * 8, c * 8, c * 4)
        self.head = nn.Sequential(
            nn.Conv2d(c * 4, c, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, output_channels, kernel_size=1),
        )

    def forward(self, x):
        # Force input to 512x512 so that s3 is guaranteed to be 64x64
        if x.shape[-1] != 512 or x.shape[-2] != 512:
            x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
            
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        x = self.down4(s3)
        x = self.mid(x)
        x = self.up3(x, s3)
        return torch.sigmoid(self.head(x))


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
