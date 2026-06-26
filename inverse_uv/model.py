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

    def forward(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.block(x)


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
        
        # Native 64x64 Decoder (no skip connections due to unaligned domains)
        self.up_dec2 = UpBlock(c * 8, 0, c * 4) # 16x16 -> 32x32
        self.up_dec1 = UpBlock(c * 4, 0, c * 2) # 32x32 -> 64x64
        self.head = nn.Sequential(
            nn.Conv2d(c * 2, c, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, output_channels, kernel_size=1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.mid(x)
        
        # Force bottleneck to 16x16 (decouples render_size from 64x64 output)
        x = F.adaptive_avg_pool2d(x, (16, 16))
        
        x = self.up_dec2(x) # -> 32x32
        x = self.up_dec1(x) # -> 64x64
        
        return torch.sigmoid(self.head(x))


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
