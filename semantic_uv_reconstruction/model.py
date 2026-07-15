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


class SpatialSelfAttention(nn.Module):
    """Lightweight spatial self-attention with channel reduction for efficiency."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced = max(channels // reduction, 16)
        self.reduced = reduced
        self.norm = nn.GroupNorm(norm_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, reduced * 3, kernel_size=1)
        self.proj = nn.Conv2d(reduced, channels, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x)).reshape(B, 3, self.reduced, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # (B, R, N)
        scale = self.reduced ** -0.5
        attn = (q.transpose(-2, -1) @ k) * scale  # (B, N, N)
        attn = attn.softmax(dim=-1)
        out = attn @ v.transpose(-2, -1)  # (B, N, R)
        out = out.transpose(-2, -1).reshape(B, self.reduced, H, W)
        return x + self.proj(out)


class UVInpaintingNet(nn.Module):
    def __init__(
        self,
        input_channels=10,
        base_channels=64,
        output_channels=4,
        preserve_known=True,
        **kwargs,
    ):
        super().__init__()
        self.preserve_known = bool(preserve_known)
        c = base_channels
        self.stem = ConvBlock(input_channels, c)
        self.down1 = DownBlock(c, c * 2)
        self.down2 = DownBlock(c * 2, c * 4)
        self.down3 = DownBlock(c * 4, c * 8)
        self.mid = ConvBlock(c * 8, c * 8)
        # Multi-scale attention for long-range dependencies (symmetry, blind spots)
        self.attn32 = SpatialSelfAttention(c * 2, reduction=4)  # 32×32, 1024 tokens
        self.attn16 = SpatialSelfAttention(c * 4, reduction=4)  # 16×16, 256 tokens
        self.up2 = UpBlock(c * 8, c * 4, c * 4)
        self.up1 = UpBlock(c * 4, c * 2, c * 2)
        self.up0 = UpBlock(c * 2, c, c)
        self.head = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, output_channels, kernel_size=1),
        )

    def forward(self, x):
        if x.shape[-1] != 64 or x.shape[-2] != 64:
            x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)

        conditioning = x
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s1 = self.attn32(s1)   # long-range at 32×32 (symmetry, blind-spot structure)
        s2 = self.down2(s1)
        s2 = self.attn16(s2)   # long-range at 16×16 (part-level coherence)
        s3 = self.down3(s2)
        x = self.mid(s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.up0(x, s0)
        pred = torch.sigmoid(self.head(x))
        if self.preserve_known and conditioning.shape[1] == 10:
            inner_rgba = conditioning[:, 0:4]
            inner_known = conditioning[:, 4:5].clamp(0.0, 1.0)
            outer_rgba = conditioning[:, 5:9]
            outer_known = conditioning[:, 9:10].clamp(0.0, 1.0)
            known_sum = inner_known + outer_known
            known = known_sum.clamp(0.0, 1.0)
            observed = (inner_rgba * inner_known + outer_rgba * outer_known) / known_sum.clamp_min(1.0)
            pred = pred * (1.0 - known) + observed * known
        return pred


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator for 64×64 RGBA UV skins.

    Outputs an N×N patch of real/fake logits instead of a single scalar,
    encouraging local texture realism (sharp edges, blocky pixel-art style).
    """

    def __init__(self, input_channels=4, base_channels=64):
        super().__init__()
        c = base_channels
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, c, kernel_size=4, stride=2, padding=1),  # 64→32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1),            # 32→16
            nn.GroupNorm(norm_groups(c * 2), c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1),        # 16→8
            nn.GroupNorm(norm_groups(c * 4), c * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 4, c * 8, kernel_size=4, stride=2, padding=1),        # 8→4
            nn.GroupNorm(norm_groups(c * 8), c * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 8, 1, kernel_size=4, stride=1, padding=1),            # 4×4 patch
        )

    def forward(self, x):
        return self.model(x)


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


# Backward compatibility alias
LightUVInpaintingNet = UVInpaintingNet
