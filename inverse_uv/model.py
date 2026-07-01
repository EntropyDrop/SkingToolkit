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


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(norm_groups(out_channels), out_channels)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(norm_groups(out_channels), out_channels)
        self.act2 = nn.SiLU(inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.GroupNorm(norm_groups(out_channels), out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.act2(self.conv2(self.norm2(self.act1(self.conv1(x)))) + self.shortcut(x))


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_cls=ConvBlock):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.block = block_cls(out_channels, out_channels)

    def forward(self, x):
        return self.block(self.down(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, block_cls=ConvBlock, use_coordconv=False):
        super().__init__()
        self.use_coordconv = use_coordconv
        self.coordconv = CoordConv() if use_coordconv else nn.Identity()
        block_in_channels = in_channels + skip_channels + (2 if use_coordconv else 0)
        self.block = block_cls(block_in_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x_cat = torch.cat([x, skip], dim=1)
        return self.block(self.coordconv(x_cat))


class CoordConv(nn.Module):
    def forward(self, x):
        batch, _, height, width = x.shape
        y = torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype)
        x_coords = torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(y, x_coords, indexing="ij")
        coords = torch.stack((xx, yy), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)
        return torch.cat((x, coords), dim=1)


class SpatialSelfAttention(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        heads = min(heads, channels)
        while channels % heads != 0 and heads > 1:
            heads -= 1
        self.heads = heads
        self.head_dim = channels // heads
        self.scale = self.head_dim ** -0.5
        self.norm = nn.GroupNorm(norm_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        batch, channels, height, width = x.shape
        tokens = height * width
        qkv = self.qkv(self.norm(x)).reshape(
            batch,
            3,
            self.heads,
            self.head_dim,
            tokens,
        )
        q, k, v = qkv.unbind(dim=1)
        q = q.transpose(-2, -1)
        v = v.transpose(-2, -1)
        attention = torch.matmul(q * self.scale, k).softmax(dim=-1)
        out = torch.matmul(attention, v)
        out = out.transpose(-2, -1).reshape(batch, channels, height, width)
        return x + self.proj(out)


class InverseUVNet(nn.Module):
    def __init__(
        self,
        input_channels=10,
        base_channels=64,
        output_channels=4,
        use_coordconv=True,
        use_attention=True,
        attention_heads=4,
        use_resnet=True,
        multi_scale_coord=True,
    ):
        super().__init__()
        c = base_channels
        self.use_coordconv = use_coordconv
        self.use_attention = use_attention
        self.use_resnet = use_resnet
        self.multi_scale_coord = multi_scale_coord
        
        self.coordconv = CoordConv() if use_coordconv else nn.Identity()
        stem_channels = input_channels + (2 if use_coordconv else 0)
        
        block_cls = ResBlock if use_resnet else ConvBlock
        
        self.stem = block_cls(stem_channels, c)
        self.down1 = DownBlock(c, c * 2, block_cls=block_cls)
        self.down2 = DownBlock(c * 2, c * 4, block_cls=block_cls)
        self.down3 = DownBlock(c * 4, c * 8, block_cls=block_cls)
        
        mid_in_channels = c * 8 + (2 if (use_coordconv and multi_scale_coord) else 0)
        self.mid_coordconv = CoordConv() if (use_coordconv and multi_scale_coord) else nn.Identity()
        self.mid = block_cls(mid_in_channels, c * 8)
        self.mid_attention = (
            SpatialSelfAttention(c * 8, heads=attention_heads) if use_attention else nn.Identity()
        )
        
        self.up2 = UpBlock(c * 8, c * 4, c * 4, block_cls=block_cls, use_coordconv=(use_coordconv and multi_scale_coord))
        self.up1 = UpBlock(c * 4, c * 2, c * 2, block_cls=block_cls, use_coordconv=(use_coordconv and multi_scale_coord))
        self.up0 = UpBlock(c * 2, c, c, block_cls=block_cls, use_coordconv=(use_coordconv and multi_scale_coord))
        self.head = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, output_channels, kernel_size=1),
        )

    def forward(self, x):
        if x.shape[-1] != 64 or x.shape[-2] != 64:
            x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)

        x = self.coordconv(x)
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        
        mid_in = self.mid_coordconv(s3)
        x = self.mid(mid_in)
        x = self.mid_attention(x)
        
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.up0(x, s0)
        return torch.sigmoid(self.head(x))


class PixelShuffleUpBlock(nn.Module):
    """Upsampling block using PixelShuffle for sharper outputs vs bilinear."""
    def __init__(self, in_channels, skip_channels, out_channels, block_cls=ConvBlock):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=1),
            nn.PixelShuffle(2),
        )
        self.block = block_cls(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class LightInverseUVNet(nn.Module):
    """Lightweight U-Net for mild-deformation UV inpainting (~1M params).

    Uses only 2 downsampling levels (bottleneck at 16×16) and CoordConv
    only at the stem.  No self-attention.  Designed for scenarios where
    the input conditioning is already a close UV-space approximation and
    the network mainly needs to fill occluded regions and correct minor
    misalignment.
    """

    def __init__(
        self,
        input_channels=10,
        base_channels=32,
        output_channels=4,
        use_coordconv=True,
        use_pixelshuffle=False,
    ):
        super().__init__()
        c = base_channels
        self.use_coordconv = use_coordconv
        self.use_pixelshuffle = use_pixelshuffle

        self.coordconv = CoordConv() if use_coordconv else nn.Identity()
        stem_in = input_channels + (2 if use_coordconv else 0)

        self.stem = ConvBlock(stem_in, c)
        self.down1 = DownBlock(c, c * 2)
        self.down2 = DownBlock(c * 2, c * 4)

        self.mid = ConvBlock(c * 4, c * 4)

        up_cls = PixelShuffleUpBlock if use_pixelshuffle else UpBlock
        self.up1 = up_cls(c * 4, c * 2, c * 2)
        self.up0 = up_cls(c * 2, c, c)
        self.head = nn.Sequential(
            nn.Conv2d(c, output_channels, kernel_size=1),
        )

    def forward(self, x):
        if x.shape[-1] != 64 or x.shape[-2] != 64:
            x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)

        x = self.coordconv(x)
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        x = self.mid(s2)
        x = self.up1(x, s1)
        x = self.up0(x, s0)
        return torch.sigmoid(self.head(x))


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
