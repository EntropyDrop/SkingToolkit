import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_groups(channels):
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_norm_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_norm_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, tensor):
        return self.block(tensor)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Conv2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.block = ConvBlock(out_channels, out_channels)

    def forward(self, tensor):
        return self.block(self.down(tensor))


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.block = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, tensor, skip):
        tensor = F.interpolate(
            tensor, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.block(torch.cat([tensor, skip], dim=1))


class FixedViewForegroundNet(nn.Module):
    """Lightweight coordinate- and view-conditioned foreground U-Net."""

    def __init__(
        self,
        input_channels=3,
        base_channels=24,
        view_classes=2,
        coordinate_channels=True,
        geometry_channels=2,
        dropout=0.05,
    ):
        super().__init__()
        if input_channels < 1 or base_channels < 4 or view_classes < 1:
            raise ValueError("Foreground model channel counts must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.input_channels = int(input_channels)
        self.base_channels = int(base_channels)
        self.view_classes = int(view_classes)
        self.coordinate_channels = bool(coordinate_channels)
        self.geometry_channels = int(geometry_channels)
        self.dropout_probability = float(dropout)
        if self.geometry_channels < 0:
            raise ValueError("geometry_channels must be non-negative.")

        conditioning_channels = self.view_classes
        if self.coordinate_channels:
            conditioning_channels += 2
        c = self.base_channels
        self.stem = ConvBlock(
            self.input_channels + conditioning_channels + self.geometry_channels,
            c,
        )
        self.down1 = DownBlock(c, c * 2)
        self.down2 = DownBlock(c * 2, c * 4)
        self.down3 = DownBlock(c * 4, c * 8)
        self.mid = ConvBlock(c * 8, c * 8)
        self.up2 = UpBlock(c * 8, c * 4, c * 4)
        self.up1 = UpBlock(c * 4, c * 2, c * 2)
        self.up0 = UpBlock(c * 2, c, c)
        self.dropout = nn.Dropout2d(self.dropout_probability)
        self.output = nn.Conv2d(c, 1, kernel_size=1)

    def _conditioning(self, images, view_ids):
        if view_ids is None or view_ids.shape != (images.shape[0],):
            raise ValueError(
                f"Expected view_ids shape {(images.shape[0],)}, got "
                f"{None if view_ids is None else tuple(view_ids.shape)}."
            )
        if view_ids.numel() and (
            view_ids.min().item() < 0 or view_ids.max().item() >= self.view_classes
        ):
            raise ValueError(f"view_ids must be in [0, {self.view_classes - 1}].")
        one_hot = F.one_hot(
            view_ids.long(), num_classes=self.view_classes
        ).to(dtype=images.dtype)
        channels = [
            one_hot.view(images.shape[0], self.view_classes, 1, 1).expand(
                -1, -1, images.shape[2], images.shape[3]
            )
        ]
        if self.coordinate_channels:
            y = torch.linspace(
                -1.0,
                1.0,
                images.shape[2],
                device=images.device,
                dtype=images.dtype,
            ).view(1, 1, images.shape[2], 1)
            x = torch.linspace(
                -1.0,
                1.0,
                images.shape[3],
                device=images.device,
                dtype=images.dtype,
            ).view(1, 1, 1, images.shape[3])
            channels.extend(
                [
                    x.expand(images.shape[0], -1, images.shape[2], -1),
                    y.expand(images.shape[0], -1, -1, images.shape[3]),
                ]
            )
        return torch.cat(channels, dim=1)

    def forward(self, images, view_ids, geometry_prior=None):
        if images.dim() != 4 or images.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected NCHW input with {self.input_channels} channels, got "
                f"{tuple(images.shape)}."
            )
        inputs = [images, self._conditioning(images, view_ids)]
        if self.geometry_channels > 0:
            expected = (
                images.shape[0],
                self.geometry_channels,
                images.shape[2],
                images.shape[3],
            )
            if geometry_prior is None or geometry_prior.shape != expected:
                raise ValueError(
                    f"Expected geometry_prior shape {expected}, got "
                    f"{None if geometry_prior is None else tuple(geometry_prior.shape)}."
                )
            inputs.append(geometry_prior.to(device=images.device, dtype=images.dtype))
        tensor = torch.cat(inputs, dim=1)
        skip0 = self.stem(tensor)
        skip1 = self.down1(skip0)
        skip2 = self.down2(skip1)
        tensor = self.mid(self.down3(skip2))
        tensor = self.up2(tensor, skip2)
        tensor = self.up1(tensor, skip1)
        tensor = self.up0(tensor, skip0)
        return self.output(self.dropout(tensor))


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())
