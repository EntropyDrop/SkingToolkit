import math

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


class DenseUVParserNet(nn.Module):
    """Predict dense Minecraft UV routing for each render pixel."""

    def __init__(
        self,
        input_channels=4,
        base_channels=32,
        part_classes=6,
        face_classes=6,
        layer_classes=2,
        layer_face_classes=12,
        uv_size=64,
        uv_classification=True,
        view_classes=0,
        predict_affine=False,
        affine_translation_scale=0.03,
        affine_scale_range=0.03,
        surface_classes=0,
    ):
        super().__init__()
        self.uv_classification = uv_classification
        self.view_classes = int(view_classes)
        self.predict_affine = bool(predict_affine)
        self.affine_translation_scale = float(affine_translation_scale)
        self.affine_scale_range = float(affine_scale_range)
        self.surface_classes = int(surface_classes)
        c = base_channels
        self.stem = ConvBlock(input_channels + self.view_classes, c)
        self.down1 = DownBlock(c, c * 2)
        self.down2 = DownBlock(c * 2, c * 4)
        self.down3 = DownBlock(c * 4, c * 8)
        self.mid = ConvBlock(c * 8, c * 8)
        self.up2 = UpBlock(c * 8, c * 4, c * 4)
        self.up1 = UpBlock(c * 4, c * 2, c * 2)
        self.up0 = UpBlock(c * 2, c, c)
        self.features = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.foreground = nn.Conv2d(c, 1, kernel_size=1)
        self.layer = nn.Conv2d(c, layer_classes, kernel_size=1)
        self.part = nn.Conv2d(c, part_classes, kernel_size=1)
        self.face = nn.Conv2d(c, face_classes, kernel_size=1)
        self.layer_face = (
            nn.Conv2d(c, layer_face_classes, kernel_size=1)
            if layer_face_classes > 0
            else None
        )
        self.uv = nn.Conv2d(c, 2, kernel_size=1)
        if uv_classification:
            self.uv_x = nn.Conv2d(c, uv_size, kernel_size=1)
            self.uv_y = nn.Conv2d(c, uv_size, kernel_size=1)
        if self.predict_affine:
            if self.surface_classes < 2:
                raise ValueError("Global-affine routing requires at least two static surface classes.")
            self.surface = nn.Conv2d(c, self.surface_classes, kernel_size=1)
            hidden = max(c * 4, 32)
            self.affine_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(c * 8, hidden),
                nn.SiLU(inplace=True),
                nn.Linear(hidden, 3),
            )
            # Start from the canonical pose; the residual head learns only the
            # small global transform introduced by the configured augmentation.
            nn.init.zeros_(self.affine_head[-1].weight)
            nn.init.zeros_(self.affine_head[-1].bias)

            self.affine_translation_limit = 2.0 * self.affine_translation_scale
            lower_log_scale = math.log(max(1.0 - self.affine_scale_range, 1e-6))
            upper_log_scale = math.log1p(self.affine_scale_range)
            self.affine_log_scale_limit = max(abs(lower_log_scale), abs(upper_log_scale))

    def forward(self, x, view_ids=None):
        if self.view_classes > 0:
            if view_ids is None:
                raise ValueError("view_ids are required for a view-conditioned dense UV parser.")
            if view_ids.shape != (x.shape[0],):
                raise ValueError(f"Expected view_ids shape {(x.shape[0],)}, got {tuple(view_ids.shape)}.")
            if view_ids.min() < 0 or view_ids.max() >= self.view_classes:
                raise ValueError(f"view_ids must be in [0, {self.view_classes - 1}].")
            view_one_hot = F.one_hot(view_ids.long(), num_classes=self.view_classes).to(dtype=x.dtype)
            view_one_hot = view_one_hot.view(x.shape[0], self.view_classes, 1, 1)
            x = torch.cat([x, view_one_hot.expand(-1, -1, x.shape[2], x.shape[3])], dim=1)

        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        x = self.mid(s3)
        affine = None
        if self.predict_affine:
            raw_affine = torch.tanh(self.affine_head(x))
            affine = torch.stack(
                [
                    raw_affine[:, 0] * self.affine_translation_limit,
                    raw_affine[:, 1] * self.affine_translation_limit,
                    raw_affine[:, 2] * self.affine_log_scale_limit,
                ],
                dim=1,
            )
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.up0(x, s0)
        x = self.features(x)
        outputs = {
            "foreground": self.foreground(x),
            "layer": self.layer(x),
            "part": self.part(x),
            "face": self.face(x),
            "uv": torch.sigmoid(self.uv(x)),
        }
        if self.layer_face is not None:
            outputs["layer_face"] = self.layer_face(x)
        if self.uv_classification:
            outputs["uv_x"] = self.uv_x(x)
            outputs["uv_y"] = self.uv_y(x)
        if affine is not None:
            # [tx, ty, log_scale]. tx/ty are affine_grid normalized coordinates.
            outputs["affine"] = affine
            outputs["surface"] = self.surface(x)
        return outputs


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
