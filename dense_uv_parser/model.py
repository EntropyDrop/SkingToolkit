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


class MultiViewSemanticFusion(nn.Module):
    """Fuse frozen per-view semantic features into parser bottleneck features."""

    def __init__(
        self,
        raw_feature_dim,
        semantic_channels,
        bottleneck_channels,
        view_classes,
        attention_heads=4,
        layers=1,
        dropout=0.05,
    ):
        super().__init__()
        if raw_feature_dim < 1 or semantic_channels < 1:
            raise ValueError("Semantic feature dimensions must be positive.")
        if view_classes < 1:
            raise ValueError("Semantic fusion requires view-conditioned parser inputs.")
        if attention_heads < 1:
            raise ValueError("semantic_attention_heads must be positive.")
        if semantic_channels % attention_heads != 0:
            raise ValueError("semantic_channels must be divisible by attention_heads.")
        if layers < 1:
            raise ValueError("semantic_layers must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("semantic_dropout must be in [0, 1).")
        self.raw_feature_dim = int(raw_feature_dim)
        self.semantic_channels = int(semantic_channels)
        self.view_classes = int(view_classes)
        self.input_projection = nn.Sequential(
            nn.LayerNorm(raw_feature_dim),
            nn.Linear(raw_feature_dim, semantic_channels),
            nn.GELU(),
        )
        self.view_embedding = nn.Parameter(
            torch.randn(view_classes, semantic_channels) * 0.02
        )
        self.encoder = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=semantic_channels,
                nhead=attention_heads,
                dim_feedforward=semantic_channels * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(layers)
        )
        self.modulation = nn.Sequential(
            nn.LayerNorm(semantic_channels * 2),
            nn.Linear(semantic_channels * 2, semantic_channels * 2),
            nn.GELU(),
            nn.Linear(semantic_channels * 2, bottleneck_channels * 2),
        )
        self.summary = nn.Sequential(
            nn.LayerNorm(semantic_channels),
            nn.Linear(semantic_channels, semantic_channels),
            nn.GELU(),
        )
        # Preserve the geometry-only initialization. The adapter starts by
        # contributing no FiLM shift/scale and learns semantic corrections.
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, raw_features, view_ids, sample_count):
        if raw_features.dim() == 3:
            if raw_features.shape[1] != self.view_classes:
                raise ValueError(
                    f"Expected {self.view_classes} semantic views, got {raw_features.shape[1]}."
                )
            raw_features = raw_features.reshape(-1, raw_features.shape[-1])
        if raw_features.dim() != 2 or raw_features.shape != (
            sample_count,
            self.raw_feature_dim,
        ):
            raise ValueError(
                "Semantic features must be shaped NxD or BxVxD; got "
                f"{tuple(raw_features.shape)}."
            )
        if sample_count % self.view_classes != 0:
            raise ValueError(
                f"Semantic sample count {sample_count} is not divisible by "
                f"view count {self.view_classes}."
            )
        grouped_view_ids = view_ids.reshape(-1, self.view_classes)
        expected_ids = torch.arange(
            self.view_classes, device=view_ids.device
        ).view(1, -1)
        if not torch.equal(grouped_view_ids, expected_ids.expand_as(grouped_view_ids)):
            raise ValueError("Semantic fusion requires canonical grouped view order.")

        batch = sample_count // self.view_classes
        tokens = self.input_projection(raw_features.float()).reshape(
            batch, self.view_classes, self.semantic_channels
        )
        tokens = tokens + self.view_embedding.unsqueeze(0)
        for layer in self.encoder:
            tokens = layer(tokens)
        pooled = tokens.mean(dim=1)
        per_view = torch.cat(
            [tokens, pooled.unsqueeze(1).expand_as(tokens)], dim=-1
        ).reshape(sample_count, self.semantic_channels * 2)
        return self.modulation(per_view), self.summary(pooled)


class DenseUVParserNet(nn.Module):
    """Predict dense Minecraft UV routing for each render pixel."""

    def __init__(
        self,
        input_channels=4,
        base_channels=32,
        part_classes=6,
        face_classes=6,
        layer_classes=None,
        layer_face_classes=12,
        uv_size=64,
        uv_classification=True,
        view_classes=0,
        predict_affine=False,
        affine_translation_scale=0.0,
        affine_scale_range=0.0,
        surface_classes=0,
        geometry_only=False,
        feature_dropout=0.0,
        semantic_feature_dim=0,
        semantic_channels=128,
        semantic_attention_heads=4,
        semantic_layers=1,
        semantic_dropout=0.05,
        predict_confidence=False,
        route_role_spatial_prior=False,
        route_prior_height=32,
        route_prior_width=16,
        route_prior_logit_cap=1.5,
        route_prior_dropout=0.10,
    ):
        super().__init__()
        self.geometry_only = bool(geometry_only)
        if layer_classes is None:
            layer_classes = 3 if self.geometry_only else 2
        self.layer_classes = int(layer_classes)
        self.uv_classification = bool(uv_classification) and not self.geometry_only
        self.view_classes = int(view_classes)
        self.predict_affine = bool(predict_affine)
        self.affine_translation_scale = float(affine_translation_scale)
        self.affine_scale_range = float(affine_scale_range)
        self.surface_classes = int(surface_classes)
        self.semantic_feature_dim = int(semantic_feature_dim)
        self.semantic_channels = int(semantic_channels)
        self.semantic_attention_heads = int(semantic_attention_heads)
        self.semantic_layers = int(semantic_layers)
        self.semantic_dropout = float(semantic_dropout)
        self.predict_confidence = bool(predict_confidence)
        self.route_role_spatial_prior = bool(route_role_spatial_prior)
        self.route_prior_height = int(route_prior_height)
        self.route_prior_width = int(route_prior_width)
        self.route_prior_logit_cap = float(route_prior_logit_cap)
        self.route_prior_dropout = float(route_prior_dropout)
        if self.route_prior_height < 1 or self.route_prior_width < 1:
            raise ValueError("Route-prior dimensions must be positive.")
        if self.route_prior_logit_cap <= 0.0:
            raise ValueError("route_prior_logit_cap must be positive.")
        if not 0.0 <= self.route_prior_dropout < 1.0:
            raise ValueError("route_prior_dropout must be in [0, 1).")
        if self.route_role_spatial_prior and (
            not self.geometry_only or self.view_classes < 1
        ):
            raise ValueError(
                "The fixed-view route-role prior requires geometry_only with view classes."
            )
        self.feature_dropout_probability = float(feature_dropout)
        if not 0.0 <= self.feature_dropout_probability < 1.0:
            raise ValueError("feature_dropout must be in [0, 1).")
        c = base_channels
        self.stem = ConvBlock(input_channels + self.view_classes, c)
        self.down1 = DownBlock(c, c * 2)
        self.down2 = DownBlock(c * 2, c * 4)
        self.down3 = DownBlock(c * 4, c * 8)
        self.mid = ConvBlock(c * 8, c * 8)
        self.semantic_fusion = (
            MultiViewSemanticFusion(
                self.semantic_feature_dim,
                self.semantic_channels,
                c * 8,
                self.view_classes,
                attention_heads=self.semantic_attention_heads,
                layers=self.semantic_layers,
                dropout=self.semantic_dropout,
            )
            if self.semantic_feature_dim > 0
            else None
        )
        self.up2 = UpBlock(c * 8, c * 4, c * 4)
        self.up1 = UpBlock(c * 4, c * 2, c * 2)
        self.up0 = UpBlock(c * 2, c, c)
        self.features = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.feature_dropout = nn.Dropout2d(self.feature_dropout_probability)
        self.foreground = nn.Conv2d(c, 1, kernel_size=1)
        self.layer = nn.Conv2d(c, self.layer_classes, kernel_size=1)
        self.route_role_prior = (
            nn.Parameter(
                torch.zeros(
                    self.view_classes,
                    self.layer_classes,
                    self.route_prior_height,
                    self.route_prior_width,
                )
            )
            if self.route_role_spatial_prior
            else None
        )
        self.route_confidence = (
            nn.Conv2d(c, 1, kernel_size=1) if self.predict_confidence else None
        )
        if self.semantic_fusion is not None:
            self.outer_presence_head = nn.Linear(self.semantic_channels, 6)
            self.outer_coverage_head = nn.Linear(self.semantic_channels, 6)
        if not self.geometry_only:
            self.part = nn.Conv2d(c, part_classes, kernel_size=1)
            self.face = nn.Conv2d(c, face_classes, kernel_size=1)
            self.layer_face = (
                nn.Conv2d(c, layer_face_classes, kernel_size=1)
                if layer_face_classes > 0
                else None
            )
            self.uv = nn.Conv2d(c, 2, kernel_size=1)
            if self.uv_classification:
                self.uv_x = nn.Conv2d(c, uv_size, kernel_size=1)
                self.uv_y = nn.Conv2d(c, uv_size, kernel_size=1)
        else:
            self.layer_face = None
        if self.predict_affine:
            if not self.geometry_only and self.surface_classes < 2:
                raise ValueError("Global-affine routing requires at least two static surface classes.")
            if self.surface_classes == 1:
                raise ValueError("Surface routing requires at least two static surface classes.")
            if self.surface_classes > 0:
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

    def _runtime_semantic_features(self, images):
        backbone = getattr(self, "_runtime_semantic_backbone", None)
        if backbone is None:
            raise ValueError(
                "This parser requires semantic_features or an attached SigLIP2 runtime backbone."
            )
        with torch.no_grad():
            return backbone.encode_global(images[:, :3])["raw_global"]

    def forward(self, x, view_ids=None, semantic_features=None):
        source_images = x
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
        semantic_summary = None
        if self.semantic_fusion is not None:
            if semantic_features is None:
                semantic_features = self._runtime_semantic_features(source_images)
            modulation, semantic_summary = self.semantic_fusion(
                semantic_features,
                view_ids,
                source_images.shape[0],
            )
            scale, shift = modulation.chunk(2, dim=1)
            x = x * (1.0 + scale.unsqueeze(-1).unsqueeze(-1))
            x = x + shift.unsqueeze(-1).unsqueeze(-1)
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
        x = self.feature_dropout(x)
        layer_evidence = self.layer(x)
        outputs = {
            "foreground": self.foreground(x),
            "layer": layer_evidence,
        }
        if self.route_role_prior is not None:
            selected_prior_raw = self.route_role_prior.index_select(
                0, view_ids.long()
            )
            selected_prior = self.route_prior_logit_cap * torch.tanh(
                selected_prior_raw / self.route_prior_logit_cap
            )
            if self.training and self.route_prior_dropout > 0.0:
                keep = (
                    torch.rand(
                        selected_prior.shape[0],
                        1,
                        1,
                        1,
                        device=selected_prior.device,
                    )
                    >= self.route_prior_dropout
                )
                selected_prior = selected_prior * keep.to(
                    dtype=selected_prior.dtype
                )
            route_prior = F.interpolate(
                selected_prior,
                size=layer_evidence.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).to(dtype=layer_evidence.dtype)
            outputs["layer"] = layer_evidence + route_prior
            outputs["route_role_evidence"] = layer_evidence
            outputs["route_role_prior"] = route_prior
            outputs["route_role_prior_raw"] = self.route_role_prior
        if self.route_confidence is not None:
            outputs["route_confidence"] = self.route_confidence(x)
        if semantic_summary is not None:
            outputs["outer_presence_logits"] = self.outer_presence_head(
                semantic_summary
            )
            outputs["outer_coverage"] = torch.sigmoid(
                self.outer_coverage_head(semantic_summary)
            )
        if not self.geometry_only:
            outputs["part"] = self.part(x)
            outputs["face"] = self.face(x)
            outputs["uv"] = torch.sigmoid(self.uv(x))
            if self.layer_face is not None:
                outputs["layer_face"] = self.layer_face(x)
            if self.uv_classification:
                outputs["uv_x"] = self.uv_x(x)
                outputs["uv_y"] = self.uv_y(x)
        if affine is not None:
            # [tx, ty, log_scale]. tx/ty are affine_grid normalized coordinates.
            outputs["affine"] = affine
            if self.surface_classes > 0:
                outputs["surface"] = self.surface(x)
        return outputs


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
