"""Topology-aware masked generator for Minecraft skin UV completion."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from SkingToolkit.semantic_uv_reconstruction.topology import (
    FACE_COUNT,
    INVALID_SURFACE,
    LAYER_COUNT,
    PART_COUNT,
    SURFACE_COUNT,
    UV_SIZE,
    build_uv_topology,
)


RGB_LEVELS = 256


class SurfaceGraphBlock(nn.Module):
    """Linear-complexity texel message passing plus global surface attention."""

    def __init__(self, channels, heads, dropout, neighbours, neighbour_valid, surface_pool, surface):
        super().__init__()
        self.channels = int(channels)
        self.register_buffer("neighbours", neighbours.long(), persistent=False)
        self.register_buffer("neighbour_valid", neighbour_valid.bool(), persistent=False)
        self.register_buffer("surface_pool", surface_pool.float(), persistent=False)
        self.register_buffer("surface", surface.long(), persistent=False)

        self.graph_norm = nn.LayerNorm(channels)
        self.query = nn.Linear(channels, channels, bias=False)
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.graph_projection = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Dropout(dropout),
        )

        self.surface_norm = nn.LayerNorm(channels)
        self.surface_embedding = nn.Parameter(torch.randn(SURFACE_COUNT, channels) * 0.02)
        self.surface_transformer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=heads,
            dim_feedforward=channels * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.surface_projection = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Dropout(dropout),
        )

        self.ff_norm = nn.LayerNorm(channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout),
        )

    def forward(self, tokens, valid):
        normalized = self.graph_norm(tokens)
        neighbour_tokens = normalized[:, self.neighbours]
        query = self.query(normalized).unsqueeze(2)
        key = self.key(neighbour_tokens)
        logits = (query * key).sum(dim=-1) / math.sqrt(self.channels)
        neighbour_valid = self.neighbour_valid.unsqueeze(0)
        logits = logits.masked_fill(~neighbour_valid, -1e4)
        weights = logits.softmax(dim=-1) * neighbour_valid.to(dtype=logits.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        aggregated = (weights.unsqueeze(-1) * self.value(neighbour_tokens)).sum(dim=2)
        tokens = tokens + self.graph_projection(aggregated)

        normalized = self.surface_norm(tokens) * valid
        surface_tokens = torch.einsum("sn,bnc->bsc", self.surface_pool, normalized)
        surface_tokens = surface_tokens + self.surface_embedding.unsqueeze(0)
        surface_tokens = self.surface_transformer(surface_tokens)
        padded_surface_tokens = torch.cat(
            [surface_tokens, surface_tokens.new_zeros(surface_tokens.shape[0], 1, surface_tokens.shape[2])],
            dim=1,
        )
        broadcast = padded_surface_tokens[:, self.surface]
        tokens = tokens + self.surface_projection(broadcast)
        tokens = tokens + self.feed_forward(self.ff_norm(tokens))
        return tokens * valid


class TopologyAwareUVCompletionNet(nn.Module):
    """Generate only missing texels while respecting Minecraft cuboid topology.

    New checkpoints use 12-channel confidence-aware parser conditioning:
    ``inner RGBA + evidence + confidence`` followed by the same outer fields.
    High-confidence evidence is copied exactly, while lower-confidence evidence
    remains context that the generator may repair. Legacy 10-channel input keeps
    its original hard-known behavior. Unknown texels are categorical RGB plus
    binary-alpha distributions, enabling deterministic or stochastic generation.
    """

    model_type = "topology_maskgit"

    def __init__(
        self,
        input_channels=10,
        hidden_channels=128,
        layers=4,
        attention_heads=4,
        dropout=0.05,
        preserve_known=True,
        hard_lock_threshold=0.85,
    ):
        super().__init__()
        if input_channels not in (10, 12):
            raise ValueError(
                "TopologyAwareUVCompletionNet requires 10- or 12-channel conditioning."
            )
        if hidden_channels % attention_heads != 0:
            raise ValueError("hidden_channels must be divisible by attention_heads.")
        if layers < 1:
            raise ValueError("layers must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if not preserve_known:
            raise ValueError(
                "Topology-aware completion requires preserve_known=True so "
                "geometry-routed texels cannot be regenerated."
            )

        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.layers = int(layers)
        self.attention_heads = int(attention_heads)
        self.dropout = float(dropout)
        self.preserve_known = bool(preserve_known)
        self.hard_lock_threshold = float(hard_lock_threshold)
        if not 0.0 <= self.hard_lock_threshold <= 1.0:
            raise ValueError("hard_lock_threshold must be in [0, 1].")
        self.outer_offset = 6 if self.input_channels == 12 else 5
        self.inner_confidence_channel = 5 if self.input_channels == 12 else 4
        self.outer_known_channel = self.outer_offset + 4
        self.outer_confidence_channel = (
            self.outer_offset + 5
            if self.input_channels == 12
            else self.outer_known_channel
        )

        topology = build_uv_topology()
        self.register_buffer("valid_mask", topology.valid.unsqueeze(0).unsqueeze(0), persistent=True)
        self.register_buffer("layer_map", topology.layer.reshape(-1), persistent=True)
        self.register_buffer("part_map", topology.part.reshape(-1), persistent=True)
        self.register_buffer("face_map", topology.face.reshape(-1), persistent=True)
        self.register_buffer("surface_map", topology.surface.reshape(-1), persistent=True)
        self.register_buffer("local_uv", topology.local_uv.reshape(-1, 2), persistent=True)
        self.register_buffer(
            "paired_layer_texel", topology.paired_layer_texel.reshape(-1), persistent=True
        )

        self.rgba_projection = nn.Linear(4, hidden_channels)
        self.known_embedding = nn.Embedding(2, hidden_channels)
        self.confidence_projection = (
            nn.Linear(1, hidden_channels) if self.input_channels == 12 else None
        )
        self.layer_embedding = nn.Embedding(LAYER_COUNT + 1, hidden_channels)
        self.part_embedding = nn.Embedding(PART_COUNT + 1, hidden_channels)
        self.face_embedding = nn.Embedding(FACE_COUNT + 1, hidden_channels)
        self.coordinate_projection = nn.Sequential(
            nn.Linear(6, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.input_norm = nn.LayerNorm(hidden_channels)

        valid_flat = topology.valid.reshape(-1, 1).float()
        # The fifth graph edge connects the exact same texel across the inner
        # and outer layers.  It lets the generator distinguish a painted base
        # surface from a transparent/opaque overlay without relying on distant
        # atlas positions or surface-token pooling alone.
        graph_neighbours = torch.cat(
            [
                topology.neighbours,
                topology.paired_layer_texel.reshape(-1, 1),
            ],
            dim=1,
        )
        graph_neighbour_valid = torch.cat(
            [
                topology.neighbour_valid,
                topology.valid.reshape(-1, 1),
            ],
            dim=1,
        )
        self.blocks = nn.ModuleList(
            SurfaceGraphBlock(
                hidden_channels,
                attention_heads,
                dropout,
                graph_neighbours,
                graph_neighbour_valid,
                topology.surface_pool,
                topology.surface.reshape(-1),
            )
            for _ in range(layers)
        )
        self.register_buffer("valid_tokens", valid_flat, persistent=False)
        self.output_norm = nn.LayerNorm(hidden_channels)
        self.rgb_head = nn.Linear(hidden_channels, 3 * RGB_LEVELS)
        self.alpha_head = nn.Linear(hidden_channels, 1)

    def checkpoint_config(self):
        return {
            "model_type": self.model_type,
            "input_channels": self.input_channels,
            "hidden_channels": self.hidden_channels,
            "layers": self.layers,
            "attention_heads": self.attention_heads,
            "dropout": self.dropout,
            "preserve_known": self.preserve_known,
            "hard_lock_threshold": self.hard_lock_threshold,
            "arm_model": "steve",
            "rgb_levels": RGB_LEVELS,
        }

    def _merged_conditioning(self, conditioning):
        if conditioning.dim() != 4 or conditioning.shape[1:] != (
            self.input_channels,
            UV_SIZE,
            UV_SIZE,
        ):
            raise ValueError(
                f"Expected Bx{self.input_channels}x64x64 conditioning, "
                f"got {tuple(conditioning.shape)}."
            )
        inner_rgba = conditioning[:, 0:4].flatten(2).transpose(1, 2)
        inner_evidence = conditioning[:, 4:5].flatten(2).transpose(1, 2)
        inner_confidence = conditioning[
            :, self.inner_confidence_channel : self.inner_confidence_channel + 1
        ].flatten(2).transpose(1, 2)
        outer_rgba = conditioning[
            :, self.outer_offset : self.outer_offset + 4
        ].flatten(2).transpose(1, 2)
        outer_evidence = conditioning[
            :, self.outer_known_channel : self.outer_known_channel + 1
        ].flatten(2).transpose(1, 2)
        outer_confidence = conditioning[
            :, self.outer_confidence_channel : self.outer_confidence_channel + 1
        ].flatten(2).transpose(1, 2)
        is_inner = (self.layer_map == 0).view(1, -1, 1)
        observed = torch.where(is_inner, inner_rgba, outer_rgba)
        evidence = torch.where(
            is_inner, inner_evidence, outer_evidence
        ).clamp(0.0, 1.0)
        confidence = torch.where(
            is_inner, inner_confidence, outer_confidence
        ).clamp(0.0, 1.0)
        if self.input_channels == 12:
            known = evidence * (
                confidence >= self.hard_lock_threshold
            ).to(dtype=evidence.dtype)
        else:
            known = evidence
        valid = self.valid_tokens.view(1, -1, 1).to(dtype=conditioning.dtype).expand(
            conditioning.shape[0], -1, -1
        )
        return (
            observed * valid,
            known * valid,
            valid,
            evidence * valid,
            confidence * valid,
        )

    def _encode(self, conditioning):
        observed, known, valid, evidence, confidence = self._merged_conditioning(
            conditioning
        )
        local_uv = self.local_uv
        coordinate_features = torch.stack(
            [
                local_uv[:, 0],
                local_uv[:, 1],
                torch.sin(2.0 * math.pi * local_uv[:, 0]),
                torch.cos(2.0 * math.pi * local_uv[:, 0]),
                torch.sin(2.0 * math.pi * local_uv[:, 1]),
                torch.cos(2.0 * math.pi * local_uv[:, 1]),
            ],
            dim=-1,
        )
        tokens = self.rgba_projection(observed)
        tokens = tokens + self.known_embedding((evidence[..., 0] > 0.5).long())
        if self.confidence_projection is not None:
            tokens = tokens + self.confidence_projection(confidence)
        tokens = tokens + self.layer_embedding(self.layer_map).unsqueeze(0)
        tokens = tokens + self.part_embedding(self.part_map).unsqueeze(0)
        tokens = tokens + self.face_embedding(self.face_map).unsqueeze(0)
        tokens = tokens + self.coordinate_projection(coordinate_features).unsqueeze(0)
        tokens = self.input_norm(tokens) * valid
        for block in self.blocks:
            tokens = block(tokens, valid)
        return (
            self.output_norm(tokens) * valid,
            observed,
            known,
            valid,
            evidence,
            confidence,
        )

    def predict_distributions(self, conditioning):
        tokens, observed, known, valid, evidence, confidence = self._encode(
            conditioning
        )
        batch = tokens.shape[0]
        rgb_logits = self.rgb_head(tokens).reshape(batch, UV_SIZE * UV_SIZE, 3, RGB_LEVELS)
        alpha_logits = self.alpha_head(tokens).squeeze(-1)
        return {
            "rgb_logits": rgb_logits,
            "alpha_logits": alpha_logits,
            "observed": observed,
            "known": known,
            "valid": valid,
            "evidence": evidence,
            "confidence": confidence,
        }

    def _distribution_uv(self, outputs):
        values = torch.linspace(
            0.0, 1.0, RGB_LEVELS, device=outputs["rgb_logits"].device, dtype=torch.float32
        )
        rgb = (outputs["rgb_logits"].float().softmax(dim=-1) * values).sum(dim=-1)
        alpha_probability = torch.sigmoid(outputs["alpha_logits"].float()).unsqueeze(-1)
        is_inner = (self.layer_map == 0).view(1, -1, 1)
        alpha = torch.where(is_inner, torch.ones_like(alpha_probability), alpha_probability)
        predicted = torch.cat([rgb, alpha], dim=-1) * outputs["valid"].float()
        predicted = predicted * (1.0 - outputs["known"].float()) + outputs[
            "observed"
        ].float() * outputs["known"].float()
        return predicted.transpose(1, 2).reshape(-1, 4, UV_SIZE, UV_SIZE)

    def forward(self, conditioning, return_logits=False):
        outputs = self.predict_distributions(conditioning)
        outputs["uv"] = self._distribution_uv(outputs)
        return outputs if return_logits else outputs["uv"]

    def augment_training_conditioning(
        self,
        conditioning,
        target_uv,
        drop_known_min=0.1,
        drop_known_max=0.5,
        teacher_reveal_unknown=0.1,
    ):
        """Create MaskGIT-style masks while retaining the parser's geometry contract."""
        if not 0.0 <= drop_known_min <= drop_known_max <= 1.0:
            raise ValueError("Known-drop ratios must satisfy 0 <= min <= max <= 1.")
        if not 0.0 <= teacher_reveal_unknown <= 1.0:
            raise ValueError("teacher_reveal_unknown must be in [0, 1].")
        augmented = conditioning.clone()
        batch = conditioning.shape[0]
        flat = augmented.flatten(2)
        target = target_uv.flatten(2)
        inner_position = (self.layer_map == 0).view(1, -1)
        outer_position = (self.layer_map == 1).view(1, -1)
        valid = self.valid_tokens[:, 0].bool().view(1, -1)
        evidence = torch.where(
            inner_position,
            flat[:, 4],
            flat[:, self.outer_known_channel],
        ) > 0.5

        ratios = torch.empty(batch, 1, device=conditioning.device).uniform_(
            float(drop_known_min), float(drop_known_max)
        )
        drop = evidence & valid & (
            torch.rand(batch, UV_SIZE * UV_SIZE, device=conditioning.device) < ratios
        )
        reveal = (
            ~evidence
            & valid
            & (
                torch.rand(batch, UV_SIZE * UV_SIZE, device=conditioning.device)
                < float(teacher_reveal_unknown)
            )
        )

        keep_inner = ~(drop & inner_position)
        keep_outer = ~(drop & outer_position)
        flat[:, 0:4] = flat[:, 0:4] * keep_inner.unsqueeze(1)
        flat[:, 4] = flat[:, 4] * keep_inner
        flat[:, self.outer_offset : self.outer_offset + 4] = (
            flat[:, self.outer_offset : self.outer_offset + 4]
            * keep_outer.unsqueeze(1)
        )
        flat[:, self.outer_known_channel] = (
            flat[:, self.outer_known_channel] * keep_outer
        )
        if self.input_channels == 12:
            flat[:, self.inner_confidence_channel] = (
                flat[:, self.inner_confidence_channel] * keep_inner
            )
            flat[:, self.outer_confidence_channel] = (
                flat[:, self.outer_confidence_channel] * keep_outer
            )

        reveal_inner = reveal & inner_position
        reveal_outer = reveal & outer_position
        flat[:, 0:4] = torch.where(reveal_inner.unsqueeze(1), target, flat[:, 0:4])
        flat[:, 4] = torch.where(reveal_inner, torch.ones_like(flat[:, 4]), flat[:, 4])
        flat[:, self.outer_offset : self.outer_offset + 4] = torch.where(
            reveal_outer.unsqueeze(1),
            target,
            flat[:, self.outer_offset : self.outer_offset + 4],
        )
        flat[:, self.outer_known_channel] = torch.where(
            reveal_outer,
            torch.ones_like(flat[:, self.outer_known_channel]),
            flat[:, self.outer_known_channel],
        )
        if self.input_channels == 12:
            flat[:, self.inner_confidence_channel] = torch.where(
                reveal_inner,
                torch.ones_like(flat[:, self.inner_confidence_channel]),
                flat[:, self.inner_confidence_channel],
            )
            flat[:, self.outer_confidence_channel] = torch.where(
                reveal_outer,
                torch.ones_like(flat[:, self.outer_confidence_channel]),
                flat[:, self.outer_confidence_channel],
            )
        return augmented

    def masked_token_loss(
        self,
        outputs,
        target_uv,
        lambda_rgb_token=1.0,
        lambda_rgb_distribution=2.0,
        lambda_alpha_token=0.5,
        ignore_covered_inner=True,
        covered_inner_alpha_threshold=0.1,
    ):
        target = target_uv.float().flatten(2).transpose(1, 2)
        target_rgb = (target[..., :3].clamp(0.0, 1.0) * 255.0).round().long()
        target_alpha = target[..., 3]
        unknown = outputs["known"][..., 0] <= 0.5
        valid = outputs["valid"][..., 0] > 0.5
        is_inner = (self.layer_map == 0).view(1, -1)
        is_outer = (self.layer_map == 1).view(1, -1)

        rgb_mask = unknown & valid & (target_alpha > 0.5)
        if ignore_covered_inner:
            paired_alpha = target_alpha[:, self.paired_layer_texel]
            covered_inner = is_inner & (paired_alpha > float(covered_inner_alpha_threshold))
            rgb_mask = rgb_mask & ~covered_inner

        rgb_loss = F.cross_entropy(
            outputs["rgb_logits"].float().reshape(-1, RGB_LEVELS),
            target_rgb.reshape(-1),
            reduction="none",
        ).reshape(target.shape[0], UV_SIZE * UV_SIZE, 3)
        rgb_weight = rgb_mask.unsqueeze(-1).to(dtype=rgb_loss.dtype)
        loss_rgb_token = (rgb_loss * rgb_weight).sum() / (rgb_weight.sum() * 3.0).clamp_min(1.0)

        # Cross entropy treats every wrong 8-bit bin equally. Add an ordinal
        # distribution penalty so probability mass at an extreme (0 or 255) is
        # much more expensive when the target is a moderate color. This aligns
        # categorical MaskGIT training with deterministic distribution-mean
        # decoding and suppresses high-saturation channel modes.
        rgb_probabilities = outputs["rgb_logits"].float().softmax(dim=-1)
        levels = torch.linspace(
            0.0,
            1.0,
            RGB_LEVELS,
            device=rgb_probabilities.device,
            dtype=rgb_probabilities.dtype,
        )
        target_rgb_unit = target_rgb.to(dtype=rgb_probabilities.dtype) / 255.0
        level_values = levels.view(1, 1, 1, -1)
        predicted_mean = (rgb_probabilities * level_values).sum(dim=-1)
        predicted_second_moment = (
            rgb_probabilities * level_values.square()
        ).sum(dim=-1)
        rgb_distribution = (
            predicted_second_moment
            - 2.0 * target_rgb_unit * predicted_mean
            + target_rgb_unit.square()
        ).clamp_min(0.0)
        loss_rgb_distribution = (
            rgb_distribution * rgb_weight
        ).sum() / (rgb_weight.sum() * 3.0).clamp_min(1.0)

        alpha_mask = unknown & is_outer
        alpha_loss = F.binary_cross_entropy_with_logits(
            outputs["alpha_logits"].float(), target_alpha, reduction="none"
        )
        alpha_weight = alpha_mask.to(dtype=alpha_loss.dtype)
        loss_alpha_token = (alpha_loss * alpha_weight).sum() / alpha_weight.sum().clamp_min(1.0)
        loss_token = (
            float(lambda_rgb_token) * loss_rgb_token
            + float(lambda_rgb_distribution) * loss_rgb_distribution
            + float(lambda_alpha_token) * loss_alpha_token
        )

        predicted_rgb = outputs["rgb_logits"].argmax(dim=-1)
        exact_rgb = (predicted_rgb == target_rgb).all(dim=-1)
        rgb_exact = (
            (exact_rgb & rgb_mask).float().sum() / rgb_mask.float().sum().clamp_min(1.0)
        )
        return {
            "loss_token": loss_token,
            "loss_rgb_token": loss_rgb_token,
            "loss_rgb_distribution": loss_rgb_distribution,
            "loss_alpha_token": loss_alpha_token,
            "acc_unknown_rgb_exact": rgb_exact,
            "unknown_texel_fraction": (unknown & valid).float().sum()
            / valid.float().sum().clamp_min(1.0),
        }

    def _write_generated(self, conditioning, selected, rgb, alpha):
        flat = conditioning.flatten(2)
        rgba = torch.cat([rgb.float() / 255.0, alpha.unsqueeze(-1).float()], dim=-1)
        inner = selected & (self.layer_map == 0).view(1, -1)
        outer = selected & (self.layer_map == 1).view(1, -1)
        flat[:, 0:4] = torch.where(inner.unsqueeze(1), rgba.transpose(1, 2), flat[:, 0:4])
        flat[:, 4] = torch.where(inner, torch.ones_like(flat[:, 4]), flat[:, 4])
        flat[:, self.outer_offset : self.outer_offset + 4] = torch.where(
            outer.unsqueeze(1),
            rgba.transpose(1, 2),
            flat[:, self.outer_offset : self.outer_offset + 4],
        )
        flat[:, self.outer_known_channel] = torch.where(
            outer,
            torch.ones_like(flat[:, self.outer_known_channel]),
            flat[:, self.outer_known_channel],
        )
        if self.input_channels == 12:
            flat[:, self.inner_confidence_channel] = torch.where(
                inner,
                torch.ones_like(flat[:, self.inner_confidence_channel]),
                flat[:, self.inner_confidence_channel],
            )
            flat[:, self.outer_confidence_channel] = torch.where(
                outer,
                torch.ones_like(flat[:, self.outer_confidence_channel]),
                flat[:, self.outer_confidence_channel],
            )

    def _snap_generated_rgb_to_evidence_palette(
        self,
        result,
        reference_observed,
        reference_evidence,
        reference_confidence,
        generated,
        min_confidence=0.5,
    ):
        """Project generated RGB onto observed topology-aware color triplets.

        Invisible texels have no evidence for inventing a new character color.
        Generated opaque texels therefore select a complete observed RGB triplet
        on the same part/layer/face. Sparse groups fall back through the same
        part/layer, the same part, and finally all visible evidence. Candidate
        colors are ranked primarily by distance to the model prediction, with a
        small spatial tie-break. This prevents independent channel decoding from
        assembling saturated RGB combinations that never occurred on the skin.
        """
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("palette min_confidence must be in [0, 1].")
        snapped = result.clone()
        valid = self.valid_tokens[:, 0].bool().view(1, -1)
        opaque_reference = reference_observed[..., 3] > 0.5
        strong_reference = (
            (reference_evidence[..., 0] > 0.5)
            & (reference_confidence[..., 0] >= float(min_confidence))
            & opaque_reference
            & valid
        )
        all_reference = (
            (reference_evidence[..., 0] > 0.5)
            & opaque_reference
            & valid
        )
        opaque_generated = generated & (result[..., 3] > 0.5) & valid
        layer = self.layer_map.view(1, -1)
        part = self.part_map.view(1, -1)
        face = self.face_map.view(1, -1)
        coordinates = self.local_uv.float()

        for batch_index in range(result.shape[0]):
            strong = strong_reference[batch_index]
            fallback = all_reference[batch_index]
            if not fallback.any():
                continue
            for part_index in range(PART_COUNT):
                part_mask = part[0] == part_index
                for layer_index in range(LAYER_COUNT):
                    group_target = (
                        opaque_generated[batch_index]
                        & part_mask
                        & (layer[0] == layer_index)
                    )
                    if not group_target.any():
                        continue
                    for face_index in range(FACE_COUNT):
                        target = group_target & (face[0] == face_index)
                        if not target.any():
                            continue
                        masks = (
                            strong & part_mask & (layer[0] == layer_index) & (face[0] == face_index),
                            fallback & part_mask & (layer[0] == layer_index) & (face[0] == face_index),
                            strong & part_mask & (layer[0] == layer_index),
                            fallback & part_mask & (layer[0] == layer_index),
                            strong & part_mask,
                            fallback & part_mask,
                            strong,
                            fallback,
                        )
                        local_reference = next(
                            (candidate for candidate in masks if candidate.any()),
                            None,
                        )
                        if local_reference is None:
                            continue
                        target_indices = target.nonzero(as_tuple=False).flatten()
                        reference_indices = local_reference.nonzero(
                            as_tuple=False
                        ).flatten()
                        predicted_rgb = snapped[
                            batch_index, target_indices, :3
                        ].float()
                        reference_rgb = reference_observed[
                            batch_index, reference_indices, :3
                        ].float()
                        color_distance = torch.cdist(
                            predicted_rgb,
                            reference_rgb,
                        )
                        spatial_distance = torch.cdist(
                            coordinates[target_indices],
                            coordinates[reference_indices],
                        )
                        nearest = (
                            color_distance + 0.05 * spatial_distance
                        ).argmin(dim=1)
                        source_indices = reference_indices[nearest]
                        snapped[batch_index, target_indices, :3] = (
                            reference_observed[
                                batch_index, source_indices, :3
                            ].to(dtype=snapped.dtype)
                        )
        return snapped

    @torch.no_grad()
    def generate(
        self,
        conditioning,
        steps=4,
        temperature=0.0,
        seed=1234,
        palette_snap=False,
        palette_min_confidence=0.5,
        rgb_decode="mean",
    ):
        if steps < 1:
            raise ValueError("Generation steps must be positive.")
        if temperature < 0.0:
            raise ValueError("temperature must be non-negative.")
        if rgb_decode not in ("mean", "argmax"):
            raise ValueError("rgb_decode must be 'mean' or 'argmax'.")
        working = conditioning.clone()
        (
            original_observed,
            original_known,
            valid,
            original_evidence,
            original_confidence,
        ) = self._merged_conditioning(conditioning)
        initial_unknown = (original_known[..., 0] <= 0.5) & (valid[..., 0] > 0.5)
        total_unknown = initial_unknown.sum(dim=1)
        generator = None
        if conditioning.device.type in ("cpu", "cuda"):
            generator = torch.Generator(device=conditioning.device)
            generator.manual_seed(int(seed))
        else:
            torch.manual_seed(int(seed))

        for step in range(steps):
            outputs = self.predict_distributions(working)
            rgb_probabilities = outputs["rgb_logits"].float().softmax(dim=-1)
            if temperature == 0.0:
                if rgb_decode == "mean":
                    levels = torch.arange(
                        RGB_LEVELS,
                        device=rgb_probabilities.device,
                        dtype=rgb_probabilities.dtype,
                    )
                    rgb = (
                        (rgb_probabilities * levels).sum(dim=-1)
                        .round()
                        .clamp(0, RGB_LEVELS - 1)
                        .long()
                    )
                else:
                    rgb = rgb_probabilities.argmax(dim=-1)
            else:
                adjusted = (outputs["rgb_logits"].float() / temperature).softmax(dim=-1)
                rgb = torch.multinomial(
                    adjusted.reshape(-1, RGB_LEVELS), 1, generator=generator
                ).reshape(conditioning.shape[0], UV_SIZE * UV_SIZE, 3)
            rgb_confidence = rgb_probabilities.amax(dim=-1).mean(dim=-1)

            alpha_probability = torch.sigmoid(outputs["alpha_logits"].float())
            if temperature == 0.0:
                alpha = alpha_probability > 0.5
            else:
                logits = outputs["alpha_logits"].float() / temperature
                alpha = torch.bernoulli(torch.sigmoid(logits), generator=generator).bool()
            alpha = torch.where(
                (self.layer_map == 0).view(1, -1), torch.ones_like(alpha), alpha
            )
            alpha_confidence = torch.maximum(alpha_probability, 1.0 - alpha_probability)
            confidence = rgb_confidence * alpha_confidence

            _, current_known, _, _, _ = self._merged_conditioning(working)
            remaining = initial_unknown & (current_known[..., 0] <= 0.5)
            selected = torch.zeros_like(remaining)
            desired_remaining_fraction = math.cos(
                0.5 * math.pi * float(step + 1) / float(steps)
            )
            for batch_index in range(conditioning.shape[0]):
                remaining_indices = remaining[batch_index].nonzero(as_tuple=False).flatten()
                if remaining_indices.numel() == 0:
                    continue
                desired_remaining = int(
                    round(float(total_unknown[batch_index]) * desired_remaining_fraction)
                )
                reveal_count = max(1, remaining_indices.numel() - desired_remaining)
                reveal_count = min(reveal_count, remaining_indices.numel())
                scores = confidence[batch_index, remaining_indices]
                chosen = remaining_indices[torch.topk(scores, reveal_count).indices]
                selected[batch_index, chosen] = True
            self._write_generated(working, selected, rgb, alpha)

        observed, _, valid, _, _ = self._merged_conditioning(working)
        result = observed * valid
        if palette_snap:
            result = self._snap_generated_rgb_to_evidence_palette(
                result,
                original_observed,
                original_evidence,
                original_confidence,
                initial_unknown,
                min_confidence=palette_min_confidence,
            )
        return result.transpose(1, 2).reshape(-1, 4, UV_SIZE, UV_SIZE)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
