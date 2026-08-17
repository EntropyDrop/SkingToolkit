import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from SkingToolkit.dense_uv_parser.uv_topology import (
    build_head_outer_face_indices,
    build_outer_uv_graph,
    build_simple_uv_topology,
)


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
            # Training may cache multiple complete view groups per skin, for
            # example the inference pair followed by a privileged right-side
            # pair. Each group is still fused independently below.
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


class SpatialSemanticFusion(nn.Module):
    """Project frozen 2D semantic features into the parser bottleneck.

    The final projection starts at zero, preserving the stable geometry parser
    at initialization while allowing spatial semantic corrections to emerge
    during training.
    """

    def __init__(
        self,
        raw_feature_dim,
        semantic_channels,
        bottleneck_channels,
    ):
        super().__init__()
        if raw_feature_dim < 1 or semantic_channels < 1:
            raise ValueError("Spatial semantic feature dimensions must be positive.")
        self.raw_feature_dim = int(raw_feature_dim)
        self.semantic_channels = int(semantic_channels)
        self.input_norm = nn.LayerNorm(raw_feature_dim)
        self.input_projection = nn.Conv2d(
            raw_feature_dim, semantic_channels, kernel_size=1
        )
        self.activation = nn.GELU()
        self.output_projection = nn.Conv2d(
            semantic_channels, bottleneck_channels, kernel_size=1
        )
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, raw_features, sample_count, output_size):
        if raw_features.dim() == 5:
            raw_features = raw_features.reshape(
                -1,
                raw_features.shape[-3],
                raw_features.shape[-2],
                raw_features.shape[-1],
            )
        if (
            raw_features.dim() != 4
            or raw_features.shape[0] != sample_count
            or raw_features.shape[1] != self.raw_feature_dim
        ):
            raise ValueError(
                "Spatial semantic features must be shaped NxCxHxW or "
                f"BxVxCxHxW; got {tuple(raw_features.shape)}."
            )
        normalized = self.input_norm(
            raw_features.float().permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2)
        residual = self.output_projection(
            self.activation(self.input_projection(normalized))
        )
        return F.interpolate(
            residual,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


class UVMultiViewSpatialFusion(nn.Module):
    """3D Geometry-aware UV cross-view fusion between front and back views.

    Instead of mixing 2D screen coordinates, this module projects intermediate
    dense features from all views (Front, Back) into the shared 64x64 Minecraft
    UV atlas using the renderer's static UV mappings.

    In 64x64 UV space, multi-view features are combined at their physical surfaces
    (Head, Torso, Arms, Legs). A 2D UV residual block reasons over whole-character
    outer-layer accessories (e.g. hats, 3D hair, jackets) across all 360 degrees.

    The fused UV representation predicts the 64x64 outer occupancy logits and is
    unprojected back to each view's screen coordinates, providing true 3D
    cross-view context to the 2D route head.
    """

    def __init__(self, channels, view_classes, uv_size=64, hidden_channels=64):
        super().__init__()
        if channels < 1 or view_classes < 2:
            raise ValueError(
                "UV multi-view spatial fusion requires channels >= 1 and "
                "view_classes >= 2."
            )
        self.channels = int(channels)
        self.view_classes = int(view_classes)
        self.uv_size = int(uv_size)
        self.hidden_channels = int(hidden_channels)

        self.to_uv_proj = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.uv_net = nn.Sequential(
            nn.Conv2d(hidden_channels + 2, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.occupancy_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        nn.init.constant_(self.occupancy_head.bias, -1.0)

        self.from_uv_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.GroupNorm(norm_groups(channels), channels),
            nn.SiLU(inplace=True),
        )
        self.fusion = nn.Conv2d(channels * 2, channels, kernel_size=1)
        with torch.no_grad():
            self.fusion.weight.zero_()
            for i in range(self.channels):
                self.fusion.weight[i, i, 0, 0] = 1.0
            self.fusion.bias.zero_()

    def forward(self, x, view_ids, static_mappings=None):
        if static_mappings is None or len(static_mappings) < self.view_classes:
            return x, None

        view_count = len(static_mappings)
        if x.shape[0] % view_count != 0 or view_count % self.view_classes != 0:
            return x, None

        skins = x.shape[0] // view_count
        groups_per_skin = view_count // self.view_classes
        total_groups = skins * groups_per_skin
        channels, height, width = x.shape[1:]
        dtype = x.dtype
        device = x.device

        x_grouped = x.reshape(
            skins, groups_per_skin, self.view_classes, channels, height, width
        )
        x_pairs = x_grouped.reshape(
            total_groups, self.view_classes, channels, height, width
        )

        proj_x = self.to_uv_proj(
            x_pairs.reshape(total_groups * self.view_classes, channels, height, width)
        )
        proj_x_pairs = proj_x.reshape(
            total_groups, self.view_classes, self.hidden_channels, height, width
        )

        uv_count = self.uv_size * self.uv_size
        uv_accum = x.new_zeros((total_groups, self.hidden_channels, uv_count))
        weight_accum = x.new_zeros((total_groups, 1, uv_count))

        for g in range(groups_per_skin):
            group_indices = torch.arange(
                g, total_groups, groups_per_skin, device=device
            )
            for v in range(self.view_classes):
                mapping_idx = g * self.view_classes + v
                mapping = static_mappings[mapping_idx]
                masks = mapping["masks"]
                flat_uv = mapping["flat_uv"]
                center_score = mapping.get(
                    "texel_center_score",
                    torch.ones_like(masks, dtype=torch.float32),
                )
                feat_v = proj_x_pairs[group_indices, v]

                for s in range(min(2, masks.shape[0])):
                    mask_s = masks[s]
                    if not mask_s.any():
                        continue
                    uv_s = flat_uv[s][mask_s]
                    w_s = (mask_s.float() * center_score[s].float())[mask_s].to(
                        dtype=dtype
                    )
                    feat_s = feat_v[:, :, mask_s]
                    weighted_feat_s = (feat_s * w_s.view(1, 1, -1)).to(dtype=dtype)

                    uv_accum[group_indices] = uv_accum[group_indices].scatter_add(
                        2,
                        uv_s.view(1, 1, -1).expand(skins, self.hidden_channels, -1),
                        weighted_feat_s,
                    )
                    weight_accum[group_indices] = weight_accum[group_indices].scatter_add(
                        2,
                        uv_s.view(1, 1, -1).expand(skins, 1, -1),
                        w_s.view(1, 1, -1).expand(skins, 1, -1),
                    )

        uv_features = (uv_accum / weight_accum.clamp_min(1e-6)).reshape(
            total_groups, self.hidden_channels, self.uv_size, self.uv_size
        ).to(dtype=dtype)
        uv_support = (weight_accum > 1e-6).to(dtype=dtype).reshape(
            total_groups, 1, self.uv_size, self.uv_size
        )
        uv_coverage = (
            weight_accum / float(self.view_classes)
        ).clamp(0.0, 1.0).to(dtype=dtype).reshape(
            total_groups, 1, self.uv_size, self.uv_size
        )

        uv_in = torch.cat([uv_features, uv_support, uv_coverage], dim=1)
        uv_out = self.uv_net(uv_in)
        occupancy_logits = self.occupancy_head(uv_out)

        uv_context = self.from_uv_proj(uv_out).reshape(
            total_groups, channels, uv_count
        )

        fused_pairs = torch.empty_like(x_pairs)
        for g in range(groups_per_skin):
            group_indices = torch.arange(
                g, total_groups, groups_per_skin, device=device
            )
            group_uv_context = uv_context[group_indices]

            for v in range(self.view_classes):
                mapping_idx = g * self.view_classes + v
                mapping = static_mappings[mapping_idx]
                masks = mapping["masks"]
                flat_uv = mapping["flat_uv"]
                view_x = x_pairs[group_indices, v]
                screen_context = view_x.new_zeros(view_x.shape)

                for s in range(min(2, masks.shape[0])):
                    mask_s = masks[s]
                    if not mask_s.any():
                        continue
                    uv_s = flat_uv[s][mask_s]
                    gathered = group_uv_context.gather(
                        2, uv_s.view(1, 1, -1).expand(skins, channels, -1)
                    )
                    screen_context[:, :, mask_s] = gathered

                fused_v = self.fusion(torch.cat([view_x, screen_context], dim=1))
                fused_pairs[group_indices, v] = fused_v

        fused_x = fused_pairs.reshape(
            skins, groups_per_skin, self.view_classes, channels, height, width
        )
        fused_all = fused_x.reshape(x.shape)

        return fused_all, occupancy_logits


class TextPromptRouteFusion(nn.Module):
    """Turn frozen SigLIP2 text/image similarities into route-logit evidence.

    Prompt embeddings remain fixed in the pretrained contrastive space. Only a
    small convolutional mixer is trained, so the route head can learn which
    semantic concepts support inner, outer, or secondary surfaces without
    fine-tuning either SigLIP2 tower.
    """

    def __init__(
        self,
        raw_feature_dim,
        prompt_count,
        route_classes,
        hidden_channels=32,
        logit_scale=1.0,
        logit_bias=0.0,
    ):
        super().__init__()
        if raw_feature_dim < 1 or prompt_count < 1:
            raise ValueError("Text-prompt feature dimensions must be positive.")
        if route_classes < 2 or hidden_channels < 1:
            raise ValueError("Text-prompt route dimensions must be positive.")
        self.raw_feature_dim = int(raw_feature_dim)
        self.prompt_count = int(prompt_count)
        self.hidden_channels = int(hidden_channels)
        self.register_buffer(
            "prompt_embeddings",
            torch.zeros(prompt_count, raw_feature_dim),
        )
        self.register_buffer(
            "logit_scale",
            torch.tensor(float(logit_scale), dtype=torch.float32),
        )
        self.register_buffer(
            "logit_bias",
            torch.tensor(float(logit_bias), dtype=torch.float32),
        )
        # SigLIP2's pooled vision and text outputs share the contrastive space,
        # but raw patch tokens do not pass through the vision pooling head.
        # Learn a small common local space instead of treating raw patch/text
        # cosine values as calibrated open-vocabulary scores.
        self.spatial_projection = nn.Conv2d(
            raw_feature_dim, hidden_channels, kernel_size=1, bias=False
        )
        self.prompt_projection = nn.Linear(
            raw_feature_dim, hidden_channels, bias=False
        )
        self.route_projection = nn.Sequential(
            nn.Conv2d(prompt_count, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, route_classes, kernel_size=1),
        )
        # Preserve the proven vision-only initialization. The prompt branch
        # starts as a no-op and learns corrections from exact route targets.
        nn.init.zeros_(self.route_projection[-1].weight)
        nn.init.zeros_(self.route_projection[-1].bias)

    def set_prompt_embeddings(self, embeddings, logit_scale=None, logit_bias=None):
        if embeddings.shape != self.prompt_embeddings.shape:
            raise ValueError(
                "Expected text prompt embeddings shaped "
                f"{tuple(self.prompt_embeddings.shape)}, got "
                f"{tuple(embeddings.shape)}."
            )
        with torch.no_grad():
            normalized = F.normalize(
                embeddings.to(
                    device=self.prompt_embeddings.device,
                    dtype=torch.float32,
                ),
                dim=-1,
            )
            self.prompt_embeddings.copy_(normalized)
            if logit_scale is not None:
                self.logit_scale.fill_(float(logit_scale))
            if logit_bias is not None:
                self.logit_bias.fill_(float(logit_bias))

    def forward(
        self,
        raw_features,
        raw_global_features,
        sample_count,
        output_size,
    ):
        if raw_features.dim() == 5:
            raw_features = raw_features.reshape(
                -1,
                raw_features.shape[-3],
                raw_features.shape[-2],
                raw_features.shape[-1],
            )
        if (
            raw_features.dim() != 4
            or raw_features.shape[0] != sample_count
            or raw_features.shape[1] != self.raw_feature_dim
        ):
            raise ValueError(
                "Text-prompt fusion requires spatial SigLIP2 features shaped "
                f"NxCxHxW or BxVxCxHxW; got {tuple(raw_features.shape)}."
            )
        if raw_global_features is None:
            raise ValueError(
                "Text-prompt fusion requires pooled SigLIP2 vision features."
            )
        if raw_global_features.dim() == 3:
            raw_global_features = raw_global_features.reshape(
                -1, raw_global_features.shape[-1]
            )
        if raw_global_features.shape != (
            sample_count,
            self.raw_feature_dim,
        ):
            raise ValueError(
                "Text-prompt fusion requires pooled SigLIP2 features shaped "
                f"NxC or BxVxC; got {tuple(raw_global_features.shape)}."
            )

        prompt_embeddings = F.normalize(
            self.prompt_embeddings.float(), dim=-1
        )
        global_features = F.normalize(
            raw_global_features.float(), dim=-1
        )
        global_logits = torch.einsum(
            "nc,pc->np", global_features, prompt_embeddings
        )
        global_logits = global_logits * self.logit_scale.clamp(0.01, 100.0)
        global_logits = global_logits + self.logit_bias

        local_image_features = F.normalize(
            self.spatial_projection(raw_features.float()), dim=1
        )
        local_prompt_features = F.normalize(
            self.prompt_projection(prompt_embeddings), dim=-1
        )
        local_similarity = torch.einsum(
            "nchw,pc->nphw", local_image_features, local_prompt_features
        )
        # The frozen pooled image/text scores are weak accessory classifiers:
        # in particular, a visible hat can receive a lower score than an image
        # without one. Do not let those scores suppress trainable local prompt
        # evidence. They are retained only as a bounded, per-image residual;
        # exact route supervision decides how the local similarities are used.
        centered_global = global_logits - global_logits.mean(
            dim=-1, keepdim=True
        )
        normalized_global = centered_global / centered_global.std(
            dim=-1, keepdim=True, unbiased=False
        ).clamp_min(1e-4)
        global_residual = 0.10 * torch.tanh(normalized_global)
        prompt_evidence = (
            local_similarity + global_residual[:, :, None, None]
        )
        route_logits = self.route_projection(prompt_evidence)
        route_logits = F.interpolate(
            route_logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        return route_logits, global_logits


class OuterUVGraphBlock(nn.Module):
    """Message passing over physical neighbours of Minecraft outer texels."""

    def __init__(self, channels, dropout=0.05):
        super().__init__()
        self.self_projection = nn.Linear(channels, channels)
        self.neighbour_projection = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes, edge_index, degree):
        source, target = edge_index
        neighbour_sum = torch.zeros_like(nodes)
        neighbour_sum.index_add_(1, target, nodes[:, source])
        neighbour_mean = neighbour_sum / degree.view(1, -1, 1).clamp_min(1.0)
        update = self.self_projection(nodes)
        update = update + self.neighbour_projection(neighbour_mean)
        return self.norm(nodes + self.dropout(F.gelu(update)))


class ProjectedOuterUVTopologyHead(nn.Module):
    """Predict outer alpha from projected image features and cube topology."""

    def __init__(
        self,
        input_channels,
        global_context_dim,
        hidden_channels=64,
        layers=3,
        dropout=0.05,
        uv_size=64,
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("outer_uv_topology_layers must be positive.")
        flat_indices, edge_index = build_outer_uv_graph()
        topology = build_simple_uv_topology()
        parts = topology.part.reshape(-1)[flat_indices]
        faces = topology.face.reshape(-1)[flat_indices]
        local_uv = topology.local_uv.reshape(-1, 2)[flat_indices]
        degree = torch.bincount(
            edge_index[1],
            minlength=flat_indices.numel(),
        ).float()
        self.uv_size = int(uv_size)
        self.register_buffer("flat_indices", flat_indices, persistent=False)
        self.register_buffer("edge_index", edge_index, persistent=False)
        self.register_buffer("degree", degree, persistent=False)
        self.register_buffer("parts", parts, persistent=False)
        self.register_buffer("faces", faces, persistent=False)
        self.register_buffer("local_uv", local_uv, persistent=False)
        self.input_projection = nn.Linear(input_channels, hidden_channels)
        self.part_embedding = nn.Embedding(6, hidden_channels)
        self.face_embedding = nn.Embedding(6, hidden_channels)
        self.position_projection = nn.Linear(2, hidden_channels)
        self.global_projection = nn.Sequential(
            nn.LayerNorm(global_context_dim),
            nn.Linear(global_context_dim, hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            OuterUVGraphBlock(hidden_channels, dropout=dropout)
            for _ in range(layers)
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, 1),
        )
        # A very negative initialization made the one-epoch recipe prone to
        # the all-transparent solution after precision penalties were added.
        # Keep a sparse prior without starving early positive gradients.
        nn.init.constant_(self.output[-1].bias, -1.0)

    def forward(self, atlas_features, global_context):
        if atlas_features.dim() != 4:
            raise ValueError("Projected outer UV features must be BCHW.")
        if atlas_features.shape[-2:] != (self.uv_size, self.uv_size):
            raise ValueError(
                "Projected outer UV feature size must match the UV atlas."
            )
        flattened = atlas_features.flatten(2).transpose(1, 2)
        nodes = flattened.index_select(1, self.flat_indices)
        nodes = self.input_projection(nodes.float())
        nodes = nodes + self.part_embedding(self.parts).unsqueeze(0)
        nodes = nodes + self.face_embedding(self.faces).unsqueeze(0)
        nodes = nodes + self.position_projection(self.local_uv).unsqueeze(0)
        nodes = nodes + self.global_projection(global_context.float()).unsqueeze(1)
        for block in self.blocks:
            nodes = block(nodes, self.edge_index, self.degree)
        node_logits = self.output(nodes).squeeze(-1)
        atlas_logits = node_logits.new_full(
            (node_logits.shape[0], self.uv_size * self.uv_size),
            -12.0,
        )
        atlas_logits[:, self.flat_indices] = node_logits
        return atlas_logits.reshape(
            node_logits.shape[0],
            1,
            self.uv_size,
            self.uv_size,
        )


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
        semantic_spatial_feature_dim=0,
        semantic_spatial_channels=64,
        semantic_text_prompt_count=0,
        semantic_text_prompt_feature_dim=0,
        semantic_text_prompt_channels=32,
        semantic_text_logit_scale=1.0,
        semantic_text_logit_bias=0.0,
        predict_confidence=False,
        route_role_spatial_prior=False,
        route_prior_height=32,
        route_prior_width=16,
        route_prior_logit_cap=1.5,
        route_prior_dropout=0.10,
        predict_outer_uv_occupancy=False,
        predict_head_outer_structure=False,
        head_outer_structure_mode="global",
        head_outer_projected_input_version=1,
        outer_uv_feature_channels=32,
        outer_uv_topology_channels=64,
        outer_uv_topology_layers=3,
        outer_uv_topology_dropout=0.05,
        outer_uv_route_evidence_dropout=1.0,
        cross_view_spatial_fusion=False,
    ):
        super().__init__()
        self.geometry_only = bool(geometry_only)
        if layer_classes is None:
            layer_classes = 3 if self.geometry_only else 2
        self.layer_classes = int(layer_classes)
        self.uv_size = int(uv_size)
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
        self.semantic_spatial_feature_dim = int(semantic_spatial_feature_dim)
        self.semantic_spatial_channels = int(semantic_spatial_channels)
        self.semantic_text_prompt_count = int(semantic_text_prompt_count)
        self.semantic_text_prompt_feature_dim = int(
            semantic_text_prompt_feature_dim
        )
        self.semantic_text_prompt_channels = int(
            semantic_text_prompt_channels
        )
        self.predict_confidence = bool(predict_confidence)
        self.route_role_spatial_prior = bool(route_role_spatial_prior)
        self.route_prior_height = int(route_prior_height)
        self.route_prior_width = int(route_prior_width)
        self.route_prior_logit_cap = float(route_prior_logit_cap)
        self.route_prior_dropout = float(route_prior_dropout)
        self.predict_outer_uv_occupancy = bool(predict_outer_uv_occupancy)
        self.predict_head_outer_structure = bool(
            predict_head_outer_structure
        )
        self.head_outer_structure_mode = str(head_outer_structure_mode)
        if self.head_outer_structure_mode not in ("global", "projected"):
            raise ValueError(
                "head_outer_structure_mode must be 'global' or 'projected'."
            )
        self.head_outer_projected_input_version = int(
            head_outer_projected_input_version
        )
        if self.head_outer_projected_input_version not in (1, 2, 3, 4):
            raise ValueError(
                "head_outer_projected_input_version must be 1, 2, 3, or 4."
            )
        self.outer_uv_feature_channels = int(outer_uv_feature_channels)
        self.outer_uv_topology_channels = int(outer_uv_topology_channels)
        self.outer_uv_topology_layers = int(outer_uv_topology_layers)
        self.outer_uv_topology_dropout = float(outer_uv_topology_dropout)
        self.outer_uv_route_evidence_dropout = float(
            outer_uv_route_evidence_dropout
        )
        self.cross_view_spatial_fusion = bool(cross_view_spatial_fusion)
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
        if not 0.0 <= self.outer_uv_route_evidence_dropout <= 1.0:
            raise ValueError(
                "outer_uv_route_evidence_dropout must be in [0, 1]."
            )
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
        self.semantic_spatial_fusion = (
            SpatialSemanticFusion(
                self.semantic_spatial_feature_dim,
                self.semantic_spatial_channels,
                c * 8,
            )
            if self.semantic_spatial_feature_dim > 0
            else None
        )
        self.up2 = UpBlock(c * 8, c * 4, c * 4)
        self.up1 = UpBlock(c * 4, c * 2, c * 2)
        self.up0 = UpBlock(c * 2, c, c)
        self.features = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.cross_view_spatial = (
            UVMultiViewSpatialFusion(c, self.view_classes, uv_size=uv_size)
            if self.cross_view_spatial_fusion and self.view_classes > 1
            else None
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
        if self.predict_outer_uv_occupancy:
            if self.view_classes < 1:
                raise ValueError(
                    "Outer UV occupancy prediction requires grouped fixed-view inputs."
                )
            if self.outer_uv_feature_channels < 1:
                raise ValueError("outer_uv_feature_channels must be positive.")
            occupancy_context_channels = c * 8 + (
                self.semantic_channels if self.semantic_fusion is not None else 0
            )
            self.outer_uv_feature_projection = nn.Sequential(
                nn.Conv2d(c, self.outer_uv_feature_channels, kernel_size=1),
                nn.GELU(),
            )
            # Projected feature mean + mean/max p_outer + foreground coverage
            # + number of supporting views.
            occupancy_input_channels = self.outer_uv_feature_channels + 4
            self.outer_uv_occupancy_head = ProjectedOuterUVTopologyHead(
                occupancy_input_channels,
                occupancy_context_channels,
                hidden_channels=self.outer_uv_topology_channels,
                layers=self.outer_uv_topology_layers,
                dropout=self.outer_uv_topology_dropout,
                uv_size=uv_size,
            )
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

        # Construct the optional prompt branch after every established parser
        # module. With a fixed seed this keeps the vision-only initialization
        # bit-for-bit unchanged; the new branch starts as an additive no-op.
        self.semantic_text_prompt_fusion = (
            TextPromptRouteFusion(
                self.semantic_text_prompt_feature_dim,
                self.semantic_text_prompt_count,
                self.layer_classes,
                hidden_channels=self.semantic_text_prompt_channels,
                logit_scale=semantic_text_logit_scale,
                logit_bias=semantic_text_logit_bias,
            )
            if self.semantic_text_prompt_count > 0
            else None
        )
        if self.semantic_text_prompt_fusion is not None and (
            self.semantic_spatial_feature_dim
            != self.semantic_text_prompt_feature_dim
        ):
            raise ValueError(
                "SigLIP2 text prompts must share the spatial vision feature "
                "dimension."
            )
        # Keep this auxiliary branch last so enabling it cannot perturb the
        # seeded initialization of the established parser or prompt branch.
        # It supervises head outer-layer structure during training but does not
        # gate inference-time route logits.
        if self.predict_head_outer_structure:
            if self.semantic_fusion is None:
                raise ValueError(
                    "Head outer structure prediction requires global semantic "
                    "features."
                )
            self.head_outer_face_presence_head = nn.Linear(
                self.semantic_channels, 6
            )
            self.head_outer_face_coverage_head = nn.Linear(
                self.semantic_channels, 6
            )
            self.head_outer_symmetry_head = (
                nn.Sequential(
                    nn.LayerNorm(self.semantic_channels),
                    nn.Linear(
                        self.semantic_channels,
                        self.semantic_channels,
                    ),
                    nn.GELU(),
                    nn.Linear(self.semantic_channels, 1),
                )
                if self.head_outer_projected_input_version >= 3
                else None
            )
            self.head_outer_accessory_head = (
                nn.Sequential(
                    nn.LayerNorm(self.semantic_channels),
                    nn.Linear(
                        self.semantic_channels,
                        self.semantic_channels,
                    ),
                    nn.GELU(),
                    nn.Linear(self.semantic_channels, 2),
                )
                if self.head_outer_projected_input_version >= 4
                else None
            )
            if self.head_outer_structure_mode == "global":
                self.head_outer_face_occupancy_head = nn.Sequential(
                    nn.LayerNorm(self.semantic_channels),
                    nn.Linear(
                        self.semantic_channels, self.semantic_channels * 2
                    ),
                    nn.GELU(),
                    nn.Linear(self.semantic_channels * 2, 6 * 8 * 8),
                )
                nn.init.zeros_(
                    self.head_outer_face_occupancy_head[-1].weight
                )
                nn.init.zeros_(
                    self.head_outer_face_occupancy_head[-1].bias
                )
            else:
                if self.view_classes < 1:
                    raise ValueError(
                        "Projected head structure prediction requires grouped "
                        "fixed-view inputs."
                    )
                self.head_outer_uv_feature_projection = nn.Sequential(
                    nn.Conv2d(
                        c,
                        self.outer_uv_feature_channels,
                        kernel_size=1,
                    ),
                    nn.GELU(),
                )
                head_context_channels = c * 8 + self.semantic_channels
                # Projected image features, foreground coverage, and view
                # support are decoded on the physical outer-UV graph. Only
                # the six head faces are supervised and returned.
                self.head_outer_projected_head = (
                    ProjectedOuterUVTopologyHead(
                        self.outer_uv_feature_channels
                        + (
                            9
                            if self.head_outer_projected_input_version >= 2
                            else 2
                        ),
                        head_context_channels,
                        hidden_channels=self.outer_uv_topology_channels,
                        layers=self.outer_uv_topology_layers,
                        dropout=self.outer_uv_topology_dropout,
                        uv_size=uv_size,
                    )
                )

    def _runtime_semantic_features(self, images, foreground=None):
        backbone = getattr(self, "_runtime_semantic_backbone", None)
        if backbone is None:
            raise ValueError(
                "This parser requires semantic_features or an attached semantic "
                "runtime backbone."
            )
        rgb = images[:, :3]
        if foreground is not None:
            if foreground.dim() == 3:
                foreground = foreground.unsqueeze(1)
            if (
                foreground.dim() != 4
                or foreground.shape[0] != rgb.shape[0]
                or foreground.shape[1] != 1
            ):
                raise ValueError(
                    "semantic_foreground must be shaped Nx1xHxW or NxHxW; got "
                    f"{tuple(foreground.shape)}."
                )
            if foreground.shape[-2:] != rgb.shape[-2:]:
                foreground = F.interpolate(
                    foreground.float(),
                    size=rgb.shape[-2:],
                    mode="nearest",
                )
            foreground = foreground.to(device=rgb.device, dtype=rgb.dtype).clamp(0, 1)
            # A fixed neutral background prevents the frozen spatial tower from
            # learning accidental correlations with randomized training colors.
            rgb = rgb * foreground + 0.5 * (1.0 - foreground)
        with torch.no_grad():
            if hasattr(backbone, "encode_dense"):
                return backbone.encode_dense(rgb)
            return backbone.encode_global(rgb)

    def set_semantic_text_prompt_embeddings(
        self,
        embeddings,
        logit_scale=None,
        logit_bias=None,
    ):
        if self.semantic_text_prompt_fusion is None:
            raise ValueError("This parser has no SigLIP2 text-prompt branch.")
        self.semantic_text_prompt_fusion.set_prompt_embeddings(
            embeddings,
            logit_scale=logit_scale,
            logit_bias=logit_bias,
        )

    def forward(
        self,
        x,
        view_ids=None,
        semantic_features=None,
        semantic_foreground=None,
        static_mappings=None,
    ):
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
        if (
            self.semantic_fusion is not None
            or self.semantic_spatial_fusion is not None
        ) and semantic_features is None:
            semantic_features = self._runtime_semantic_features(
                source_images, foreground=semantic_foreground
            )
        if isinstance(semantic_features, dict):
            semantic_global = semantic_features.get(
                "raw_global", semantic_features.get("global")
            )
            semantic_spatial = semantic_features.get(
                "raw_spatial", semantic_features.get("spatial")
            )
        else:
            semantic_global = semantic_features
            semantic_spatial = None
        if self.semantic_spatial_fusion is not None:
            if semantic_spatial is None:
                raise ValueError(
                    "This parser checkpoint requires spatial semantic features."
                )
            x = x + self.semantic_spatial_fusion(
                semantic_spatial,
                source_images.shape[0],
                x.shape[-2:],
            ).to(dtype=x.dtype)
        visual_summary = x.mean(dim=(2, 3))
        grouped_visual_summary = visual_summary
        if self.predict_outer_uv_occupancy or (
            self.predict_head_outer_structure
            and self.head_outer_structure_mode == "projected"
        ):
            if visual_summary.shape[0] % self.view_classes != 0:
                raise ValueError(
                    "Projected UV prediction requires complete fixed-view "
                    "groups."
                )
            grouped_visual_summary = visual_summary.reshape(
                -1, self.view_classes, visual_summary.shape[-1]
            ).mean(dim=1)
        semantic_summary = None
        if self.semantic_fusion is not None:
            if semantic_global is None:
                raise ValueError(
                    "This parser checkpoint requires global semantic features."
                )
            modulation, semantic_summary = self.semantic_fusion(
                semantic_global,
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
        cross_view_occupancy = None
        if self.cross_view_spatial is not None:
            x, cross_view_occupancy = self.cross_view_spatial(
                x, view_ids, static_mappings=static_mappings
            )
        occupancy_feature_source = x
        x = self.feature_dropout(x)
        layer_evidence = self.layer(x)
        text_prompt_route_logits = None
        text_prompt_scores = None
        if self.semantic_text_prompt_fusion is not None:
            if semantic_spatial is None:
                raise ValueError(
                    "This parser checkpoint requires spatial SigLIP2 features "
                    "for text-prompt fusion."
                )
            (
                text_prompt_route_logits,
                text_prompt_scores,
            ) = self.semantic_text_prompt_fusion(
                semantic_spatial,
                semantic_global,
                source_images.shape[0],
                layer_evidence.shape[-2:],
            )
            layer_evidence = layer_evidence + text_prompt_route_logits.to(
                dtype=layer_evidence.dtype
            )
        outputs = {
            "foreground": self.foreground(x),
            "layer": layer_evidence,
        }
        if cross_view_occupancy is not None:
            outputs["outer_uv_occupancy_logits"] = cross_view_occupancy
        if text_prompt_route_logits is not None:
            outputs["text_prompt_route_logits"] = text_prompt_route_logits
            outputs["text_prompt_scores"] = text_prompt_scores
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
            if self.predict_head_outer_structure:
                outputs["head_outer_face_presence_logits"] = (
                    self.head_outer_face_presence_head(semantic_summary)
                )
                outputs["head_outer_face_coverage"] = torch.sigmoid(
                    self.head_outer_face_coverage_head(semantic_summary)
                )
                if self.head_outer_symmetry_head is not None:
                    # The head-outer structure heads are allowed to backprop
                    # into the semantic/parser trunk so that crown, brim, and
                    # accessory evidence can improve the shared representation.
                    outputs["head_outer_symmetry_logit"] = (
                        self.head_outer_symmetry_head(
                            semantic_summary
                        ).squeeze(-1)
                    )
                if self.head_outer_accessory_head is not None:
                    # Explicit structural labels are more selective than the
                    # nearly universal left/right alpha-symmetry score. They
                    # decide whether deterministic completion may close a
                    # four-face brim or an open top rim, and their gradients
                    # now also improve the trunk for these rare structures.
                    outputs["head_outer_accessory_logits"] = (
                        self.head_outer_accessory_head(
                            semantic_summary
                        )
                    )
                if self.head_outer_structure_mode == "global":
                    outputs["head_outer_face_occupancy_logits"] = (
                        self.head_outer_face_occupancy_head(semantic_summary)
                        .reshape(-1, 6, 8, 8)
                    )
        if self.predict_outer_uv_occupancy:
            occupancy_summary = (
                torch.cat([grouped_visual_summary, semantic_summary], dim=1)
                if semantic_summary is not None
                else grouped_visual_summary
            )
            # The outer-UV occupancy head is still detached to keep the sparse
            # atlas objective from destabilizing the primary image-space route.
            outputs["outer_uv_features"] = self.outer_uv_feature_projection(
                occupancy_feature_source.detach()
            )
            outputs["outer_uv_global_context"] = occupancy_summary.detach()
        if (
            self.predict_head_outer_structure
            and self.head_outer_structure_mode == "projected"
        ):
            # Unlike the optional full-atlas occupancy head, the head-outer
            # branch is intentionally not detached: crown/hat/brim errors are
            # allowed to improve the shared parser trunk directly.
            outputs["head_outer_uv_features"] = (
                self.head_outer_uv_feature_projection(
                    occupancy_feature_source
                )
            )
            outputs["head_outer_uv_global_context"] = torch.cat(
                [grouped_visual_summary, semantic_summary], dim=1
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

    def predict_projected_outer_uv_occupancy(
        self,
        atlas_features,
        global_context,
    ):
        if not self.predict_outer_uv_occupancy:
            raise ValueError("This parser has no outer UV occupancy head.")
        return self.outer_uv_occupancy_head(
            atlas_features,
            global_context,
        )

    def predict_projected_head_outer_structure(
        self,
        atlas_features,
        global_context,
    ):
        if (
            not self.predict_head_outer_structure
            or self.head_outer_structure_mode != "projected"
        ):
            raise ValueError(
                "This parser has no projected head outer structure head."
            )
        atlas_logits = self.head_outer_projected_head(
            atlas_features,
            global_context,
        )
        head_indices = build_head_outer_face_indices().to(
            atlas_logits.device
        )
        return (
            atlas_logits.flatten(2)[:, 0]
            .index_select(1, head_indices)
            .reshape(-1, 6, 8, 8)
        )


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
