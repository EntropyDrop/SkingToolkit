import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from SkingToolkit.semantic_uv_reconstruction.dataset import load_uv_masks
from SkingToolkit.semantic_uv_reconstruction.model import ConvBlock, norm_groups
from SkingToolkit.semantic_uv_reconstruction.semantic_backbone import SigLIP2VisionBackbone


class ResidualDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.block = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        return self.block(self.down(x))


class FixedViewEncoder(nn.Module):
    """Shared encoder returning both texel-detail and semantic feature maps."""

    def __init__(self, base_channels=32, token_channels=128):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(5, c, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(norm_groups(c), c),
            nn.SiLU(inplace=True),
        )
        self.down1 = ResidualDownBlock(c, c * 2)
        self.down2 = ResidualDownBlock(c * 2, c * 4)
        self.down3 = ResidualDownBlock(c * 4, token_channels)

    def forward(self, image):
        height, width = image.shape[-2:]
        y = torch.linspace(-1.0, 1.0, height, device=image.device, dtype=image.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=image.device, dtype=image.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(image.shape[0], -1, -1, -1)
        features = torch.cat([image, coords], dim=1)
        features = self.stem(features)
        features = self.down1(features)
        detail_features = self.down2(features)
        coarse_features = self.down3(detail_features)
        return coarse_features, detail_features


class UVQueryBlock(nn.Module):
    def __init__(self, channels, heads=4, dropout=0.0):
        super().__init__()
        self.self_norm = nn.LayerNorm(channels)
        self.self_attention = nn.MultiheadAttention(
            channels, heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(channels)
        self.memory_norm = nn.LayerNorm(channels)
        self.cross_attention = nn.MultiheadAttention(
            channels, heads, dropout=dropout, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, queries, memory, memory_valid=None):
        normalized = self.self_norm(queries)
        attended, _ = self.self_attention(normalized, normalized, normalized, need_weights=False)
        queries = queries + attended
        attended, _ = self.cross_attention(
            self.cross_norm(queries),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=None if memory_valid is None else ~memory_valid,
            need_weights=False,
        )
        queries = queries + attended
        return queries + self.feed_forward(self.ff_norm(queries))


class UVCrossAttentionBlock(nn.Module):
    """Cross-attend UV queries to compact latents without quadratic self-attention."""

    def __init__(self, channels, heads=4, dropout=0.0):
        super().__init__()
        self.query_norm = nn.LayerNorm(channels)
        self.memory_norm = nn.LayerNorm(channels)
        self.cross_attention = nn.MultiheadAttention(
            channels, heads, dropout=dropout, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, queries, memory):
        normalized_memory = self.memory_norm(memory)
        attended, _ = self.cross_attention(
            self.query_norm(queries),
            normalized_memory,
            normalized_memory,
            need_weights=False,
        )
        queries = queries + attended
        return queries + self.feed_forward(self.ff_norm(queries))


class UVSpatialMixer(nn.Module):
    """Cheap local communication between UV queries on their native 2D grid."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.GroupNorm(norm_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, queries, query_size):
        batch, _, channels = queries.shape
        features = queries.transpose(1, 2).reshape(
            batch, channels, query_size, query_size
        )
        features = features + self.block(features)
        return features.flatten(2).transpose(1, 2)


class UVUpsampleBlock(nn.Module):
    """Learn a separate feature vector for every 2x2 output texel block.

    Bilinear interpolation irreversibly averages neighboring UV query features,
    which is especially damaging for Minecraft pixel art. PixelShuffle lets each
    coarse query decode four distinct child texels without interpolation blur.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.expand = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.block = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.shuffle(self.expand(x))
        return self.block(x)


class SemanticUVReconstructor(nn.Module):
    """Fuse exact image detail with frozen open-vocabulary visual semantics.

    A high-resolution CNN preserves pixels and edges. UV queries jointly attend
    to its front/back tokens and frozen SigLIP2 patch tokens. Only Minecraft's
    finite atlas topology is classified; clothing appearance remains a continuous
    language-aligned embedding rather than a closed garment vocabulary.
    """

    def __init__(
        self,
        view_count=2,
        base_channels=32,
        token_channels=128,
        query_size=32,
        attention_heads=4,
        attention_layers=2,
        attention_dropout=0.0,
        memory_latents=256,
        semantic_classes=13,
        semantic_backbone="none",
        siglip_model="google/siglip2-base-patch16-224",
        siglip_local_files_only=False,
        open_semantic_backbone=None,
        use_siglip_patch_tokens=False,
    ):
        super().__init__()
        if view_count < 1:
            raise ValueError("view_count must be positive.")
        if query_size < 4 or query_size > 64 or 64 % query_size != 0:
            raise ValueError("query_size must be a divisor of 64 in [4, 64].")
        upsample_ratio = 64 // query_size
        if upsample_ratio & (upsample_ratio - 1):
            raise ValueError("64 / query_size must be a power of two.")
        if token_channels % attention_heads != 0:
            raise ValueError("token_channels must be divisible by attention_heads.")
        if memory_latents < 16:
            raise ValueError("memory_latents must be at least 16.")

        self.view_count = int(view_count)
        self.base_channels = int(base_channels)
        self.token_channels = int(token_channels)
        self.query_size = int(query_size)
        self.attention_heads = int(attention_heads)
        self.attention_layers = int(attention_layers)
        self.attention_dropout = float(attention_dropout)
        self.memory_latent_count = int(memory_latents)
        self.semantic_classes = int(semantic_classes)
        self.semantic_backbone_name = str(semantic_backbone)
        self.siglip_model = str(siglip_model)
        self.use_siglip_patch_tokens = bool(use_siglip_patch_tokens)
        self.architecture_version = 3

        self.encoder = FixedViewEncoder(base_channels, token_channels)
        self.detail_projection = nn.Sequential(
            nn.Conv2d(base_channels * 4, token_channels, kernel_size=1),
            nn.GroupNorm(norm_groups(token_channels), token_channels),
            nn.SiLU(inplace=True),
        )
        self.view_embedding = nn.Parameter(torch.randn(view_count, token_channels) * 0.02)
        # Coarse CNN, high-resolution CNN detail, and SigLIP2 are distinct sources.
        self.source_embedding = nn.Parameter(torch.randn(3, token_channels) * 0.02)
        if open_semantic_backbone is not None:
            self.open_semantic_backbone = open_semantic_backbone
            self.semantic_backbone_name = "injected"
        elif semantic_backbone == "siglip2":
            self.open_semantic_backbone = SigLIP2VisionBackbone(
                model_name=siglip_model,
                token_channels=token_channels,
                local_files_only=siglip_local_files_only,
            )
        elif semantic_backbone == "none":
            self.open_semantic_backbone = None
        else:
            raise ValueError("semantic_backbone must be 'siglip2' or 'none'.")
        if (
            self.open_semantic_backbone is not None
            and not self.use_siglip_patch_tokens
            and hasattr(self.open_semantic_backbone, "token_projection")
        ):
            self.open_semantic_backbone.token_projection.requires_grad_(False)
        self.uv_queries = nn.Parameter(
            torch.randn(query_size * query_size, token_channels) * 0.02
        )
        self.memory_latents = nn.Parameter(
            torch.randn(memory_latents, token_channels) * 0.02
        )
        self.semantic_bottleneck = nn.Sequential(
            nn.LayerNorm(token_channels),
            nn.Linear(token_channels, token_channels),
            nn.GELU(),
            nn.Linear(token_channels, token_channels),
        )
        self.memory_resampler = UVQueryBlock(
            token_channels, attention_heads, attention_dropout
        )
        self.query_blocks = nn.ModuleList(
            UVCrossAttentionBlock(token_channels, attention_heads, attention_dropout)
            for _ in range(attention_layers)
        )
        self.query_mixers = nn.ModuleList(
            UVSpatialMixer(token_channels) for _ in range(attention_layers)
        )

        decoder_blocks = []
        decoder_channels = token_channels
        for _ in range(int(math.log2(upsample_ratio))):
            next_channels = max(base_channels, decoder_channels // 2)
            decoder_blocks.append(UVUpsampleBlock(decoder_channels, next_channels))
            decoder_channels = next_channels
        self.decoder = nn.Sequential(*decoder_blocks)
        self.output_features = ConvBlock(decoder_channels, max(base_channels, decoder_channels))
        decoder_channels = max(base_channels, decoder_channels)
        self.rgb_head = nn.Conv2d(decoder_channels, 3, kernel_size=1)
        self.alpha_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.semantic_uv_head = (
            nn.Conv2d(decoder_channels, semantic_classes, kernel_size=1)
            if semantic_classes > 0
            else None
        )

        self.outer_presence_head = nn.Linear(token_channels, 6)
        self.outer_coverage_head = nn.Linear(token_channels, 6)
        self.part_color_head = nn.Linear(token_channels, 12 * 3)

        masks = load_uv_masks()
        if masks is None:
            raise FileNotFoundError("skin-mask.png and skin-decor-mask.png are required.")
        base_mask, decor_mask = masks
        self.register_buffer("base_mask", base_mask.unsqueeze(0), persistent=True)
        self.register_buffer("decor_mask", decor_mask.unsqueeze(0), persistent=True)

    @property
    def has_open_semantics(self):
        return self.open_semantic_backbone is not None

    def encode_open_semantics(self, images):
        """Return a continuous front/back appearance embedding.

        This method intentionally keeps autograd enabled. Frozen SigLIP2 weights
        do not receive gradients, but predicted render pixels do.
        """
        if self.open_semantic_backbone is None:
            raise RuntimeError("No open semantic backbone is configured.")
        if images.dim() != 5 or images.shape[2] != 3:
            raise ValueError(f"Expected images shaped BxVx3xHxW, got {tuple(images.shape)}.")
        batch, views, channels, height, width = images.shape
        flat_images = images.reshape(batch * views, channels, height, width)
        if flat_images.is_cuda:
            flat_images = flat_images.contiguous(memory_format=torch.channels_last)
        encoded = self.open_semantic_backbone(flat_images)
        raw_global = encoded["raw_global"].reshape(batch, views, -1)
        return F.normalize(raw_global.float(), dim=-1)

    def forward(self, images, open_semantic_raw=None):
        if images.dim() != 5 or images.shape[2] != 3:
            raise ValueError(f"Expected images shaped BxVx3xHxW, got {tuple(images.shape)}.")
        batch, views, channels, height, width = images.shape
        if views != self.view_count:
            raise ValueError(f"Expected {self.view_count} views, got {views}.")
        if open_semantic_raw is not None and self.open_semantic_backbone is None:
            raise ValueError("Cached SigLIP features require an open semantic backbone.")

        flat_images = images.reshape(batch * views, channels, height, width)
        if flat_images.is_cuda:
            flat_images = flat_images.contiguous(memory_format=torch.channels_last)
        encoded, detail_encoded = self.encoder(flat_images)
        _, token_channels, feature_height, feature_width = encoded.shape
        coarse_memory = encoded.flatten(2).transpose(1, 2)
        coarse_memory = coarse_memory.reshape(
            batch, views, feature_height * feature_width, token_channels
        )
        coarse_memory = (
            coarse_memory
            + self.view_embedding.view(1, views, 1, token_channels)
            + self.source_embedding[0].view(1, 1, 1, token_channels)
        )
        coarse_memory = coarse_memory.reshape(
            batch, views * feature_height * feature_width, token_channels
        )

        detail_encoded = self.detail_projection(detail_encoded)
        detail_height, detail_width = detail_encoded.shape[-2:]
        detail_memory = detail_encoded.flatten(2).transpose(1, 2).reshape(
            batch, views, detail_height * detail_width, token_channels
        )
        detail_memory = (
            detail_memory
            + self.view_embedding.view(1, views, 1, token_channels)
            + self.source_embedding[1].view(1, 1, 1, token_channels)
        ).reshape(batch, views * detail_height * detail_width, token_channels)

        memory_parts = [detail_memory, coarse_memory]
        open_memory_requires_mask = False
        semantic_seed = coarse_memory.mean(dim=1)
        open_semantic_embedding = None
        if self.open_semantic_backbone is not None:
            if open_semantic_raw is None:
                open_features = (
                    self.open_semantic_backbone(flat_images)
                    if self.use_siglip_patch_tokens
                    else self.open_semantic_backbone.encode_global(flat_images)
                )
                raw_global = open_features["raw_global"].reshape(batch, views, -1)
                projected_global = open_features["global"].reshape(
                    batch, views, token_channels
                )
                if self.use_siglip_patch_tokens:
                    open_tokens = open_features["tokens"].reshape(
                        batch, views, -1, token_channels
                    )
                    open_tokens = (
                        open_tokens
                        + self.view_embedding.view(1, views, 1, token_channels)
                        + self.source_embedding[2].view(1, 1, 1, token_channels)
                    )
                    open_tokens = open_tokens.reshape(batch, -1, token_channels)
                    open_token_mask = open_features["token_mask"].reshape(batch, -1)
                    memory_parts.append(open_tokens)
                    open_memory_requires_mask = not bool(
                        open_features.get("tokens_compact", False)
                    )
            else:
                if open_semantic_raw.dim() != 3 or open_semantic_raw.shape[:2] != (
                    batch,
                    views,
                ):
                    raise ValueError(
                        "Cached SigLIP features must be shaped BxVxD; got "
                        f"{tuple(open_semantic_raw.shape)}."
                    )
                raw_global = open_semantic_raw.float()
                projected_global = self.open_semantic_backbone.project_global(
                    raw_global.reshape(batch * views, -1)
                ).reshape(batch, views, token_channels)
                if self.use_siglip_patch_tokens:
                    raise ValueError(
                        "A global-only SigLIP cache cannot supply patch tokens."
                    )
            open_global = projected_global.mean(dim=1)
            semantic_seed = semantic_seed + open_global
            open_semantic_view_embeddings = F.normalize(raw_global.float(), dim=-1)
            open_semantic_embedding = F.normalize(raw_global.mean(dim=1).float(), dim=-1)

        memory = torch.cat(memory_parts, dim=1)
        memory_valid = None
        if open_memory_requires_mask:
            cnn_valid = torch.ones(
                (batch, detail_memory.shape[1] + coarse_memory.shape[1]),
                dtype=torch.bool,
                device=images.device,
            )
            memory_valid = torch.cat([cnn_valid, open_token_mask], dim=1)
        semantic = self.semantic_bottleneck(semantic_seed)
        latents = self.memory_latents.unsqueeze(0).expand(batch, -1, -1)
        latents = latents + semantic.unsqueeze(1)
        latents = self.memory_resampler(latents, memory, memory_valid)
        queries = self.uv_queries.unsqueeze(0).expand(batch, -1, -1)
        queries = queries + semantic.unsqueeze(1)
        for block, mixer in zip(self.query_blocks, self.query_mixers):
            queries = block(queries, latents)
            queries = mixer(queries, self.query_size)

        features = queries.transpose(1, 2).reshape(
            batch, token_channels, self.query_size, self.query_size
        )
        features = self.decoder(features)
        features = self.output_features(features)
        if features.shape[-2:] != (64, 64):
            features = F.interpolate(features, size=(64, 64), mode="bilinear", align_corners=False)

        rgb = torch.sigmoid(self.rgb_head(features))
        alpha_logits = self.alpha_head(features)
        outer_alpha = torch.sigmoid(alpha_logits) * self.decor_mask
        alpha = (self.base_mask + outer_alpha).clamp(0.0, 1.0)
        valid_mask = (self.base_mask + self.decor_mask).clamp(0.0, 1.0)
        uv = torch.cat([rgb * valid_mask, alpha], dim=1)

        outputs = {
            "uv": uv,
            "rgb": rgb,
            "alpha_logits": alpha_logits,
            "semantic": semantic,
            "outer_presence_logits": self.outer_presence_head(semantic),
            "outer_coverage": torch.sigmoid(self.outer_coverage_head(semantic)),
            "part_colors": torch.sigmoid(self.part_color_head(semantic)).reshape(batch, 12, 3),
        }
        if open_semantic_embedding is not None:
            outputs["open_semantic_embedding"] = open_semantic_embedding
            outputs["open_semantic_view_embeddings"] = open_semantic_view_embeddings
        if self.semantic_uv_head is not None:
            outputs["semantic_uv_logits"] = self.semantic_uv_head(features)
        return outputs


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
