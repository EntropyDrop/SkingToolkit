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
    """Shared encoder for each fixed render view."""

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
        features = self.down2(features)
        return self.down3(features)


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


class UVUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
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
        query_size=16,
        attention_heads=4,
        attention_layers=2,
        attention_dropout=0.0,
        semantic_classes=13,
        semantic_backbone="none",
        siglip_model="google/siglip2-base-patch16-224",
        siglip_local_files_only=False,
        open_semantic_backbone=None,
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

        self.view_count = int(view_count)
        self.base_channels = int(base_channels)
        self.token_channels = int(token_channels)
        self.query_size = int(query_size)
        self.attention_heads = int(attention_heads)
        self.attention_layers = int(attention_layers)
        self.attention_dropout = float(attention_dropout)
        self.semantic_classes = int(semantic_classes)
        self.semantic_backbone_name = str(semantic_backbone)
        self.siglip_model = str(siglip_model)

        self.encoder = FixedViewEncoder(base_channels, token_channels)
        self.view_embedding = nn.Parameter(torch.randn(view_count, token_channels) * 0.02)
        self.source_embedding = nn.Parameter(torch.randn(2, token_channels) * 0.02)
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
        self.uv_queries = nn.Parameter(
            torch.randn(query_size * query_size, token_channels) * 0.02
        )
        self.semantic_bottleneck = nn.Sequential(
            nn.LayerNorm(token_channels),
            nn.Linear(token_channels, token_channels),
            nn.GELU(),
            nn.Linear(token_channels, token_channels),
        )
        self.query_blocks = nn.ModuleList(
            UVQueryBlock(token_channels, attention_heads, attention_dropout)
            for _ in range(attention_layers)
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
        encoded = self.open_semantic_backbone(
            images.reshape(batch * views, channels, height, width)
        )
        raw_global = encoded["raw_global"].reshape(batch, views, -1)
        return F.normalize(raw_global.float(), dim=-1)

    def forward(self, images):
        if images.dim() != 5 or images.shape[2] != 3:
            raise ValueError(f"Expected images shaped BxVx3xHxW, got {tuple(images.shape)}.")
        batch, views, channels, height, width = images.shape
        if views != self.view_count:
            raise ValueError(f"Expected {self.view_count} views, got {views}.")

        encoded = self.encoder(images.reshape(batch * views, channels, height, width))
        _, token_channels, feature_height, feature_width = encoded.shape
        cnn_memory = encoded.flatten(2).transpose(1, 2)
        cnn_memory = cnn_memory.reshape(
            batch, views, feature_height * feature_width, token_channels
        )
        cnn_memory = (
            cnn_memory
            + self.view_embedding.view(1, views, 1, token_channels)
            + self.source_embedding[0].view(1, 1, 1, token_channels)
        )
        cnn_memory = cnn_memory.reshape(
            batch, views * feature_height * feature_width, token_channels
        )
        memory_parts = [cnn_memory]
        memory_valid_parts = [
            torch.ones(cnn_memory.shape[:2], dtype=torch.bool, device=images.device)
        ]
        semantic_seed = cnn_memory.mean(dim=1)
        open_semantic_embedding = None
        if self.open_semantic_backbone is not None:
            open_features = self.open_semantic_backbone(
                images.reshape(batch * views, channels, height, width)
            )
            open_tokens = open_features["tokens"].reshape(batch, views, -1, token_channels)
            open_tokens = (
                open_tokens
                + self.view_embedding.view(1, views, 1, token_channels)
                + self.source_embedding[1].view(1, 1, 1, token_channels)
            )
            open_tokens = open_tokens.reshape(batch, -1, token_channels)
            open_token_mask = open_features["token_mask"].reshape(batch, -1)
            memory_parts.append(open_tokens)
            memory_valid_parts.append(open_token_mask)
            open_global = open_features["global"].reshape(batch, views, token_channels).mean(dim=1)
            semantic_seed = semantic_seed + open_global
            raw_global = open_features["raw_global"].reshape(batch, views, -1)
            open_semantic_view_embeddings = F.normalize(raw_global.float(), dim=-1)
            open_semantic_embedding = F.normalize(raw_global.mean(dim=1).float(), dim=-1)

        memory = torch.cat(memory_parts, dim=1)
        memory_valid = torch.cat(memory_valid_parts, dim=1)
        semantic = self.semantic_bottleneck(semantic_seed)
        queries = self.uv_queries.unsqueeze(0).expand(batch, -1, -1)
        queries = queries + semantic.unsqueeze(1)
        for block in self.query_blocks:
            queries = block(queries, memory, memory_valid)

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
