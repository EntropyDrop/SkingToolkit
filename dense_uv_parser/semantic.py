"""Runtime and cached frozen semantics for Dense UV Parser."""

from pathlib import Path

import torch

from SkingToolkit.semantic_uv_reconstruction.semantic_backbone import (
    SigLIP2VisionBackbone,
    TIPSv2VisionBackbone,
)


def build_semantic_runtime(
    backbone_name,
    model_name,
    device,
    semantic_channels=128,
    local_files_only=False,
    runtime_batch_size=32,
):
    if backbone_name == "siglip2":
        backbone = SigLIP2VisionBackbone(
            model_name=model_name,
            token_channels=semantic_channels,
            local_files_only=local_files_only,
        )
    elif backbone_name == "tipsv2":
        backbone = TIPSv2VisionBackbone(
            model_name=model_name,
            local_files_only=local_files_only,
            inference_batch_size=runtime_batch_size,
        )
    else:
        raise ValueError(f"Unsupported semantic backbone: {backbone_name!r}.")
    backbone = backbone.to(device).eval()
    backbone.requires_grad_(False)
    return backbone


def attach_semantic_runtime(
    model,
    backbone_name,
    model_name,
    device,
    local_files_only=False,
    backbone=None,
    runtime_batch_size=32,
):
    """Attach a frozen tower without adding it to parser checkpoints."""
    if (
        model.semantic_feature_dim <= 0
        and getattr(model, "semantic_spatial_feature_dim", 0) <= 0
    ):
        return None
    if backbone is None:
        backbone = build_semantic_runtime(
            backbone_name,
            model_name,
            device,
            semantic_channels=model.semantic_channels,
            local_files_only=local_files_only,
            runtime_batch_size=runtime_batch_size,
        )
    if (
        model.semantic_feature_dim > 0
        and backbone.raw_feature_dim != model.semantic_feature_dim
    ):
        raise ValueError(
            "Semantic feature dimension does not match the parser checkpoint: "
            f"backbone={backbone.raw_feature_dim}, parser={model.semantic_feature_dim}."
        )
    backbone_spatial_dim = int(
        getattr(backbone, "raw_spatial_feature_dim", 0)
    )
    parser_spatial_dim = int(
        getattr(model, "semantic_spatial_feature_dim", 0)
    )
    if parser_spatial_dim > 0 and backbone_spatial_dim != parser_spatial_dim:
        raise ValueError(
            "Semantic spatial feature dimension does not match the parser "
            f"checkpoint: backbone={backbone_spatial_dim}, "
            f"parser={parser_spatial_dim}."
        )
    # Bypass nn.Module registration: the frozen pretrained tower is referenced
    # by name in model_config and must not be duplicated in every parser state_dict.
    object.__setattr__(model, "_runtime_semantic_backbone", backbone)
    return backbone


def build_siglip_runtime(
    model_name,
    device,
    semantic_channels=128,
    local_files_only=False,
    runtime_batch_size=32,
):
    """Backward-compatible wrapper for existing SigLIP2 tooling."""
    return build_semantic_runtime(
        "siglip2",
        model_name,
        device,
        semantic_channels=semantic_channels,
        local_files_only=local_files_only,
        runtime_batch_size=runtime_batch_size,
    )


def attach_siglip_runtime(
    model,
    model_name,
    device,
    local_files_only=False,
    backbone=None,
    runtime_batch_size=32,
):
    """Backward-compatible wrapper for existing SigLIP2 checkpoints."""
    return attach_semantic_runtime(
        model,
        "siglip2",
        model_name,
        device,
        local_files_only=local_files_only,
        backbone=backbone,
        runtime_batch_size=runtime_batch_size,
    )


def cached_semantic_batch(cache, paths, device):
    if cache is None:
        return None
    filenames = [Path(path).name for path in paths]
    global_features = torch.stack(
        [cache.get(filename) for filename in filenames],
        dim=0,
    ).to(device=device, non_blocking=True)
    if not getattr(cache, "has_spatial", False):
        return global_features
    spatial_features = torch.stack(
        [cache.get_spatial(filename) for filename in filenames],
        dim=0,
    ).to(device=device, non_blocking=True)
    return {
        "raw_global": global_features,
        "raw_spatial": spatial_features,
    }
