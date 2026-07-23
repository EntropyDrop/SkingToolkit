"""Runtime and cached frozen semantics for Dense UV Parser."""

from pathlib import Path

import torch

from SkingToolkit.semantic_uv_reconstruction.semantic_backbone import (
    SigLIP2VisionBackbone,
)


def build_siglip_runtime(
    model_name,
    device,
    semantic_channels=128,
    local_files_only=False,
):
    backbone = SigLIP2VisionBackbone(
        model_name=model_name,
        token_channels=semantic_channels,
        local_files_only=local_files_only,
    ).to(device).eval()
    backbone.requires_grad_(False)
    return backbone


def attach_siglip_runtime(
    model,
    model_name,
    device,
    local_files_only=False,
    backbone=None,
):
    """Attach a frozen tower without adding it to parser checkpoints."""
    if model.semantic_feature_dim <= 0:
        return None
    if backbone is None:
        backbone = build_siglip_runtime(
            model_name,
            device,
            semantic_channels=model.semantic_channels,
            local_files_only=local_files_only,
        )
    if backbone.raw_feature_dim != model.semantic_feature_dim:
        raise ValueError(
            "SigLIP feature dimension does not match the parser checkpoint: "
            f"backbone={backbone.raw_feature_dim}, parser={model.semantic_feature_dim}."
        )
    # Bypass nn.Module registration: the frozen pretrained tower is referenced
    # by name in model_config and must not be duplicated in every parser state_dict.
    object.__setattr__(model, "_runtime_semantic_backbone", backbone)
    return backbone


def cached_semantic_batch(cache, paths, device):
    if cache is None:
        return None
    features = [cache.get(Path(path).name) for path in paths]
    return torch.stack(features, dim=0).to(device=device, non_blocking=True)
