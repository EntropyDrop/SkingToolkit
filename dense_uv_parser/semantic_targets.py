"""Atlas-level semantic targets used by Dense UV Parser."""

import torch

from SkingToolkit.dense_uv_parser.uv_layout import build_part_layer_masks


def build_semantic_attribute_targets(target_uv, inner_part_masks, outer_part_masks):
    alpha = target_uv[:, 3:4].float()
    rgb = target_uv[:, :3].float()
    outer_masks = outer_part_masks.unsqueeze(0).to(device=target_uv.device)
    outer_weight = alpha.unsqueeze(1) * outer_masks
    outer_area = outer_masks.sum(dim=(2, 3, 4)).clamp_min(1.0)
    outer_coverage = outer_weight.sum(dim=(2, 3, 4)) / outer_area
    outer_presence = (outer_coverage > 0.0).float()

    all_masks = torch.cat([inner_part_masks, outer_part_masks], dim=0)
    all_masks = all_masks.unsqueeze(0).to(device=target_uv.device)
    color_weight = alpha.unsqueeze(1) * all_masks
    color_denominator = color_weight.sum(dim=(3, 4)).clamp_min(1.0)
    part_colors = (
        rgb.unsqueeze(1) * color_weight
    ).sum(dim=(3, 4)) / color_denominator
    color_known = color_weight.sum(dim=(2, 3, 4)) > 0.0
    return {
        "outer_presence": outer_presence,
        "outer_coverage": outer_coverage,
        "part_colors": part_colors,
        "part_colors_known": color_known,
    }


__all__ = ["build_part_layer_masks", "build_semantic_attribute_targets"]
