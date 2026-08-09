"""Atlas-level semantic targets used by Dense UV Parser."""

import torch

from SkingToolkit.dense_uv_parser.uv_layout import (
    FACE_COUNT,
    build_part_layer_masks,
    minecraft_layer_rects,
)


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


def build_head_outer_face_targets(target_uv, alpha_threshold=0.5):
    """Return exact 8x8 occupancy, presence and coverage for six head faces."""
    if target_uv.dim() != 4 or target_uv.shape[1] != 4:
        raise ValueError(
            "Expected target UV shaped Bx4x64x64, got "
            f"{tuple(target_uv.shape)}."
        )
    occupancy = []
    for inner_x, inner_y, width, height, decor_dx, decor_dy in (
        minecraft_layer_rects()[:FACE_COUNT]
    ):
        if (width, height) != (8, 8):
            raise ValueError("All head faces must be 8x8 texels.")
        outer_x = inner_x + decor_dx
        outer_y = inner_y + decor_dy
        occupancy.append(
            target_uv[
                :, 3, outer_y : outer_y + height, outer_x : outer_x + width
            ]
            > float(alpha_threshold)
        )
    occupancy = torch.stack(occupancy, dim=1).float()
    coverage = occupancy.mean(dim=(-2, -1))
    return {
        "occupancy": occupancy,
        "presence": (coverage > 0.0).float(),
        "coverage": coverage,
    }


__all__ = [
    "build_part_layer_masks",
    "build_head_outer_face_targets",
    "build_semantic_attribute_targets",
]
