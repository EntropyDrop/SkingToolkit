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
    side_face_counts = occupancy[:, :4].sum(dim=-1)
    closed_ring_rows = (
        (side_face_counts >= 4.0).all(dim=1)
        & (side_face_counts.sum(dim=1) >= 24.0)
    )
    top_face = occupancy[:, 5]
    top_perimeter_mask = torch.zeros(
        8, 8, dtype=torch.bool, device=occupancy.device
    )
    top_perimeter_mask[[0, -1], :] = True
    top_perimeter_mask[:, [0, -1]] = True
    top_perimeter_count = top_face[:, top_perimeter_mask].sum(dim=1)
    top_interior_count = top_face[:, ~top_perimeter_mask].sum(dim=1)
    closed_side_ring = closed_ring_rows.any(dim=1)
    open_top_rim = (
        closed_side_ring
        & (top_perimeter_count >= 12.0)
        & (top_interior_count <= 12.0)
    )
    return {
        "occupancy": occupancy,
        "presence": (coverage > 0.0).float(),
        "coverage": coverage,
        "closed_ring_rows": closed_ring_rows.float(),
        "closed_side_ring": closed_side_ring.float(),
        "open_top_rim": open_top_rim.float(),
    }


def head_outer_face_values_to_uv(values):
    """Scatter face-major Bx6x8x8 head values into a 64x64 atlas."""
    if values.dim() != 4 or values.shape[1:] != (FACE_COUNT, 8, 8):
        raise ValueError(
            "Expected head face values shaped Bx6x8x8, got "
            f"{tuple(values.shape)}."
        )
    atlas = values.new_zeros(values.shape[0], 1, 64, 64)
    for face, (inner_x, inner_y, width, height, decor_dx, decor_dy) in enumerate(
        minecraft_layer_rects()[:FACE_COUNT]
    ):
        if (width, height) != (8, 8):
            raise ValueError("All head faces must be 8x8 texels.")
        outer_x = inner_x + decor_dx
        outer_y = inner_y + decor_dy
        atlas[
            :, :, outer_y : outer_y + height, outer_x : outer_x + width
        ] = values[:, face : face + 1]
    return atlas


def build_dense_view_semantic_targets(
    target_uv,
    renderer,
    views,
    device=None,
    alpha_threshold=0.5,
):
    """Build exact 2D pixel-level multi-class semantic ground truth (0..14).

    Returns:
        semantic_targets: (B * V, H, W) long tensor with class indices 0..14.
    """
    from SkingToolkit.dense_uv_parser.utils import (
        build_static_surface_routing,
        parse_views,
    )

    B = target_uv.shape[0]
    parsed_views = parse_views(views)
    if device is None:
        device = target_uv.device

    targets = []
    for b in range(B):
        uv_b = target_uv[b : b + 1]
        flat_alpha = uv_b[0, 3].flatten()
        for v_name in parsed_views:
            static = build_static_surface_routing(renderer, v_name, device)
            H, W = static["masks"].shape[-2:]
            sem = torch.full((H, W), 14, dtype=torch.long, device=device)

            # 1. Inner Layer Base Classification
            inner_mask = static["masks"][0]
            inner_part = static["part"][0]
            inner_face = static["face"][0]
            inner_flat_uv = static["flat_uv"][0]
            inner_y = inner_flat_uv // 64

            # On head front face (part 0, face 0): rows 2..7 are face features (eyes/mouth)
            is_head_front = inner_mask & (inner_part == 0) & (inner_face == 0)
            inner_face_row = inner_y - 8
            is_inner_face_features = is_head_front & (inner_face_row >= 2)
            is_inner_face_hairline = is_head_front & (inner_face_row < 2)
            is_inner_head_other = inner_mask & (inner_part == 0) & (inner_face != 0)
            is_inner_torso = inner_mask & (inner_part == 1)
            is_inner_limbs = inner_mask & (inner_part >= 2)

            sem[is_inner_face_features] = 8  # inner_face (eyes, mouth, facial skin)
            sem[is_inner_face_hairline] = 9  # inner_hair (forehead hairline)
            sem[is_inner_head_other] = 9     # inner_hair (scalp, sides, back)
            sem[is_inner_torso] = 11         # inner_clothes
            sem[is_inner_limbs] = 10         # inner_skin / limbs

            # 2. Outer Layer 3D Decor Classification (only where alpha > alpha_threshold)
            outer_mask = static["masks"][1]
            outer_part = static["part"][1]
            outer_face = static["face"][1]
            outer_flat_uv = static["flat_uv"][1]
            outer_y = outer_flat_uv // 64

            outer_active = outer_mask & (flat_alpha[outer_flat_uv] > float(alpha_threshold))

            # Head outer breakdown:
            is_head_outer = outer_active & (outer_part == 0)

            # Crown / hat: top face (5) or top forehead hairline (outer_y <= 8)
            is_outer_crown = is_head_outer & ((outer_face == 5) | (outer_y <= 8))
            # Glasses / goggles: eye/sunglasses level (rows 9..13) across Front, Right, Left faces
            is_glasses_face = (outer_face == 0) | (outer_face == 2) | (outer_face == 3)
            is_outer_glasses = is_head_outer & is_glasses_face & (outer_y >= 9) & (outer_y <= 13) & ~is_outer_crown
            # Outer hair / lower face (beard, chin, back hair):
            is_outer_head_other = is_head_outer & ~is_outer_glasses & ~is_outer_crown

            is_outer_torso = outer_active & (outer_part == 1)
            is_outer_limbs = outer_active & (outer_part >= 2)

            sem[is_outer_glasses] = 0        # outer_glasses (sunglasses, goggles, temples)
            sem[is_outer_crown] = 1          # outer_crown_hat (crown, hat, tiara)
            sem[is_outer_head_other] = 5     # outer_hair (hair volume, lower face accessories)
            sem[is_outer_torso] = 3          # outer_jacket / hoodie
            sem[is_outer_limbs] = 4          # outer_limbs

            targets.append(sem)

    return torch.stack(targets, dim=0)


__all__ = [
    "build_part_layer_masks",
    "build_head_outer_face_targets",
    "head_outer_face_values_to_uv",
    "build_semantic_attribute_targets",
    "build_dense_view_semantic_targets",
]
