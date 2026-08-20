"""Atlas-level semantic targets used by Dense UV Parser."""

import torch

from SkingToolkit.dense_uv_parser.uv_layout import (
    FACE_COUNT,
    build_part_layer_masks,
    minecraft_layer_rects,
)
from SkingToolkit.dense_uv_parser.uv_topology import (
    build_head_outer_face_graph,
)


def _graph_component_labels(candidate, edge_index, max_steps=48):
    """Label boolean graph components without creating gradient paths."""
    batch, node_count = candidate.shape
    source, destination = edge_index.to(candidate.device)
    labels = torch.arange(
        node_count, device=candidate.device, dtype=torch.long
    ).view(1, -1).expand(batch, -1).clone()
    labels[~candidate] = node_count
    connected_edge = candidate[:, source] & candidate[:, destination]
    expanded_destination = destination.view(1, -1).expand(batch, -1)
    with torch.no_grad():
        for _ in range(int(max_steps)):
            neighbour_label = torch.where(
                connected_edge,
                labels[:, source],
                torch.full_like(labels[:, source], node_count),
            )
            updated = labels.clone()
            updated.scatter_reduce_(
                1,
                expanded_destination,
                neighbour_label,
                reduce="amin",
                include_self=True,
            )
            if torch.equal(updated, labels):
                break
            labels = updated
    return labels


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


def build_head_top_accessory_face_targets(
    target_uv,
    alpha_threshold=0.5,
    side_seed_rows=2,
):
    """Return exact outer components physically connected to the head top.

    Components are found on the six-face cube graph, not in 2D atlas space.
    A component is selected when any of its true outer-alpha texels touches
    the top face or the upper rows of a side face.  The complete connected
    component is returned, so crown tips and hat sides remain one semantic
    object across UV seams.  Every selected texel is still a real outer-alpha
    texel; no bounding rectangle or transparent gap is fabricated.
    """
    if not 1 <= int(side_seed_rows) <= 8:
        raise ValueError("side_seed_rows must be in [1, 8].")
    occupancy = build_head_outer_face_targets(
        target_uv, alpha_threshold=alpha_threshold
    )["occupancy"] > 0.5
    nodes = occupancy.flatten(1)
    labels = _graph_component_labels(
        nodes,
        build_head_outer_face_graph(),
    )
    node_count = nodes.shape[1]
    seed_template = torch.zeros(
        FACE_COUNT, 8, 8, dtype=torch.bool, device=nodes.device
    )
    seed_template[:4, : int(side_seed_rows)] = True
    seed_template[5] = True
    seed_nodes = nodes & seed_template.flatten().unsqueeze(0)

    component_has_seed = torch.zeros(
        nodes.shape[0],
        node_count + 1,
        dtype=torch.uint8,
        device=nodes.device,
    )
    component_has_seed.scatter_reduce_(
        1,
        labels,
        seed_nodes.to(torch.uint8),
        reduce="amax",
        include_self=True,
    )
    selected = nodes & component_has_seed.gather(1, labels).bool()
    return {
        "mask": selected.reshape(-1, FACE_COUNT, 8, 8).float(),
        "presence": selected.any(dim=1).float(),
        "component_labels": labels,
    }


def build_head_eye_accessory_face_targets(
    target_uv,
    alpha_threshold=0.5,
    eye_seed_row_start=1,
    eye_seed_row_end=6,
):
    """Return exact head-outer alpha in the physical eye-level band.

    The target is intentionally geometric rather than an object-name guess:
    every selected texel must exist in ground-truth outer alpha.  Front,
    left, and right head faces share the same vertical eye band, so glasses,
    goggles, visors, masks, and headset temples receive one consistent class
    across atlas seams.  Hair or helmet alpha crossing this band is harmless:
    it is still genuinely outer-layer evidence and should support the same
    inner/outer route.
    """
    row_start = int(eye_seed_row_start)
    row_end = int(eye_seed_row_end)
    if not 0 <= row_start < row_end <= 8:
        raise ValueError(
            "Require 0 <= eye_seed_row_start < eye_seed_row_end <= 8."
        )
    occupancy = build_head_outer_face_targets(
        target_uv, alpha_threshold=alpha_threshold
    )["occupancy"] > 0.5
    band = torch.zeros_like(occupancy)
    # Face order is front, back, right, left, bottom, top.  The back face is
    # excluded because an eye accessory seen from behind is represented by
    # its side temples, not arbitrary back-hair alpha at the same height.
    band[:, (0, 2, 3), row_start:row_end] = True
    selected = occupancy & band
    return {
        "mask": selected.float(),
        "presence": selected.flatten(1).any(dim=1).float(),
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
    target_version=1,
):
    """Build dense pixel semantics for the requested target schema.

    Returns:
        semantic_targets: (B * V, H, W) long tensor. Version 1 uses the
            legacy 15-class pseudo labels; version 2 uses four exact
            layer/topology classes; version 3 separates exact eye-level head
            outer alpha into a fifth class.
    """
    from SkingToolkit.dense_uv_parser.utils import (
        build_static_surface_routing,
        parse_views,
    )

    B = target_uv.shape[0]
    parsed_views = parse_views(views)
    if device is None:
        device = target_uv.device

    target_version = int(target_version)
    if target_version not in (1, 2, 3):
        raise ValueError("target_version must be 1, 2, or 3.")
    targets = []
    top_accessory_atlas = None
    eye_accessory_atlas = None
    if target_version in (2, 3):
        top_faces = build_head_top_accessory_face_targets(
            target_uv,
            alpha_threshold=alpha_threshold,
        )["mask"]
        top_accessory_atlas = head_outer_face_values_to_uv(top_faces)[
            :, 0
        ].flatten(1).bool()
    if target_version == 3:
        eye_faces = build_head_eye_accessory_face_targets(
            target_uv,
            alpha_threshold=alpha_threshold,
        )["mask"]
        eye_accessory_atlas = head_outer_face_values_to_uv(eye_faces)[
            :, 0
        ].flatten(1).bool()
    for b in range(B):
        uv_b = target_uv[b : b + 1]
        flat_alpha = uv_b[0, 3].flatten()
        for v_name in parsed_views:
            static = build_static_surface_routing(renderer, v_name, device)
            H, W = static["masks"].shape[-2:]
            if target_version in (2, 3):
                # Version 2: 0 head_top_accessory, 1 other_outer,
                # 2 inner, 3 background.
                # Version 3: 0 head_top_accessory,
                # 1 head_eye_accessory, 2 other_outer, 3 inner,
                # 4 background.  Eye-level alpha takes priority where a tall
                # top-connected component overlaps the physical eye band.
                background_class = 4 if target_version == 3 else 3
                other_outer_class = 2 if target_version == 3 else 1
                inner_class = 3 if target_version == 3 else 2
                sem = torch.full(
                    (H, W),
                    background_class,
                    dtype=torch.long,
                    device=device,
                )
                inner_mask = static["masks"][0]
                outer_mask = static["masks"][1]
                outer_part = static["part"][1]
                outer_flat_uv = static["flat_uv"][1]
                outer_alpha = flat_alpha[outer_flat_uv]
                outer_active = outer_mask & (
                    outer_alpha > float(alpha_threshold)
                )
                sem[inner_mask] = inner_class
                sem[outer_active] = other_outer_class
                top_active = (
                    outer_active
                    & (outer_part == 0)
                    & top_accessory_atlas[b, outer_flat_uv]
                )
                sem[top_active] = 0
                if target_version == 3:
                    eye_active = (
                        outer_active
                        & (outer_part == 0)
                        & eye_accessory_atlas[b, outer_flat_uv]
                    )
                    sem[eye_active] = 1
                targets.append(sem)
                continue

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

            outer_alpha = flat_alpha[outer_flat_uv]
            # Use lower alpha threshold (0.05) for head outer layer to capture translucent sunglasses / tinted lenses
            head_outer_active = outer_mask & (outer_part == 0) & (outer_alpha > 0.05)
            body_outer_active = outer_mask & (outer_part > 0) & (outer_alpha > float(alpha_threshold))
            outer_active = head_outer_active | body_outer_active

            # Detect if this skin has outer glasses / goggles on head front or sides
            skin_has_front_glasses = (uv_b[0, 3, 9:14, 40:48] > 0.05).any()
            skin_has_side_glasses = (
                (uv_b[0, 3, 9:14, 32:40] > 0.05).any()
                | (uv_b[0, 3, 9:14, 48:56] > 0.05).any()
            )
            skin_has_glasses = skin_has_front_glasses | skin_has_side_glasses
            is_head_outer = head_outer_active

            # Crown / hat: top face (5) or top forehead hairline (outer_y <= 8)
            is_outer_crown = is_head_outer & ((outer_face == 5) | (outer_y <= 8))
            # Glasses / goggles: eye/sunglasses level (rows 9..13) across Front, Right, Left faces
            is_glasses_face = (outer_face == 0) | (outer_face == 2) | (outer_face == 3)
            is_glasses_box = (
                outer_mask
                & (outer_part == 0)
                & is_glasses_face
                & (outer_y >= 9)
                & (outer_y <= 13)
            )
            is_outer_glasses = (
                (head_outer_active & is_glasses_box)
                | (skin_has_front_glasses & is_glasses_box & (outer_face == 0))
            ) & ~is_outer_crown
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
    "build_head_top_accessory_face_targets",
    "build_head_eye_accessory_face_targets",
    "head_outer_face_values_to_uv",
    "build_semantic_attribute_targets",
    "build_dense_view_semantic_targets",
]
