"""Differentiable Analysis-by-Synthesis Hypothesis Refiner.

Validates and purifies outer vs inner skin layers by testing candidate UV
hypotheses against original input renders using DifferentiableRenderer.
Eliminates monochromatic transparency illusions and false outer decors.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F

from SkingToolkit.dense_uv_parser.simple_inpainting import (
    build_basic_minecraft_metadata,
)
from SkingToolkit.dense_uv_parser.uv_layout import (
    PART_COUNT,
    UV_SIZE,
)
from SkingToolkit.renderer import DifferentiableRenderer


def compute_multiview_render_loss(
    renderer: DifferentiableRenderer,
    skin_rgba: torch.Tensor,
    input_renders: torch.Tensor,
    views: List[str],
    rgb_weight: float = 1.0,
    alpha_weight: float = 3.0,
    overflow_penalty_weight: float = 5.0,
) -> torch.Tensor:
    """Compute differentiable rendering loss with silhouette overflow penalty."""
    B = skin_rgba.shape[0]
    device = skin_rgba.device
    dtype = skin_rgba.dtype
    num_views = len(views)

    if input_renders.dim() == 4 and input_renders.shape[0] == B * num_views:
        target_renders = input_renders.reshape(B, num_views, *input_renders.shape[1:])
    elif input_renders.dim() == 5:
        target_renders = input_renders
    else:
        raise ValueError(
            f"Unexpected input_renders shape {tuple(input_renders.shape)} for B={B}, views={num_views}."
        )

    total_loss = torch.zeros(B, device=device, dtype=torch.float32)

    for v_idx, view_name in enumerate(views):
        rendered = renderer.forward_view(skin_rgba, view_name)
        target = target_renders[:, v_idx].to(device=device, dtype=dtype)

        rendered_rgb = rendered[:, :3].float()
        target_rgb = target[:, :3].float()
        rendered_alpha = rendered[:, 3:4].float() if rendered.shape[1] >= 4 else torch.ones_like(rendered[:, :1]).float()

        if target.shape[1] >= 4:
            target_alpha = target[:, 3:4].float()
            fg_mask = (target_alpha > 0.1).float()
            bg_mask = (target_alpha <= 0.05).float()

            # 1. RGB reconstruction loss inside target character
            rgb_diff = (rendered_rgb - target_rgb).abs() * fg_mask
            rgb_loss = rgb_diff.sum(dim=(1, 2, 3)) / fg_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)

            # 2. General Alpha mismatch loss
            alpha_loss = F.l1_loss(rendered_alpha, target_alpha, reduction="none").mean(dim=(1, 2, 3))

            # 3. Silhouette Overflow Penalty:
            # Outer layer expands the 3D cuboid. If the true image had a flat inner layer,
            # placing a false outer layer causes rendered_alpha to bleed into target background!
            overflow_diff = (rendered_alpha * bg_mask).clamp_min(0.0)
            overflow_loss = overflow_diff.sum(dim=(1, 2, 3)) / bg_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)

            view_loss = (
                rgb_weight * rgb_loss
                + alpha_weight * alpha_loss
                + overflow_penalty_weight * overflow_loss
            )
        else:
            rgb_loss = F.l1_loss(rendered_rgb, target_rgb, reduction="none").mean(dim=(1, 2, 3))
            view_loss = rgb_loss

        total_loss = total_loss + view_loss

    return total_loss / max(num_views, 1)


def _find_outer_connected_components(
    active_mask_2d: torch.Tensor,
) -> List[torch.Tensor]:
    """Find 4-connected components of active outer texels on a 64x64 grid."""
    H, W = active_mask_2d.shape
    visited = torch.zeros_like(active_mask_2d, dtype=torch.bool)
    components = []

    active_indices = active_mask_2d.nonzero(as_tuple=False)
    for idx in range(active_indices.shape[0]):
        r, c = int(active_indices[idx, 0].item()), int(active_indices[idx, 1].item())
        if visited[r, c]:
            continue

        comp_mask = torch.zeros_like(active_mask_2d, dtype=torch.bool)
        queue = [(r, c)]
        visited[r, c] = True
        comp_mask[r, c] = True

        while queue:
            curr_r, curr_c = queue.pop(0)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if active_mask_2d[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        comp_mask[nr, nc] = True
                        queue.append((nr, nc))

        components.append(comp_mask.flatten())

    return components


def refine_uv_by_analysis_by_synthesis(
    skin_uv: torch.Tensor,
    input_renders: torch.Tensor,
    renderer: DifferentiableRenderer,
    views: List[str],
    alpha_threshold: float = 0.5,
    min_required_benefit: float = 0.0003,
    protect_chin_occlusion: bool = True,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Arbitrate outer vs inner skin hypotheses via differentiable re-rendering.

    Default policy: Inner layer is canonical. Outer layer must strictly and
    significantly reduce multi-view rendering error (improving 3D silhouette /
    parallax) to be retained; otherwise it is safely pruned to transparent.

    Args:
        skin_uv: (B, 4, 64, 64) or (4, 64, 64) skin tensor.
        input_renders: (B * V, 3 or 4, H, W) or (B, V, 3 or 4, H, W) input views.
        renderer: DifferentiableRenderer instance.
        views: List of view names (e.g. ['front_left', 'back_left']).
        alpha_threshold: Alpha threshold for active texels.
        min_required_benefit: Minimum error reduction required to justify keeping outer decor.
        protect_chin_occlusion: Whether to veto outer head patches occluding the chin/face.
    Returns:
        refined_skin: (B, 4, 64, 64) or (4, 64, 64) purified skin tensor.
        stats: Summary statistics of accepted / stripped components.
    """
    squeeze_batch = False
    if skin_uv.dim() == 3:
        skin_uv = skin_uv.unsqueeze(0)
        squeeze_batch = True

    B = skin_uv.shape[0]
    device = skin_uv.device
    dtype = skin_uv.dtype

    meta = build_basic_minecraft_metadata(device=device)
    valid = meta["valid"]
    layer = meta["layer"]
    part = meta["part"]
    face = meta["face"]
    grid_y = meta["grid_y"]
    counterpart_texel = meta.get("counterpart_texel")

    refined = skin_uv.clone().to(device=device, dtype=dtype)

    total_tested_components = 0
    accepted_outer_components = 0
    stripped_outer_components = 0
    homogeneity_stripped_texels = 0
    chin_protected_texels = 0

    with torch.no_grad():
        for b in range(B):
            current_skin = refined[b : b + 1]
            renders_b = (
                input_renders[b : b + 1]
                if input_renders.dim() == 5
                else input_renders[b * len(views) : (b + 1) * len(views)].unsqueeze(0)
            )

            flat_current = current_skin.reshape(4, UV_SIZE * UV_SIZE)
            active_outer = (flat_current[3] > alpha_threshold) & (layer == 1) & valid

            # Step 1: Ensure underlying inner base layer always has the observed color.
            # If an outer texel is candidate, mirror its RGB to the inner base texel
            # so stripping the outer layer never leaves an empty hole on the character.
            if counterpart_texel is not None:
                outer_active_idx = active_outer.nonzero(as_tuple=False).flatten()
                for o_idx in outer_active_idx.tolist():
                    i_idx = int(counterpart_texel[o_idx].item())
                    if i_idx >= 0 and flat_current[3, i_idx] <= alpha_threshold:
                        flat_current[:3, i_idx] = flat_current[:3, o_idx]
                        flat_current[3, i_idx] = 1.0

            # Step 2: Chin / Lower Face Occlusion Protection
            # Head front face (part 0, face 0) outer layer: rows 13..15 are chin/mouth
            if protect_chin_occlusion:
                chin_outer_mask = (
                    active_outer
                    & (part == 0)
                    & (face == 0)
                    & (grid_y >= 13)
                )
                if chin_outer_mask.any():
                    test_no_chin = current_skin.clone()
                    flat_test = test_no_chin.reshape(4, UV_SIZE * UV_SIZE)
                    flat_test[:, chin_outer_mask] = 0.0

                    loss_with_chin = compute_multiview_render_loss(renderer, current_skin, renders_b, views)
                    loss_no_chin = compute_multiview_render_loss(renderer, test_no_chin, renders_b, views)

                    # If removing chin outer does not worsen loss by more than tiny tolerance, strip it!
                    if loss_no_chin.item() <= loss_with_chin.item() + min_required_benefit:
                        flat_current[:, chin_outer_mask] = 0.0
                        chin_protected_texels += int(chin_outer_mask.sum().item())
                        active_outer = (flat_current[3] > alpha_threshold) & (layer == 1) & valid

            # Step 3: Same-Color Homogeneity Pruning
            # If an outer texel has identical color to the inner texel beneath it and has no
            # distinct silhouette boundary, it is a monochromatic penetration illusion.
            if counterpart_texel is not None:
                outer_active_idx = active_outer.nonzero(as_tuple=False).flatten()
                for o_idx in outer_active_idx.tolist():
                    i_idx = int(counterpart_texel[o_idx].item())
                    if i_idx >= 0 and flat_current[3, i_idx] > alpha_threshold:
                        color_diff = (flat_current[:3, o_idx] - flat_current[:3, i_idx]).abs().max().item()
                        # If color is identical (< 3/255 difference) on body/limbs without facial features:
                        p_val = int(part[o_idx].item())
                        f_val = int(face[o_idx].item())
                        # Head face=0 (front face) is reserved for glasses/hair testing
                        if color_diff < 0.015 and not (p_val == 0 and f_val == 0):
                            # Test if stripping this single identical texel alters multi-view render
                            test_strip = current_skin.clone()
                            flat_strip = test_strip.reshape(4, UV_SIZE * UV_SIZE)
                            flat_strip[:, o_idx] = 0.0

                            loss_with = compute_multiview_render_loss(renderer, current_skin, renders_b, views)
                            loss_without = compute_multiview_render_loss(renderer, test_strip, renders_b, views)

                            if loss_without.item() <= loss_with.item() + min_required_benefit:
                                flat_current[:, o_idx] = 0.0
                                homogeneity_stripped_texels += 1
                                active_outer[o_idx] = False

            # Step 4: Component-wise Multi-view Hypothesis Testing
            # Group remaining active outer texels into connected components
            active_2d = active_outer.reshape(UV_SIZE, UV_SIZE)
            components = _find_outer_connected_components(active_2d)

            for comp_mask in components:
                total_tested_components += 1

                # Current loss with this component present
                loss_with_comp = compute_multiview_render_loss(renderer, current_skin, renders_b, views)

                # Test loss without this component (stripped to transparent)
                test_skin = current_skin.clone()
                flat_test = test_skin.reshape(4, UV_SIZE * UV_SIZE)
                flat_test[:, comp_mask] = 0.0

                loss_without_comp = compute_multiview_render_loss(renderer, test_skin, renders_b, views)

                # Benefit of having the outer component:
                # benefit > 0 means removing the component makes the loss worse (having it is good!)
                benefit = loss_without_comp.item() - loss_with_comp.item()

                if benefit > min_required_benefit:
                    # Genuine 3D feature (crown peak, glasses, outer ears, jacket edge)
                    accepted_outer_components += 1
                else:
                    # Redundant or false outer layer -> Strip to 100% transparent!
                    flat_current[:, comp_mask] = 0.0
                    stripped_outer_components += 1

            refined[b] = flat_current.reshape(4, UV_SIZE, UV_SIZE)

    # Final Guarantee: All non-accepted outer texels are 100% clean transparent (Alpha = 0, RGB = 0)
    outer_transparent = (refined[:, 3:4] <= alpha_threshold) & (layer.view(1, 1, UV_SIZE, UV_SIZE) == 1)
    refined = torch.where(outer_transparent.expand_as(refined), torch.zeros_like(refined), refined)

    # Base inner layer is 100% opaque
    inner_mask = (layer.view(1, 1, UV_SIZE, UV_SIZE) == 0) & valid.view(1, 1, UV_SIZE, UV_SIZE)
    refined[:, 3:4] = torch.where(inner_mask, torch.ones_like(refined[:, 3:4]), refined[:, 3:4])

    stats = {
        "tested_components": total_tested_components,
        "accepted_outer_components": accepted_outer_components,
        "stripped_outer_components": stripped_outer_components,
        "homogeneity_stripped_texels": homogeneity_stripped_texels,
        "chin_protected_texels": chin_protected_texels,
    }

    if squeeze_batch:
        return refined[0], stats
    return refined, stats
