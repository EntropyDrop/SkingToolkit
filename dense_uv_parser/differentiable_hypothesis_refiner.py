"""Differentiable Analysis-by-Synthesis Hypothesis Refiner.

Validates and purifies outer vs inner skin layers by testing candidate UV
hypotheses against original input renders using DifferentiableRenderer.
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
    bg_color: Tuple[float, float, float] = (128 / 255, 128 / 255, 128 / 255),
    rgb_weight: float = 1.0,
    alpha_weight: float = 2.0,
) -> torch.Tensor:
    """Compute differentiable rendering loss against input multi-view images."""
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

        rendered_rgb = rendered[:, :3]
        target_rgb = target[:, :3]

        if target.shape[1] >= 4:
            target_alpha = target[:, 3:4]
            rendered_alpha = rendered[:, 3:4]
            alpha_loss = F.l1_loss(rendered_alpha.float(), target_alpha.float(), reduction="none").mean(dim=(1, 2, 3))
            fg_mask = (target_alpha > 0.1).float()
            rgb_diff = (rendered_rgb.float() - target_rgb.float()).abs() * fg_mask
            rgb_loss = rgb_diff.sum(dim=(1, 2, 3)) / fg_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
            view_loss = rgb_weight * rgb_loss + alpha_weight * alpha_loss
        else:
            rgb_loss = F.l1_loss(rendered_rgb.float(), target_rgb.float(), reduction="none").mean(dim=(1, 2, 3))
            view_loss = rgb_loss

        total_loss = total_loss + view_loss

    return total_loss / max(num_views, 1)


def refine_uv_by_analysis_by_synthesis(
    skin_uv: torch.Tensor,
    input_renders: torch.Tensor,
    renderer: DifferentiableRenderer,
    views: List[str],
    alpha_threshold: float = 0.5,
    min_improvement_margin: float = 0.0005,
    protect_chin_occlusion: bool = True,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Arbitrate outer vs inner skin hypotheses via differentiable re-rendering.

    Args:
        skin_uv: (B, 4, 64, 64) or (4, 64, 64) skin tensor.
        input_renders: (B * V, 3 or 4, H, W) or (B, V, 3 or 4, H, W) input views.
        renderer: DifferentiableRenderer instance.
        views: List of view names (e.g. ['front_left', 'back_left']).
        alpha_threshold: Alpha threshold for active texels.
        min_improvement_margin: Required error improvement for outer layer elements.
        protect_chin_occlusion: Whether to veto outer head patches occluding the chin/face.
    Returns:
        refined_skin: (B, 4, 64, 64) or (4, 64, 64) purified skin tensor.
        stats: Summary statistics.
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

    refined = skin_uv.clone().to(device=device, dtype=dtype)

    total_tested_parts = 0
    accepted_outer_parts = 0
    rejected_outer_parts = 0
    chin_protected_texels = 0

    with torch.no_grad():
        for b in range(B):
            current_skin = refined[b : b + 1]
            renders_b = input_renders[b : b + 1] if input_renders.dim() == 5 else input_renders[b * len(views) : (b + 1) * len(views)].unsqueeze(0)

            flat_current = current_skin.reshape(4, UV_SIZE * UV_SIZE)
            active_outer = (flat_current[3] > alpha_threshold) & (layer == 1) & valid

            # 1. Chin / Lower Face Occlusion Protection
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

                    if loss_no_chin <= loss_with_chin + min_improvement_margin:
                        flat_current[:, chin_outer_mask] = 0.0
                        chin_protected_texels += int(chin_outer_mask.sum().item())
                        active_outer = (flat_current[3] > alpha_threshold) & (layer == 1) & valid

            # 2. Part-wise Outer Layer Verification
            current_loss = compute_multiview_render_loss(renderer, current_skin, renders_b, views)

            for p in range(PART_COUNT):
                part_outer_mask = active_outer & (part == p)
                if not part_outer_mask.any():
                    continue

                total_tested_parts += 1
                test_skin = current_skin.clone()
                flat_test = test_skin.reshape(4, UV_SIZE * UV_SIZE)
                flat_test[:, part_outer_mask] = 0.0

                loss_without = compute_multiview_render_loss(renderer, test_skin, renders_b, views)
                improvement = loss_without.item() - current_loss.item()

                if improvement >= -min_improvement_margin:
                    accepted_outer_parts += 1
                else:
                    flat_current[:, part_outer_mask] = 0.0
                    rejected_outer_parts += 1
                    current_loss = loss_without

            refined[b] = flat_current.reshape(4, UV_SIZE, UV_SIZE)

    outer_transparent = (refined[:, 3:4] <= alpha_threshold) & (layer.view(1, 1, UV_SIZE, UV_SIZE) == 1)
    refined = torch.where(outer_transparent.expand_as(refined), torch.zeros_like(refined), refined)

    stats = {
        "tested_parts": total_tested_parts,
        "accepted_outer_parts": accepted_outer_parts,
        "rejected_outer_parts": rejected_outer_parts,
        "chin_protected_texels": chin_protected_texels,
    }

    if squeeze_batch:
        return refined[0], stats
    return refined, stats
