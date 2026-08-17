"""Differentiable Analysis-by-Synthesis Hypothesis Refiner.

Validates and purifies outer vs inner skin layers by testing candidate UV
hypotheses against original input renders using DifferentiableRenderer.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from SkingToolkit.dense_uv_parser.uv_layout import (
    LAYER_COUNT,
    PART_COUNT,
    UV_SIZE,
)
from SkingToolkit.dense_uv_parser.uv_topology import build_simple_uv_topology
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
    """Compute differentiable rendering loss against input multi-view images.

    Args:
        renderer: DifferentiableRenderer instance.
        skin_rgba: (B, 4, 64, 64) skin tensor in [0, 1].
        input_renders: (B * len(views), 3 or 4, H, W) or (B, len(views), 3 or 4, H, W).
        views: list of registered view names.
        bg_color: RGB background color.
    Returns:
        Scalar total loss per batch item (B,).
    """
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


def find_outer_connected_components(
    outer_active_mask: torch.Tensor,
    edge_index: torch.Tensor,
    outer_flat_indices: torch.Tensor,
) -> List[torch.Tensor]:
    """Partition active outer texels into connected 3D components.

    Args:
        outer_active_mask: (4096,) boolean mask of active outer texels.
        edge_index: (2, E) outer graph edges within the outer atlas.
        outer_flat_indices: (N_outer,) flat indices of outer atlas nodes.
    Returns:
        List of 1D tensors, each containing the flat indices for one component.
    """
    device = outer_active_mask.device
    flat_to_node = torch.full((UV_SIZE * UV_SIZE,), -1, dtype=torch.long, device=device)
    flat_to_node[outer_flat_indices] = torch.arange(outer_flat_indices.numel(), device=device)

    active_nodes = flat_to_node[outer_active_mask.nonzero(as_tuple=False).flatten()]
    active_nodes = active_nodes[active_nodes >= 0]

    if active_nodes.numel() == 0:
        return []

    visited = torch.zeros(outer_flat_indices.numel(), dtype=torch.bool, device=device)
    active_set = torch.zeros(outer_flat_indices.numel(), dtype=torch.bool, device=device)
    active_set[active_nodes] = True

    adj: Dict[int, List[int]] = {}
    edges_cpu = edge_index.cpu()
    for e in range(edges_cpu.shape[1]):
        u, v = int(edges_cpu[0, e]), int(edges_cpu[1, e])
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    components = []
    for start_node in active_nodes.tolist():
        if bool(visited[start_node]):
            continue
        comp_nodes = []
        queue = [start_node]
        visited[start_node] = True

        while queue:
            curr = queue.pop(0)
            comp_nodes.append(curr)
            for neighbor in adj.get(curr, []):
                if bool(active_set[neighbor]) and not bool(visited[neighbor]):
                    visited[neighbor] = True
                    queue.append(neighbor)

        comp_flat = outer_flat_indices[torch.tensor(comp_nodes, dtype=torch.long, device=device)]
        components.append(comp_flat)

    return components


def refine_uv_by_analysis_by_synthesis(
    skin_uv: torch.Tensor,
    input_renders: torch.Tensor,
    renderer: DifferentiableRenderer,
    views: List[str],
    alpha_threshold: float = 0.5,
    min_improvement_margin: float = 0.0005,
    protect_chin_occlusion: bool = True,
    chin_max_v: float = 0.625,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Arbitrate outer vs inner skin hypotheses via differentiable re-rendering.

    Args:
        skin_uv: (B, 4, 64, 64) or (4, 64, 64) skin tensor with candidate outer layer.
        input_renders: (B * V, 3 or 4, H, W) or (B, V, 3 or 4, H, W) original input images.
        renderer: DifferentiableRenderer instance.
        views: List of view names (e.g. ['front_left', 'back_left']).
        alpha_threshold: Cutoff for active alpha.
        min_improvement_margin: Required error improvement for outer components.
        protect_chin_occlusion: Whether to strictly veto outer head patches occluding face.
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

    topology = build_simple_uv_topology()
    valid = topology.valid.to(device=device)
    layer = topology.layer.to(device=device)
    part = topology.part.to(device=device)
    face = topology.face.to(device=device)
    local_uv = topology.local_uv.to(device=device)
    edge_index = topology.outer_edge_index.to(device=device)
    outer_flat_indices = topology.outer_flat_indices.to(device=device)

    refined = skin_uv.clone().to(device=device, dtype=dtype)

    total_tested_components = 0
    accepted_outer_components = 0
    rejected_outer_components = 0
    chin_protected_texels = 0

    with torch.no_grad():
        for b in range(B):
            current_skin = refined[b : b + 1]
            renders_b = input_renders[b : b + 1] if input_renders.dim() == 5 else input_renders[b * len(views) : (b + 1) * len(views)].unsqueeze(0)

            flat_current = current_skin.reshape(4, UV_SIZE * UV_SIZE)
            active_outer = (flat_current[3] > alpha_threshold) & (layer == 1) & valid

            # 1. Chin / Lower Face Occlusion Protection
            if protect_chin_occlusion:
                chin_outer_mask = (
                    active_outer
                    & (part == 0)
                    & (face == 0)
                    & (local_uv[:, 1] >= chin_max_v)
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

            # 2. Component-Level Hypothesis Testing
            components = find_outer_connected_components(
                active_outer,
                edge_index,
                outer_flat_indices,
            )

            current_loss = compute_multiview_render_loss(renderer, current_skin, renders_b, views)

            for comp_flat in components:
                total_tested_components += 1
                test_skin = current_skin.clone()
                flat_test = test_skin.reshape(4, UV_SIZE * UV_SIZE)
                flat_test[:, comp_flat] = 0.0

                loss_without = compute_multiview_render_loss(renderer, test_skin, renders_b, views)
                improvement = loss_without.item() - current_loss.item()

                if improvement >= -min_improvement_margin:
                    accepted_outer_components += 1
                else:
                    flat_current[:, comp_flat] = 0.0
                    rejected_outer_components += 1
                    current_loss = loss_without

            refined[b] = flat_current.reshape(4, UV_SIZE, UV_SIZE)

    outer_transparent = (refined[:, 3:4] <= alpha_threshold) & (layer.view(1, 1, UV_SIZE, UV_SIZE) == 1)
    refined = torch.where(outer_transparent.expand_as(refined), torch.zeros_like(refined), refined)

    stats = {
        "tested_components": total_tested_components,
        "accepted_outer_components": accepted_outer_components,
        "rejected_outer_components": rejected_outer_components,
        "chin_protected_texels": chin_protected_texels,
    }

    if squeeze_batch:
        return refined[0], stats
    return refined, stats
