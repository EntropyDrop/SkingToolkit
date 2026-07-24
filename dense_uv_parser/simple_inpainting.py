"""Deterministic, topology-aware repair of missing inner-layer texels."""

import torch

from SkingToolkit.dense_uv_parser.uv_layout import UV_SIZE
from SkingToolkit.dense_uv_parser.uv_topology import build_simple_uv_topology


def _nearest_defined_source(
    target_index,
    defined,
    valid,
    part,
    face,
    local_v,
    positions,
    prefer_same_row=False,
):
    source_mask = defined & valid & (part == part[target_index])
    used_same_row = False
    if prefer_same_row:
        same_row_mask = (
            source_mask
            & (face < 4)
            & torch.isclose(
                local_v,
                local_v[target_index],
                rtol=0.0,
                atol=1e-6,
            )
        )
        source_indices = same_row_mask.nonzero(as_tuple=False).flatten()
        if source_indices.numel() > 0:
            used_same_row = True
        else:
            source_indices = source_mask.nonzero(as_tuple=False).flatten()
    else:
        source_indices = source_mask.nonzero(as_tuple=False).flatten()
    if source_indices.numel() == 0:
        return None, False
    squared_distance = (
        positions[source_indices] - positions[target_index]
    ).square().sum(dim=1)
    return source_indices[squared_distance.argmin()], used_same_row


def simple_symmetry_nearest_inpaint(uv, alpha_threshold=0.5):
    """Fill unknown inner texels while preserving every outer-layer texel."""
    squeeze_batch = uv.dim() == 3
    if squeeze_batch:
        uv = uv.unsqueeze(0)
    if uv.dim() != 4 or uv.shape[1:] != (4, UV_SIZE, UV_SIZE):
        raise ValueError(
            f"Expected 4x{UV_SIZE}x{UV_SIZE} or Bx4x{UV_SIZE}x{UV_SIZE} UV, "
            f"got {tuple(uv.shape)}."
        )
    if not 0.0 <= alpha_threshold <= 1.0:
        raise ValueError("alpha_threshold must be in [0, 1].")

    topology = build_simple_uv_topology()
    topology_face = topology.face.reshape(-1)
    device = uv.device
    valid = topology.valid.reshape(-1).to(device=device)
    layer = topology.layer.reshape(-1).to(device=device)
    part = topology.part.reshape(-1).to(device=device)
    face = topology.face.reshape(-1).to(device=device)
    local_v = topology.local_uv.reshape(-1, 2)[:, 1].to(
        device=device,
        dtype=torch.float32,
    )
    mirrored = topology.mirrored_texel.reshape(-1).to(device=device)
    positions = topology.world_position.reshape(-1, 3).to(
        device=device,
        dtype=torch.float32,
    )
    result = uv.flatten(2).transpose(1, 2).clone()
    stats = []

    for batch_index in range(result.shape[0]):
        original_defined = valid & (
            result[batch_index, :, 3] > float(alpha_threshold)
        )
        defined = original_defined.clone()
        symmetry_filled = 0
        nearest_filled = 0
        same_row_nearest_filled = 0
        for target_index in topology.inner_fill_order.tolist():
            if bool(defined[target_index]):
                continue
            mirror_index = int(mirrored[target_index])
            if bool(defined[mirror_index]):
                result[batch_index, target_index] = result[
                    batch_index, mirror_index
                ]
                defined[target_index] = True
                symmetry_filled += 1
                continue

            source_index, used_same_row = _nearest_defined_source(
                target_index,
                defined,
                valid,
                part,
                face,
                local_v,
                positions,
                prefer_same_row=int(topology_face[target_index]) in (2, 3),
            )
            if source_index is None:
                continue
            result[batch_index, target_index] = result[
                batch_index, source_index
            ]
            defined[target_index] = True
            nearest_filled += 1
            same_row_nearest_filled += int(used_same_row)

        unresolved_inner = valid & (layer == 0) & ~defined
        stats.append(
            {
                "known_texels": int(original_defined.sum().item()),
                "known_inner_texels": int(
                    (original_defined & (layer == 0)).sum().item()
                ),
                "known_outer_texels": int(
                    (original_defined & (layer == 1)).sum().item()
                ),
                "symmetry_filled_texels": symmetry_filled,
                "nearest_3d_filled_texels": nearest_filled,
                "same_row_nearest_filled_texels": same_row_nearest_filled,
                "preserved_outer_texels": int((valid & (layer == 1)).sum().item()),
                "fill_order": "front_back_rings_side_edges_top_bottom_rings",
                "color_sources": "currently_defined_only",
                "side_nearest_policy": "same_vertical_row_then_same_part_3d",
                "unresolved_texels": int(unresolved_inner.sum().item()),
            }
        )

    result[:, ~valid] = 0.0
    result = result.transpose(1, 2).reshape_as(uv).clamp(0.0, 1.0)
    if squeeze_batch:
        return result[0], stats[0]
    return result, stats
