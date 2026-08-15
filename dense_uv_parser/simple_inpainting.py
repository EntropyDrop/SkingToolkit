"""Deterministic, topology-aware repair of missing inner-layer texels."""

import torch

from SkingToolkit.dense_uv_parser.uv_layout import UV_SIZE
from SkingToolkit.dense_uv_parser.uv_topology import (
    build_head_outer_face_graph,
    build_head_outer_face_indices,
    build_simple_uv_topology,
)


def _complete_head_outer_structure(
    pixels,
    defined,
    valid,
    layer,
    part,
    local_v,
    positions,
    mirrored,
    probability,
    threshold=0.65,
    min_component_seeds=2,
    symmetry_probability=None,
    symmetry_threshold=0.80,
    symmetry_candidate_threshold=0.20,
    closed_ring_probability=None,
    closed_ring_threshold=0.70,
    open_top_probability=None,
    open_top_threshold=0.70,
    open_top_max_gap=3,
):
    """Fill only model-backed head-outer texels anchored to observations."""
    head_indices = build_head_outer_face_indices().to(pixels.device)
    edge_index = build_head_outer_face_graph().to(pixels.device)
    known_nodes = defined.index_select(0, head_indices)
    probability_nodes = probability.reshape(-1).index_select(0, head_indices)
    flat_to_node = torch.full(
        (UV_SIZE * UV_SIZE,), -1, dtype=torch.long, device=pixels.device
    )
    flat_to_node[head_indices] = torch.arange(
        head_indices.numel(), device=pixels.device
    )
    mirrored_nodes = flat_to_node[mirrored.index_select(0, head_indices)]
    safe_mirrored_nodes = mirrored_nodes.clamp_min(0)
    mirrored_known = (
        (mirrored_nodes >= 0)
        & known_nodes.index_select(0, safe_mirrored_nodes)
    )
    use_symmetric_completion = (
        symmetry_probability is not None
        and float(symmetry_probability) >= float(symmetry_threshold)
        and int(known_nodes.sum().item()) >= int(min_component_seeds)
    )
    symmetric_candidates = (
        mirrored_known
        & (probability_nodes >= float(symmetry_candidate_threshold))
        if use_symmetric_completion
        else torch.zeros_like(known_nodes)
    )
    candidates = probability_nodes >= float(threshold)
    candidates = candidates | symmetric_candidates
    active = candidates | known_nodes
    node_count = head_indices.numel()
    labels = torch.where(
        active,
        torch.arange(node_count, device=pixels.device),
        torch.full((node_count,), node_count, device=pixels.device),
    )
    source, destination = edge_index
    for _ in range(64):
        propagated = torch.full_like(labels, node_count)
        propagated.scatter_reduce_(
            0,
            destination,
            labels[source],
            reduce="amin",
            include_self=False,
        )
        updated = torch.where(active, torch.minimum(labels, propagated), labels)
        if torch.equal(updated, labels):
            break
        labels = updated
    seed_count = torch.zeros(node_count + 1, dtype=torch.long, device=pixels.device)
    seed_count.scatter_add_(
        0,
        labels[known_nodes],
        torch.ones_like(labels[known_nodes]),
    )
    accepted = (
        candidates
        & (seed_count[labels.clamp_max(node_count)] >= int(min_component_seeds))
    )
    # A confidently symmetric accessory may contain isolated crown tips or an
    # unseen hat-brim face. They cannot satisfy the ordinary anchored-component
    # rule, but a directly observed mirrored texel supplies both a geometric
    # anchor and an exact color source.
    accepted = accepted | symmetric_candidates
    symmetry_filled = 0
    topology_filled = 0
    for node_index in probability_nodes.argsort(descending=True).tolist():
        if not bool(accepted[node_index]) or bool(known_nodes[node_index]):
            continue
        target_index = int(head_indices[node_index])
        mirror_node = int(mirrored_nodes[node_index])
        if mirror_node >= 0 and bool(known_nodes[mirror_node]):
            source_index = int(head_indices[mirror_node])
            pixels[target_index] = pixels[source_index]
            defined[target_index] = True
            known_nodes[node_index] = True
            symmetry_filled += 1
            continue

        same_component = (
            known_nodes
            & (labels == labels[node_index])
        )
        source_nodes = same_component.nonzero(as_tuple=False).flatten()
        if source_nodes.numel() == 0:
            continue
        target_face = node_index // 64
        if target_face < 4:
            same_row = torch.isclose(
                local_v.index_select(0, head_indices[source_nodes]),
                local_v[target_index],
                rtol=0.0,
                atol=1e-6,
            )
            if same_row.any():
                source_nodes = source_nodes[same_row]
        source_indices = head_indices[source_nodes]
        distance = (
            positions.index_select(0, source_indices) - positions[target_index]
        ).square().sum(dim=1)
        source_index = int(source_indices[distance.argmin()])
        pixels[target_index] = pixels[source_index]
        defined[target_index] = True
        known_nodes[node_index] = True
        topology_filled += 1

    closed_ring_filled = 0
    use_closed_ring = (
        closed_ring_probability is not None
        and float(closed_ring_probability) >= float(closed_ring_threshold)
    )
    if use_closed_ring:
        # Side nodes are face-major 4x8x8. A genuine brim may be invisible on
        # one face because its colour exactly matches the base layer there.
        # Close only rows already anchored on three physical faces, and only
        # on a missing/near-empty fourth face. Column voting prevents a single
        # stray routed texel from growing into a full band.
        for row in range(8):
            side_known = known_nodes[: 4 * 64].reshape(4, 8, 8)
            face_counts = side_known[:, row].sum(dim=1)
            present_faces = face_counts >= 2
            if int(present_faces.sum().item()) < 3:
                continue
            if int(face_counts.sum().item()) < 8:
                continue
            column_votes = side_known[present_faces, row].sum(dim=0)
            voted_columns = column_votes >= 2
            if not voted_columns.any():
                continue
            source_nodes = (
                side_known[:, row]
                & present_faces.unsqueeze(1)
            ).nonzero(as_tuple=False)
            source_nodes = (
                source_nodes[:, 0] * 64
                + row * 8
                + source_nodes[:, 1]
            )
            source_indices = head_indices[source_nodes]
            for face_index in (face_counts <= 1).nonzero(
                as_tuple=False
            ).flatten().tolist():
                for column in voted_columns.nonzero(
                    as_tuple=False
                ).flatten().tolist():
                    node_index = face_index * 64 + row * 8 + column
                    if bool(known_nodes[node_index]):
                        continue
                    target_index = int(head_indices[node_index])
                    distance = (
                        positions.index_select(0, source_indices)
                        - positions[target_index]
                    ).square().sum(dim=1)
                    source_index = int(source_indices[distance.argmin()])
                    pixels[target_index] = pixels[source_index]
                    defined[target_index] = True
                    known_nodes[node_index] = True
                    closed_ring_filled += 1

    open_top_filled = 0
    use_open_top = (
        open_top_probability is not None
        and float(open_top_probability) >= float(open_top_threshold)
    )
    if use_open_top:
        top_face_offset = 5 * 64
        perimeter_nodes = []
        perimeter_nodes.extend(top_face_offset + column for column in range(8))
        perimeter_nodes.extend(
            top_face_offset + row * 8 + 7 for row in range(1, 8)
        )
        perimeter_nodes.extend(
            top_face_offset + 7 * 8 + column
            for column in range(6, -1, -1)
        )
        perimeter_nodes.extend(
            top_face_offset + row * 8 for row in range(6, 0, -1)
        )
        perimeter_nodes = torch.tensor(
            perimeter_nodes, dtype=torch.long, device=pixels.device
        )
        perimeter_known = known_nodes.index_select(0, perimeter_nodes)
        if int(perimeter_known.sum().item()) >= 6:
            # First use exact left/right correspondence on the top face.
            for node_index in perimeter_nodes.tolist():
                if bool(known_nodes[node_index]):
                    continue
                mirror_node = int(mirrored_nodes[node_index])
                if mirror_node < 0 or not bool(known_nodes[mirror_node]):
                    continue
                target_index = int(head_indices[node_index])
                source_index = int(head_indices[mirror_node])
                pixels[target_index] = pixels[source_index]
                defined[target_index] = True
                known_nodes[node_index] = True
                open_top_filled += 1

            # Close only short holes bounded on both sides of the physical top
            # perimeter. This repairs a broken crown rim without turning a few
            # unrelated top pixels into a solid 8x8 cap.
            perimeter_count = int(perimeter_nodes.numel())
            fill_nodes = []
            for start in range(perimeter_count):
                if not bool(known_nodes[int(perimeter_nodes[start])]):
                    continue
                gap = []
                for offset in range(1, int(open_top_max_gap) + 2):
                    candidate = (start + offset) % perimeter_count
                    candidate_node = int(perimeter_nodes[candidate])
                    if bool(known_nodes[candidate_node]):
                        if gap:
                            fill_nodes.extend(gap)
                        break
                    gap.append(candidate_node)
            for node_index in dict.fromkeys(fill_nodes):
                if bool(known_nodes[node_index]):
                    continue
                source_nodes = perimeter_nodes[
                    known_nodes.index_select(0, perimeter_nodes)
                ]
                if source_nodes.numel() == 0:
                    break
                target_index = int(head_indices[node_index])
                source_indices = head_indices[source_nodes]
                distance = (
                    positions.index_select(0, source_indices)
                    - positions[target_index]
                ).square().sum(dim=1)
                source_index = int(source_indices[distance.argmin()])
                pixels[target_index] = pixels[source_index]
                defined[target_index] = True
                known_nodes[node_index] = True
                open_top_filled += 1

    return (
        symmetry_filled,
        topology_filled,
        closed_ring_filled,
        open_top_filled,
    )


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


def simple_symmetry_nearest_inpaint(
    uv,
    alpha_threshold=0.5,
    head_outer_probability=None,
    head_outer_threshold=0.65,
    head_outer_min_component_seeds=2,
    head_outer_symmetry_probability=None,
    head_outer_symmetry_threshold=0.80,
    head_outer_symmetry_candidate_threshold=0.20,
    head_outer_closed_ring_probability=None,
    head_outer_closed_ring_threshold=0.70,
    head_outer_open_top_probability=None,
    head_outer_open_top_threshold=0.70,
    head_outer_open_top_max_gap=3,
):
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
    if head_outer_probability is not None:
        if head_outer_probability.dim() == 2:
            head_outer_probability = head_outer_probability.unsqueeze(0)
        if head_outer_probability.shape != (
            result.shape[0],
            UV_SIZE,
            UV_SIZE,
        ):
            raise ValueError(
                "head_outer_probability must be shaped 64x64 or Bx64x64."
            )
        head_outer_probability = head_outer_probability.to(
            device=device, dtype=torch.float32
        )
    stats = []

    for batch_index in range(result.shape[0]):
        original_defined = valid & (
            result[batch_index, :, 3] > float(alpha_threshold)
        )
        defined = original_defined.clone()
        symmetry_filled = 0
        nearest_filled = 0
        same_row_nearest_filled = 0
        head_outer_symmetry_filled = 0
        head_outer_topology_filled = 0
        head_outer_closed_ring_filled = 0
        head_outer_open_top_filled = 0
        if head_outer_probability is not None:
            (
                head_outer_symmetry_filled,
                head_outer_topology_filled,
                head_outer_closed_ring_filled,
                head_outer_open_top_filled,
            ) = _complete_head_outer_structure(
                result[batch_index],
                defined,
                valid,
                layer,
                part,
                local_v,
                positions,
                mirrored,
                head_outer_probability[batch_index],
                threshold=head_outer_threshold,
                min_component_seeds=head_outer_min_component_seeds,
                symmetry_probability=(
                    head_outer_symmetry_probability[batch_index]
                    if torch.is_tensor(head_outer_symmetry_probability)
                    and head_outer_symmetry_probability.ndim > 0
                    else head_outer_symmetry_probability
                ),
                symmetry_threshold=head_outer_symmetry_threshold,
                symmetry_candidate_threshold=(
                    head_outer_symmetry_candidate_threshold
                ),
                closed_ring_probability=(
                    head_outer_closed_ring_probability[batch_index]
                    if torch.is_tensor(head_outer_closed_ring_probability)
                    and head_outer_closed_ring_probability.ndim > 0
                    else head_outer_closed_ring_probability
                ),
                closed_ring_threshold=head_outer_closed_ring_threshold,
                open_top_probability=(
                    head_outer_open_top_probability[batch_index]
                    if torch.is_tensor(head_outer_open_top_probability)
                    and head_outer_open_top_probability.ndim > 0
                    else head_outer_open_top_probability
                ),
                open_top_threshold=head_outer_open_top_threshold,
                open_top_max_gap=head_outer_open_top_max_gap,
            )
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
                "head_outer_symmetry_filled_texels": (
                    head_outer_symmetry_filled
                ),
                "head_outer_topology_filled_texels": (
                    head_outer_topology_filled
                ),
                "head_outer_closed_ring_filled_texels": (
                    head_outer_closed_ring_filled
                ),
                "head_outer_open_top_filled_texels": (
                    head_outer_open_top_filled
                ),
                "head_outer_symmetry_probability": (
                    round(
                        float(
                            head_outer_symmetry_probability[batch_index]
                            if torch.is_tensor(
                                head_outer_symmetry_probability
                            )
                            and head_outer_symmetry_probability.ndim > 0
                            else head_outer_symmetry_probability
                        ),
                        6,
                    )
                    if head_outer_symmetry_probability is not None
                    else None
                ),
                "head_outer_closed_ring_probability": (
                    round(
                        float(
                            head_outer_closed_ring_probability[batch_index]
                            if torch.is_tensor(
                                head_outer_closed_ring_probability
                            )
                            and head_outer_closed_ring_probability.ndim > 0
                            else head_outer_closed_ring_probability
                        ),
                        6,
                    )
                    if head_outer_closed_ring_probability is not None
                    else None
                ),
                "head_outer_open_top_probability": (
                    round(
                        float(
                            head_outer_open_top_probability[batch_index]
                            if torch.is_tensor(head_outer_open_top_probability)
                            and head_outer_open_top_probability.ndim > 0
                            else head_outer_open_top_probability
                        ),
                        6,
                    )
                    if head_outer_open_top_probability is not None
                    else None
                ),
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
