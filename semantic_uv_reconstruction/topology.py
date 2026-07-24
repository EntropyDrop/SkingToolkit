"""Minecraft UV-atlas topology used by surface-aware completion models.

The ordinary 64x64 atlas layout places unrelated faces next to one another and
separates several faces that meet on the actual cuboid.  This module converts
the fixed Steve atlas into per-texel structural metadata and a four-neighbour
surface graph whose edges continue across cuboid seams.
"""

from dataclasses import dataclass
import math

import torch

from SkingToolkit.semantic_uv_reconstruction.losses import minecraft_layer_rects


UV_SIZE = 64
PART_COUNT = 6
FACE_COUNT = 6
LAYER_COUNT = 2
SURFACE_COUNT = PART_COUNT * FACE_COUNT * LAYER_COUNT
INVALID_LAYER = LAYER_COUNT
INVALID_PART = PART_COUNT
INVALID_FACE = FACE_COUNT
INVALID_SURFACE = SURFACE_COUNT


@dataclass(frozen=True)
class UVTopology:
    valid: torch.Tensor
    layer: torch.Tensor
    part: torch.Tensor
    face: torch.Tensor
    surface: torch.Tensor
    local_uv: torch.Tensor
    neighbours: torch.Tensor
    neighbour_valid: torch.Tensor
    paired_layer_texel: torch.Tensor
    surface_pool: torch.Tensor
    world_position: torch.Tensor
    mirrored_texel: torch.Tensor
    inner_fill_order: torch.Tensor


PART_CENTRES = (
    (0.0, 28.0, 0.0),   # head
    (0.0, 18.0, 0.0),   # body
    (6.0, 18.0, 0.0),   # left arm
    (-6.0, 18.0, 0.0),  # right arm
    (2.0, 6.0, 0.0),    # left leg
    (-2.0, 6.0, 0.0),   # right leg
)
MIRRORED_PART = (0, 1, 3, 2, 5, 4)


def _axis_coordinate(index, size):
    return -0.5 * float(size) + float(index) + 0.5


def _surface_coordinate(face, u, v, width, height, depth):
    """Map one face texel centre to a consistent cuboid-space coordinate."""
    x = _axis_coordinate(u, width)
    y = _axis_coordinate(v, height)
    z_from_front = 0.5 * depth - float(u) - 0.5
    z_from_back = -0.5 * depth + float(u) + 0.5

    if face == 0:  # front
        return (x, y, 0.5 * depth)
    if face == 1:  # back; atlas horizontal direction is mirrored in 3D
        return (-x, y, -0.5 * depth)
    if face == 2:  # right
        return (0.5 * width, y, z_from_front)
    if face == 3:  # left
        return (-0.5 * width, y, z_from_back)
    if face == 4:  # top; bottom edge meets the front face
        return (x, -0.5 * height, -0.5 * depth + float(v) + 0.5)
    if face == 5:  # bottom; bottom edge meets the front face
        return (x, 0.5 * height, -0.5 * depth + float(v) + 0.5)
    raise ValueError(f"Unknown face index {face}.")


def _flat_index(x, y):
    return int(y) * UV_SIZE + int(x)


def build_uv_topology(is_slim=False):
    if is_slim:
        raise ValueError("Topology-aware completion currently supports Steve arms only.")

    valid = torch.zeros(UV_SIZE * UV_SIZE, dtype=torch.bool)
    layer_map = torch.full((UV_SIZE * UV_SIZE,), INVALID_LAYER, dtype=torch.long)
    part_map = torch.full((UV_SIZE * UV_SIZE,), INVALID_PART, dtype=torch.long)
    face_map = torch.full((UV_SIZE * UV_SIZE,), INVALID_FACE, dtype=torch.long)
    surface_map = torch.full((UV_SIZE * UV_SIZE,), INVALID_SURFACE, dtype=torch.long)
    local_uv = torch.zeros(UV_SIZE * UV_SIZE, 2, dtype=torch.float32)
    world_position = torch.zeros(UV_SIZE * UV_SIZE, 3, dtype=torch.float32)
    paired = torch.arange(UV_SIZE * UV_SIZE, dtype=torch.long)

    rects = minecraft_layer_rects(is_slim=False)
    cells_by_group = {}
    lookup = {}

    for layer in range(LAYER_COUNT):
        for part in range(PART_COUNT):
            part_rects = rects[part * FACE_COUNT : (part + 1) * FACE_COUNT]
            width = part_rects[0][2]
            height = part_rects[0][3]
            depth = part_rects[2][2]
            group_cells = []

            for face, (x, y, face_width, face_height, decor_dx, decor_dy) in enumerate(
                part_rects
            ):
                if layer == 1:
                    x += decor_dx
                    y += decor_dy
                surface = layer * PART_COUNT * FACE_COUNT + part * FACE_COUNT + face
                for v in range(face_height):
                    for u in range(face_width):
                        atlas_x = x + u
                        atlas_y = y + v
                        flat = _flat_index(atlas_x, atlas_y)
                        if valid[flat]:
                            raise ValueError(
                                f"Overlapping Minecraft atlas rectangles at {(atlas_x, atlas_y)}."
                            )
                        valid[flat] = True
                        layer_map[flat] = layer
                        part_map[flat] = part
                        face_map[flat] = face
                        surface_map[flat] = surface
                        local_uv[flat, 0] = (u + 0.5) / max(face_width, 1)
                        local_uv[flat, 1] = (v + 0.5) / max(face_height, 1)
                        coordinate = _surface_coordinate(
                            face, u, v, width, height, depth
                        )
                        # The renderer expands the head overlay by one block
                        # and every other outer cuboid by half a block. Scale
                        # texel centres away from the part centre so inner and
                        # outer texels occupy their actual rendered shells.
                        if layer == 1:
                            expansion = 1.0 if part == 0 else 0.5
                            coordinate = tuple(
                                value * (size + expansion) / size
                                for value, size in zip(
                                    coordinate, (width, height, depth)
                                )
                            )
                        centre = PART_CENTRES[part]
                        world_position[flat] = torch.tensor(
                            [
                                coordinate[0] + centre[0],
                                coordinate[1] + centre[1],
                                coordinate[2] + centre[2],
                            ],
                            dtype=torch.float32,
                        )
                        cell = {
                            "flat": flat,
                            "face": face,
                            "u": u,
                            "v": v,
                            "coordinate": coordinate,
                        }
                        group_cells.append(cell)
                        lookup[(layer, part, face, u, v)] = flat
            cells_by_group[(layer, part)] = group_cells

    for part in range(PART_COUNT):
        part_rects = rects[part * FACE_COUNT : (part + 1) * FACE_COUNT]
        for face, (_, _, width, height, _, _) in enumerate(part_rects):
            for v in range(height):
                for u in range(width):
                    inner = lookup[(0, part, face, u, v)]
                    outer = lookup[(1, part, face, u, v)]
                    paired[inner] = outer
                    paired[outer] = inner

    neighbours = torch.arange(UV_SIZE * UV_SIZE, dtype=torch.long).view(-1, 1).repeat(1, 4)
    neighbour_valid = torch.zeros(UV_SIZE * UV_SIZE, 4, dtype=torch.bool)

    for (layer, part), cells in cells_by_group.items():
        del layer, part
        face_lookup = {(cell["face"], cell["u"], cell["v"]): cell["flat"] for cell in cells}
        for cell in cells:
            adjacent = []
            for du, dv in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                candidate = face_lookup.get(
                    (cell["face"], cell["u"] + du, cell["v"] + dv)
                )
                if candidate is not None:
                    adjacent.append(candidate)

            missing = 4 - len(adjacent)
            if missing:
                x, y, z = cell["coordinate"]
                cross_face = []
                for candidate in cells:
                    if candidate["face"] == cell["face"]:
                        continue
                    cx, cy, cz = candidate["coordinate"]
                    distance = math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
                    cross_face.append((distance, candidate["flat"]))
                cross_face.sort(key=lambda item: (item[0], item[1]))
                for _, candidate_flat in cross_face:
                    if candidate_flat not in adjacent:
                        adjacent.append(candidate_flat)
                    if len(adjacent) == 4:
                        break

            if len(adjacent) != 4:
                raise RuntimeError(
                    f"Expected four surface neighbours for texel {cell['flat']}, got {adjacent}."
                )
            neighbours[cell["flat"]] = torch.tensor(adjacent, dtype=torch.long)
            neighbour_valid[cell["flat"]] = True

    surface_pool = torch.zeros(SURFACE_COUNT, UV_SIZE * UV_SIZE, dtype=torch.float32)
    for surface in range(SURFACE_COUNT):
        members = surface_map == surface
        count = int(members.sum())
        if count == 0:
            raise RuntimeError(f"Surface {surface} has no texels.")
        surface_pool[surface, members] = 1.0 / count

    # A horizontal character-space reflection gives an exact correspondence
    # for every Steve texel. Restrict candidates to the matching layer and the
    # mirrored body part so touching cuboids cannot steal the correspondence.
    mirrored = torch.arange(UV_SIZE * UV_SIZE, dtype=torch.long)
    for layer in range(LAYER_COUNT):
        for part in range(PART_COUNT):
            targets = (
                valid & (layer_map == layer) & (part_map == part)
            ).nonzero(as_tuple=False).flatten()
            candidates = (
                valid
                & (layer_map == layer)
                & (part_map == MIRRORED_PART[part])
            ).nonzero(as_tuple=False).flatten()
            reflected = world_position[targets].clone()
            reflected[:, 0] = -reflected[:, 0]
            nearest = torch.cdist(
                reflected, world_position[candidates]
            ).argmin(dim=1)
            mirrored[targets] = candidates[nearest]

    # Deterministic repair stages for every body part:
    #   1. front and back, one rectangular ring at a time from border to centre;
    #   2. left and right, top-to-bottom with each row moving from both seams
    #      toward its centre;
    #   3. top and bottom, retaining the rectangular-ring order.
    # Completing a face before entering the next stage lets only already-defined
    # texels become colour evidence for later faces.
    inner_fill_order = []
    for part in range(PART_COUNT):
        for face in range(FACE_COUNT):
            cells = [
                cell
                for cell in cells_by_group[(0, part)]
                if cell["face"] == face
            ]
            width = max(cell["u"] for cell in cells) + 1
            height = max(cell["v"] for cell in cells) + 1

            def inward_ring_key(cell):
                u = cell["u"]
                v = cell["v"]
                ring = min(u, v, width - 1 - u, height - 1 - v)
                right = width - 1 - ring
                bottom = height - 1 - ring
                if v == ring:
                    edge = 0
                    offset = u - ring
                elif u == right:
                    edge = 1
                    offset = v - ring
                elif v == bottom:
                    edge = 2
                    offset = right - u
                else:
                    edge = 3
                    offset = bottom - v
                return ring, edge, offset

            if face in (2, 3):
                cells.sort(
                    key=lambda cell: (
                        cell["v"],
                        min(cell["u"], width - 1 - cell["u"]),
                        0
                        if cell["u"] < width - 1 - cell["u"]
                        else 1,
                    )
                )
            else:
                cells.sort(key=inward_ring_key)
            inner_fill_order.extend(cell["flat"] for cell in cells)

    return UVTopology(
        valid=valid.reshape(UV_SIZE, UV_SIZE),
        layer=layer_map.reshape(UV_SIZE, UV_SIZE),
        part=part_map.reshape(UV_SIZE, UV_SIZE),
        face=face_map.reshape(UV_SIZE, UV_SIZE),
        surface=surface_map.reshape(UV_SIZE, UV_SIZE),
        local_uv=local_uv.reshape(UV_SIZE, UV_SIZE, 2),
        neighbours=neighbours,
        neighbour_valid=neighbour_valid,
        paired_layer_texel=paired.reshape(UV_SIZE, UV_SIZE),
        surface_pool=surface_pool,
        world_position=world_position.reshape(UV_SIZE, UV_SIZE, 3),
        mirrored_texel=mirrored.reshape(UV_SIZE, UV_SIZE),
        inner_fill_order=torch.tensor(inner_fill_order, dtype=torch.long),
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
    """Choose a defined same-part source, preferring a side-face vertical row."""
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
    """Fill unknown inner texels with symmetry, then 3D nearest colours.

    For every part, front/back are completed first from their borders inward,
    followed by left/right from both row edges toward the centre, then top/bottom
    with the same border-inward ring order. Horizontal symmetry stays on the
    inner layer. Symmetry and nearest-neighbour sources must already have valid
    alpha at the current step; RGB values in still-undefined texels are never
    sampled. The nearest fallback stays within the target body part and may use
    that part's defined outer-layer texels. Newly repaired inner texels become
    defined sources for later texels. The outer layer itself is never filled or
    cleared, and every existing opaque RGBA value is untouched.
    """
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
    topology = build_uv_topology()
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
        device=device, dtype=torch.float32
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

        resolved_inner = defined & valid & (layer == 0)
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
                "unresolved_texels": int(
                    (valid & (layer == 0) & ~resolved_inner).sum().item()
                ),
            }
        )

    result[:, ~valid] = 0.0
    result = result.transpose(1, 2).reshape_as(uv).clamp(0.0, 1.0)
    if squeeze_batch:
        return result[0], stats[0]
    return result, stats
