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
    if face == 5:  # bottom; top edge meets the front face
        return (x, 0.5 * height, 0.5 * depth - float(v) - 0.5)
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
    )
