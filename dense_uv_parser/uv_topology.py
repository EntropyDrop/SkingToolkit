"""Deterministic Minecraft UV topology needed by parser post-processing."""

from dataclasses import dataclass
from functools import lru_cache

import torch

from SkingToolkit.dense_uv_parser.uv_layout import (
    FACE_COUNT,
    LAYER_COUNT,
    PART_COUNT,
    UV_SIZE,
    minecraft_layer_rects,
)


INVALID_LAYER = LAYER_COUNT
INVALID_PART = PART_COUNT
INVALID_FACE = FACE_COUNT
MIRRORED_PART = (0, 1, 3, 2, 5, 4)
PART_CENTRES = (
    (0.0, 28.0, 0.0),
    (0.0, 18.0, 0.0),
    (6.0, 18.0, 0.0),
    (-6.0, 18.0, 0.0),
    (2.0, 6.0, 0.0),
    (-2.0, 6.0, 0.0),
)


@dataclass(frozen=True)
class SimpleUVTopology:
    valid: torch.Tensor
    layer: torch.Tensor
    part: torch.Tensor
    face: torch.Tensor
    local_uv: torch.Tensor
    world_position: torch.Tensor
    mirrored_texel: torch.Tensor
    inner_fill_order: torch.Tensor


def _axis_coordinate(index, size):
    return -0.5 * float(size) + float(index) + 0.5


def _surface_coordinate(face, u, v, width, height, depth):
    x = _axis_coordinate(u, width)
    y = _axis_coordinate(v, height)
    z_from_front = 0.5 * depth - float(u) - 0.5
    z_from_back = -0.5 * depth + float(u) + 0.5
    if face == 0:
        return (x, y, 0.5 * depth)
    if face == 1:
        return (-x, y, -0.5 * depth)
    if face == 2:
        return (0.5 * width, y, z_from_front)
    if face == 3:
        return (-0.5 * width, y, z_from_back)
    if face == 4:
        return (x, -0.5 * height, -0.5 * depth + float(v) + 0.5)
    if face == 5:
        return (x, 0.5 * height, -0.5 * depth + float(v) + 0.5)
    raise ValueError(f"Unknown face index {face}.")


def _flat_index(x, y):
    return int(y) * UV_SIZE + int(x)


def _inward_ring_key(cell, width, height):
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


@lru_cache(maxsize=1)
def build_simple_uv_topology():
    """Build only the metadata consumed by deterministic inner-layer repair."""
    valid = torch.zeros(UV_SIZE * UV_SIZE, dtype=torch.bool)
    layer_map = torch.full((UV_SIZE * UV_SIZE,), INVALID_LAYER, dtype=torch.long)
    part_map = torch.full((UV_SIZE * UV_SIZE,), INVALID_PART, dtype=torch.long)
    face_map = torch.full((UV_SIZE * UV_SIZE,), INVALID_FACE, dtype=torch.long)
    local_uv = torch.zeros(UV_SIZE * UV_SIZE, 2, dtype=torch.float32)
    world_position = torch.zeros(UV_SIZE * UV_SIZE, 3, dtype=torch.float32)
    cells_by_part = {}

    rects = minecraft_layer_rects(is_slim=False)
    for layer in range(LAYER_COUNT):
        for part in range(PART_COUNT):
            part_rects = rects[part * FACE_COUNT : (part + 1) * FACE_COUNT]
            width = part_rects[0][2]
            height = part_rects[0][3]
            depth = part_rects[2][2]
            cells = []
            for face, (
                x,
                y,
                face_width,
                face_height,
                decor_dx,
                decor_dy,
            ) in enumerate(part_rects):
                if layer == 1:
                    x += decor_dx
                    y += decor_dy
                for v in range(face_height):
                    for u in range(face_width):
                        flat = _flat_index(x + u, y + v)
                        if valid[flat]:
                            raise ValueError(
                                f"Overlapping Minecraft atlas rectangles at {(x + u, y + v)}."
                            )
                        valid[flat] = True
                        layer_map[flat] = layer
                        part_map[flat] = part
                        face_map[flat] = face
                        local_uv[flat, 0] = (u + 0.5) / max(face_width, 1)
                        local_uv[flat, 1] = (v + 0.5) / max(face_height, 1)
                        coordinate = _surface_coordinate(
                            face, u, v, width, height, depth
                        )
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
                        cells.append(
                            {
                                "flat": flat,
                                "face": face,
                                "u": u,
                                "v": v,
                            }
                        )
            cells_by_part[(layer, part)] = cells

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

    inner_fill_order = []
    for part in range(PART_COUNT):
        for face in range(FACE_COUNT):
            cells = [
                cell
                for cell in cells_by_part[(0, part)]
                if cell["face"] == face
            ]
            width = max(cell["u"] for cell in cells) + 1
            height = max(cell["v"] for cell in cells) + 1
            if face in (2, 3):
                cells.sort(
                    key=lambda cell: (
                        cell["v"],
                        min(cell["u"], width - 1 - cell["u"]),
                        0 if cell["u"] < width - 1 - cell["u"] else 1,
                    )
                )
            else:
                cells.sort(
                    key=lambda cell: _inward_ring_key(cell, width, height)
                )
            inner_fill_order.extend(cell["flat"] for cell in cells)

    return SimpleUVTopology(
        valid=valid.reshape(UV_SIZE, UV_SIZE),
        layer=layer_map.reshape(UV_SIZE, UV_SIZE),
        part=part_map.reshape(UV_SIZE, UV_SIZE),
        face=face_map.reshape(UV_SIZE, UV_SIZE),
        local_uv=local_uv.reshape(UV_SIZE, UV_SIZE, 2),
        world_position=world_position.reshape(UV_SIZE, UV_SIZE, 3),
        mirrored_texel=mirrored.reshape(UV_SIZE, UV_SIZE),
        inner_fill_order=torch.tensor(inner_fill_order, dtype=torch.long),
    )
