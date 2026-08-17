"""Deterministic, minimal inpainting of missing inner-layer Minecraft texels.

Only inner-layer blind spots (e.g. armpits, between legs, bottom of head) are
repaired using body symmetry and same-part nearest neighbors.
Outer-layer texels are NEVER filled or expanded: unobserved outer texels remain
100% transparent (alpha = 0.0, RGB = 0.0).
"""

from functools import lru_cache
from typing import Dict, Tuple
import torch

from SkingToolkit.dense_uv_parser.uv_layout import (
    FACE_COUNT,
    PART_COUNT,
    UV_SIZE,
    finalize_minecraft_alpha,
    minecraft_layer_rects,
)


@lru_cache(maxsize=1)
def build_basic_minecraft_metadata(device=None):
    """Build atlas metadata tensors directly from Minecraft cuboid rectangles."""
    valid = torch.zeros(UV_SIZE * UV_SIZE, dtype=torch.bool)
    layer = torch.zeros(UV_SIZE * UV_SIZE, dtype=torch.long)
    part = torch.full((UV_SIZE * UV_SIZE,), -1, dtype=torch.long)
    face = torch.full((UV_SIZE * UV_SIZE,), -1, dtype=torch.long)
    grid_x = torch.zeros(UV_SIZE * UV_SIZE, dtype=torch.long)
    grid_y = torch.zeros(UV_SIZE * UV_SIZE, dtype=torch.long)
    mirrored_texel = torch.full((UV_SIZE * UV_SIZE,), -1, dtype=torch.long)

    # Face layout: 0:Front, 1:Back, 2:Right, 3:Left, 4:Bottom, 5:Top
    # Symmetric face mapping:
    # Front(0) <-> Front(0) mirrored horizontally
    # Back(1) <-> Back(1) mirrored horizontally
    # Right(2) <-> Left(3)
    # Bottom(4) <-> Bottom(4) mirrored horizontally
    # Top(5) <-> Top(5) mirrored horizontally
    face_mirror_map = {0: 0, 1: 1, 2: 3, 3: 2, 4: 4, 5: 5}

    part_face_rects = {}
    for rect_idx, (ix, iy, w, h, dx, dy) in enumerate(minecraft_layer_rects()):
        p = rect_idx // FACE_COUNT
        f = rect_idx % FACE_COUNT

        # Inner Layer (base)
        for row in range(h):
            for col in range(w):
                x = ix + col
                y = iy + row
                flat = y * UV_SIZE + x
                valid[flat] = True
                layer[flat] = 0
                part[flat] = p
                face[flat] = f
                grid_x[flat] = x
                grid_y[flat] = y
                part_face_rects[(p, 0, f, row, col)] = flat

        # Outer Layer (decor)
        ox = ix + dx
        oy = iy + dy
        for row in range(h):
            for col in range(w):
                x = ox + col
                y = oy + row
                flat = y * UV_SIZE + x
                valid[flat] = True
                layer[flat] = 1
                part[flat] = p
                face[flat] = f
                grid_x[flat] = x
                grid_y[flat] = y
                part_face_rects[(p, 1, f, row, col)] = flat

    # Build exact bilateral mirrored_texel pointers within each part
    for rect_idx, (ix, iy, w, h, dx, dy) in enumerate(minecraft_layer_rects()):
        p = rect_idx // FACE_COUNT
        f = rect_idx % FACE_COUNT
        m_face = face_mirror_map[f]

        for lay in (0, 1):
            for row in range(h):
                for col in range(w):
                    src_flat = part_face_rects.get((p, lay, f, row, col))
                    # Horizontal mirroring across the face
                    m_col = (w - 1) - col
                    m_flat = part_face_rects.get((p, lay, m_face, row, m_col))
                    if src_flat is not None and m_flat is not None:
                        mirrored_texel[src_flat] = m_flat

    metadata = {
        "valid": valid,
        "layer": layer,
        "part": part,
        "face": face,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "mirrored_texel": mirrored_texel,
    }
    if device is not None:
        metadata = {k: v.to(device) for k, v in metadata.items()}
    return metadata


def simple_symmetry_nearest_inpaint(
    skin_uv: torch.Tensor,
    alpha_threshold: float = 0.5,
    **kwargs,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Clean, deterministic repair of missing inner-layer texels only.

    Args:
        skin_uv: (4, 64, 64) tensor with RGB and Alpha in [0, 1].
        alpha_threshold: Alpha threshold to consider a texel defined.
    Returns:
        repaired: (4, 64, 64) skin tensor with complete inner layer.
        stats: Dictionary of inpainting operations.
    """
    device = skin_uv.device
    dtype = skin_uv.dtype
    meta = build_basic_minecraft_metadata(device=device)

    valid = meta["valid"]
    layer = meta["layer"]
    part = meta["part"]
    grid_x = meta["grid_x"]
    grid_y = meta["grid_y"]
    mirrored_texel = meta["mirrored_texel"]

    flat = skin_uv.clone().reshape(4, UV_SIZE * UV_SIZE)
    rgb = flat[:3]
    alpha = flat[3]

    # Inner layer masks
    is_inner = (layer == 0) & valid
    is_outer = (layer == 1) & valid

    defined_inner = is_inner & (alpha > alpha_threshold)
    missing_inner = is_inner & ~defined_inner

    stats = {
        "missing_inner_texels": int(missing_inner.sum().item()),
        "symmetry_filled": 0,
        "nearest_filled": 0,
    }

    if not missing_inner.any():
        # Outer layer transparent pixels are zeroed
        outer_transparent = is_outer & (alpha <= alpha_threshold)
        flat[:, outer_transparent] = 0.0
        # Inner layer always 1.0 alpha
        flat[3, is_inner] = 1.0
        flat[:, ~valid] = 0.0
        return flat.reshape(4, UV_SIZE, UV_SIZE), stats

    # 1. Symmetry Fill on Inner Layer
    missing_indices = missing_inner.nonzero(as_tuple=False).flatten()
    for idx in missing_indices.tolist():
        m_idx = int(mirrored_texel[idx].item())
        if m_idx >= 0 and bool(defined_inner[m_idx]):
            rgb[:, idx] = rgb[:, m_idx]
            defined_inner[idx] = True
            missing_inner[idx] = False
            stats["symmetry_filled"] += 1

    # 2. Same-Part 2D Nearest Neighbor Fill for Remaining Missing Inner Texels
    missing_indices = missing_inner.nonzero(as_tuple=False).flatten()
    for idx in missing_indices.tolist():
        p = int(part[idx].item())
        # Candidate defined inner texels on the same body part
        cand_mask = defined_inner & (part == p)
        if not cand_mask.any():
            # Fallback to any defined inner texel on the whole body
            cand_mask = defined_inner
        if not cand_mask.any():
            # Solid default if absolutely nothing is defined
            rgb[:, idx] = torch.tensor([0.75, 0.60, 0.50], device=device, dtype=dtype)
            defined_inner[idx] = True
            missing_inner[idx] = False
            stats["nearest_filled"] += 1
            continue

        cand_indices = cand_mask.nonzero(as_tuple=False).flatten()
        gx = grid_x[idx].float()
        gy = grid_y[idx].float()
        cand_x = grid_x[cand_indices].float()
        cand_y = grid_y[cand_indices].float()

        dist_sq = (cand_x - gx) ** 2 + (cand_y - gy) ** 2
        best_cand = cand_indices[torch.argmin(dist_sq)]

        rgb[:, idx] = rgb[:, best_cand]
        defined_inner[idx] = True
        missing_inner[idx] = False
        stats["nearest_filled"] += 1

    # 3. Outer Layer Policy:
    # Defined outer texels keep their predicted colors.
    # Unobserved / undefined outer texels are 100% transparent.
    outer_transparent = is_outer & (alpha <= alpha_threshold)
    flat[:, outer_transparent] = 0.0

    # 4. Final Alpha Enforcement
    flat[3, is_inner] = 1.0
    flat[:, ~valid] = 0.0

    repaired = flat.reshape(4, UV_SIZE, UV_SIZE)
    return repaired, stats
