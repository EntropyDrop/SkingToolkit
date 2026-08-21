"""Deterministic, minimal inpainting of missing Minecraft texels.

Inner-layer blind spots (e.g. armpits, between legs, bottom of head) are repaired
using body symmetry and same-part nearest neighbors.  Outer-layer completion is
normally forbidden.  The one deliberately narrow exception is a missing
head-eye texel that has all three of the following forms of evidence:

* the accessory-level head prediction says the structure is symmetric;
* dense v3 semantics project a strong outer candidate to that exact UV texel;
* either its bilateral counterpart is observed outer, or both semantic
  candidates have observed inner-layer colors at their exact counterparts.

That exception repairs split glasses/headphones without turning ordinary eyes
or arbitrary head pixels into outer-layer skin.  Every other unobserved outer
texel remains 100% transparent (alpha = 0.0, RGB = 0.0).
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

    counterpart_texel = torch.full((UV_SIZE * UV_SIZE,), -1, dtype=torch.long)
    for rect_idx, (ix, iy, w, h, dx, dy) in enumerate(minecraft_layer_rects()):
        p = rect_idx // FACE_COUNT
        f = rect_idx % FACE_COUNT
        for row in range(h):
            for col in range(w):
                in_flat = part_face_rects.get((p, 0, f, row, col))
                out_flat = part_face_rects.get((p, 1, f, row, col))
                if in_flat is not None and out_flat is not None:
                    counterpart_texel[in_flat] = out_flat
                    counterpart_texel[out_flat] = in_flat

    metadata = {
        "valid": valid,
        "layer": layer,
        "part": part,
        "face": face,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "mirrored_texel": mirrored_texel,
        "counterpart_texel": counterpart_texel,
    }
    if device is not None:
        metadata = {k: v.to(device) for k, v in metadata.items()}
    return metadata


def _complete_symmetric_head_eye_outer(
    flat,
    valid,
    layer,
    part,
    face,
    grid_x,
    grid_y,
    mirrored_texel,
    counterpart_texel,
    candidate_probability,
    alpha_threshold,
    candidate_threshold,
    symmetry_probability,
    symmetry_threshold,
):
    """Copy only directly mirrored, model-backed head-eye outer texels.

    Sources are snapshotted before filling so a newly completed texel cannot
    seed a second completion.  Existing outer evidence is mirrored exactly;
    when both sides are strong candidates but both were routed inner, each side
    is promoted from its own corresponding inner texel.  A physical front-face
    row bounded by two such observed symmetric endpoints may fill only the
    texels between them.  This remains a one-row repair rather than graph
    diffusion.
    """
    if candidate_probability is None:
        return 0, 0, 0, 0
    if symmetry_probability is None or float(symmetry_probability) < float(
        symmetry_threshold
    ):
        return 0, 0, 0, 0

    probability = candidate_probability.to(
        device=flat.device,
        dtype=torch.float32,
    ).reshape(-1)
    if probability.numel() != UV_SIZE * UV_SIZE:
        raise ValueError(
            "head_outer_symmetric_candidate_probability must contain "
            f"{UV_SIZE * UV_SIZE} texels, got {probability.numel()}."
        )

    original_defined_outer = (
        valid
        & (layer == 1)
        & (part == 0)
        & (flat[3] > float(alpha_threshold))
    )
    safe_mirror = mirrored_texel.clamp_min(0)
    mirror_is_observed_outer = (
        (mirrored_texel >= 0)
        & original_defined_outer.index_select(0, safe_mirror)
    )
    semantic_candidates = probability >= float(candidate_threshold)
    direct_mirror_candidates = (
        valid
        & (layer == 1)
        & (part == 0)
        & ~original_defined_outer
        & mirror_is_observed_outer
        & semantic_candidates
    )
    safe_counterpart = counterpart_texel.clamp_min(0)
    inner_source_is_observed = (
        (counterpart_texel >= 0)
        & (layer.index_select(0, safe_counterpart) == 0)
        & (flat[3].index_select(0, safe_counterpart) > float(alpha_threshold))
    )
    mirrored_semantic_candidate = (
        (mirrored_texel >= 0)
        & semantic_candidates.index_select(0, safe_mirror)
    )
    mirrored_inner_source_is_observed = (
        (mirrored_texel >= 0)
        & inner_source_is_observed.index_select(0, safe_mirror)
    )
    paired_semantic_candidates = (
        valid
        & (layer == 1)
        & (part == 0)
        & ~original_defined_outer
        & semantic_candidates
        & mirrored_semantic_candidate
        & inner_source_is_observed
        & mirrored_inner_source_is_observed
    )
    paired_semantic_candidates &= ~direct_mirror_candidates
    # A glasses/visor row can be confidently identified at its two protruding
    # endpoints while the lens and bridge pixels between them are projected on
    # the inner plane.  Complete only the bounded span on the physical front
    # face, and only when both endpoints are observed outer, semantically
    # outer, and exact bilateral mirrors of one another.
    front_span_candidates = torch.zeros_like(direct_mirror_candidates)
    front_face = valid & (layer == 1) & (part == 0) & (face == 0)
    symmetric_outer_seeds = (
        front_face
        & original_defined_outer
        & semantic_candidates
        & mirrored_semantic_candidate
        & mirror_is_observed_outer
    )
    for row_value in torch.unique(grid_y[front_face]).tolist():
        row_seeds = symmetric_outer_seeds & (grid_y == int(row_value))
        seed_indices = row_seeds.nonzero(as_tuple=False).flatten()
        if seed_indices.numel() < 2:
            continue
        left = seed_indices[grid_x[seed_indices].argmin()]
        right = seed_indices[grid_x[seed_indices].argmax()]
        if int(mirrored_texel[left]) != int(right):
            continue
        left_x = int(grid_x[left])
        right_x = int(grid_x[right])
        front_span_candidates |= (
            front_face
            & (grid_y == int(row_value))
            & (grid_x >= left_x)
            & (grid_x <= right_x)
            & ~original_defined_outer
            & inner_source_is_observed
        )

    front_span_candidates &= ~direct_mirror_candidates
    front_span_candidates &= ~paired_semantic_candidates
    candidates = (
        direct_mirror_candidates
        | paired_semantic_candidates
        | front_span_candidates
    )
    candidate_count = int(candidates.sum().item())
    if candidate_count == 0:
        return 0, 0, 0, 0

    direct_targets = direct_mirror_candidates.nonzero(
        as_tuple=False
    ).flatten()
    if direct_targets.numel() > 0:
        direct_sources = mirrored_texel.index_select(0, direct_targets)
        flat[:, direct_targets] = flat[:, direct_sources]

    paired_targets = paired_semantic_candidates.nonzero(
        as_tuple=False
    ).flatten()
    if paired_targets.numel() > 0:
        paired_sources = counterpart_texel.index_select(0, paired_targets)
        flat[:, paired_targets] = flat[:, paired_sources]
        flat[3, paired_targets] = 1.0

    span_targets = front_span_candidates.nonzero(as_tuple=False).flatten()
    if span_targets.numel() > 0:
        span_sources = counterpart_texel.index_select(0, span_targets)
        flat[:, span_targets] = flat[:, span_sources]
        flat[3, span_targets] = 1.0
    return (
        candidate_count,
        int(direct_targets.numel()),
        int(paired_targets.numel()),
        int(span_targets.numel()),
    )


def simple_symmetry_nearest_inpaint(
    skin_uv: torch.Tensor,
    alpha_threshold: float = 0.5,
    head_outer_probability=None,
    head_outer_threshold=0.65,
    head_outer_min_component_seeds=2,
    head_outer_symmetric_candidate_probability=None,
    head_outer_symmetry_probability=None,
    head_outer_symmetry_threshold=0.80,
    head_outer_symmetry_candidate_threshold=0.65,
    head_outer_closed_ring_probability=None,
    head_outer_closed_ring_threshold=0.70,
    head_outer_open_top_probability=None,
    head_outer_open_top_threshold=0.70,
    head_outer_open_top_max_gap=3,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Clean, deterministic repair of inner holes and mirrored eye accessories.

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
    face = meta["face"]
    grid_x = meta["grid_x"]
    grid_y = meta["grid_y"]
    mirrored_texel = meta["mirrored_texel"]
    counterpart_texel = meta["counterpart_texel"]

    flat = skin_uv.clone().reshape(4, UV_SIZE * UV_SIZE)
    rgb = flat[:3]
    alpha = flat[3]

    # Inner layer masks
    is_inner = (layer == 0) & valid
    is_outer = (layer == 1) & valid

    defined_inner = is_inner & (alpha > alpha_threshold)
    missing_inner = is_inner & ~defined_inner

    (
        symmetric_candidate_count,
        head_outer_direct_mirror_filled,
        head_outer_paired_semantic_filled,
        head_outer_front_span_filled,
    ) = (
        _complete_symmetric_head_eye_outer(
            flat,
            valid,
            layer,
            part,
            face,
            grid_x,
            grid_y,
            mirrored_texel,
            counterpart_texel,
            head_outer_symmetric_candidate_probability,
            alpha_threshold=alpha_threshold,
            candidate_threshold=head_outer_symmetry_candidate_threshold,
            symmetry_probability=head_outer_symmetry_probability,
            symmetry_threshold=head_outer_symmetry_threshold,
        )
    )

    stats = {
        "missing_inner_texels": int(missing_inner.sum().item()),
        "symmetry_filled": 0,
        "nearest_filled": 0,
        "head_outer_symmetric_candidate_texels": symmetric_candidate_count,
        "head_outer_symmetry_filled_texels": (
            head_outer_direct_mirror_filled
            + head_outer_paired_semantic_filled
            + head_outer_front_span_filled
        ),
        "head_outer_direct_mirror_filled_texels": (
            head_outer_direct_mirror_filled
        ),
        "head_outer_paired_semantic_filled_texels": (
            head_outer_paired_semantic_filled
        ),
        "head_outer_front_span_filled_texels": (
            head_outer_front_span_filled
        ),
        "head_outer_symmetry_probability": (
            round(float(head_outer_symmetry_probability), 6)
            if head_outer_symmetry_probability is not None
            else None
        ),
        "outer_completion_policy": (
            "direct_mirror_paired_or_bounded_front_eye_candidates_only"
        ),
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
