import sys
from pathlib import Path

import torch
import torch.nn.functional as F

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TOOLKIT_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from SkingToolkit.inverse_uv.losses import minecraft_layer_rects  # noqa: E402

IGNORE_INDEX = 255
PART_CLASSES = 6
FACE_CLASSES = 6
LAYER_CLASSES = 2
UV_SIZE = 64


def parse_views(views):
    if isinstance(views, str):
        return [view.strip() for view in views.split(",") if view.strip()]
    return list(views)


def build_part_face_lookups(device=None):
    inner_part = torch.full((UV_SIZE, UV_SIZE), IGNORE_INDEX, dtype=torch.long, device=device)
    inner_face = torch.full((UV_SIZE, UV_SIZE), IGNORE_INDEX, dtype=torch.long, device=device)
    outer_part = torch.full((UV_SIZE, UV_SIZE), IGNORE_INDEX, dtype=torch.long, device=device)
    outer_face = torch.full((UV_SIZE, UV_SIZE), IGNORE_INDEX, dtype=torch.long, device=device)

    rects = minecraft_layer_rects(is_slim=False)
    for rect_index, (inner_x, inner_y, width, height, decor_dx, decor_dy) in enumerate(rects):
        part = rect_index // FACE_CLASSES
        face = rect_index % FACE_CLASSES
        inner_part[inner_y : inner_y + height, inner_x : inner_x + width] = part
        inner_face[inner_y : inner_y + height, inner_x : inner_x + width] = face

        outer_x = inner_x + decor_dx
        outer_y = inner_y + decor_dy
        outer_part[outer_y : outer_y + height, outer_x : outer_x + width] = part
        outer_face[outer_y : outer_y + height, outer_x : outer_x + width] = face

    return {
        "inner_part": inner_part,
        "inner_face": inner_face,
        "outer_part": outer_part,
        "outer_face": outer_face,
    }


def _grid_to_uv01(grid):
    return ((grid + 1.0) * 0.5).clamp(0.0, 1.0)


def _grid_to_xy(grid):
    coords = (_grid_to_uv01(grid) * (UV_SIZE - 1)).round().long()
    x = coords[..., 0].clamp(0, UV_SIZE - 1)
    y = coords[..., 1].clamp(0, UV_SIZE - 1)
    return x, y


def build_dense_parser_batch(skins, renderer, view, alpha_threshold=0.5):
    """Render skins and create per-pixel dense parser targets for one view."""
    device = skins.device
    dtype = skins.dtype
    B = skins.shape[0]

    rendered = renderer.forward_view(skins, view)
    _, _, H, W = rendered.shape

    inner_grid = getattr(renderer, f"{view}_inner_grid").to(device=device, dtype=dtype)
    outer_grid = getattr(renderer, f"{view}_outer_grid").to(device=device, dtype=dtype)
    inner_mask = getattr(renderer, f"{view}_inner_mask").to(device=device).bool()
    outer_mask = getattr(renderer, f"{view}_outer_mask").to(device=device).bool()

    inner_grid_b = inner_grid.unsqueeze(0).expand(B, -1, -1, -1)
    outer_grid_b = outer_grid.unsqueeze(0).expand(B, -1, -1, -1)
    inner_alpha = F.grid_sample(
        skins[:, 3:4],
        inner_grid_b,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    outer_alpha = F.grid_sample(
        skins[:, 3:4],
        outer_grid_b,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    inner_visible = inner_mask.unsqueeze(0).unsqueeze(1) & (inner_alpha > alpha_threshold)
    outer_visible = outer_mask.unsqueeze(0).unsqueeze(1) & (outer_alpha > alpha_threshold)
    inner_visible = inner_visible & ~outer_visible
    foreground = inner_visible | outer_visible

    layer = torch.full((B, H, W), IGNORE_INDEX, dtype=torch.long, device=device)
    layer[inner_visible[:, 0]] = 0
    layer[outer_visible[:, 0]] = 1

    uv = rendered.new_zeros(B, 2, H, W)
    inner_uv01 = _grid_to_uv01(inner_grid).permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)
    outer_uv01 = _grid_to_uv01(outer_grid).permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)
    uv = torch.where(inner_visible.expand(-1, 2, -1, -1), inner_uv01, uv)
    uv = torch.where(outer_visible.expand(-1, 2, -1, -1), outer_uv01, uv)

    lookups = build_part_face_lookups(device=device)
    inner_x, inner_y = _grid_to_xy(inner_grid)
    outer_x, outer_y = _grid_to_xy(outer_grid)
    inner_part = lookups["inner_part"][inner_y, inner_x].unsqueeze(0).expand(B, -1, -1)
    inner_face = lookups["inner_face"][inner_y, inner_x].unsqueeze(0).expand(B, -1, -1)
    outer_part = lookups["outer_part"][outer_y, outer_x].unsqueeze(0).expand(B, -1, -1)
    outer_face = lookups["outer_face"][outer_y, outer_x].unsqueeze(0).expand(B, -1, -1)

    part = torch.full((B, H, W), IGNORE_INDEX, dtype=torch.long, device=device)
    face = torch.full((B, H, W), IGNORE_INDEX, dtype=torch.long, device=device)
    part[inner_visible[:, 0]] = inner_part[inner_visible[:, 0]]
    face[inner_visible[:, 0]] = inner_face[inner_visible[:, 0]]
    part[outer_visible[:, 0]] = outer_part[outer_visible[:, 0]]
    face[outer_visible[:, 0]] = outer_face[outer_visible[:, 0]]

    targets = {
        "foreground": foreground.to(dtype=dtype),
        "layer": layer,
        "part": part,
        "face": face,
        "uv": uv,
    }
    return rendered, targets


def _sample_index_target(target, grid):
    sampled = F.grid_sample(
        (target.float() + 1.0).unsqueeze(1),
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    ).squeeze(1)
    sampled = sampled.round().long() - 1
    sampled[sampled < 0] = IGNORE_INDEX
    return sampled


def augment_dense_batch(
    rendered,
    targets,
    translation_scale=0.03,
    scale_range=0.03,
    bg_color=(128, 128, 128),
    generator=None,
):
    if translation_scale <= 0 and scale_range <= 0:
        return rendered, targets

    B, C, H, W = rendered.shape
    device = rendered.device
    dtype = rendered.dtype

    dx = (torch.rand(B, device=device, dtype=dtype, generator=generator) - 0.5) * 2 * translation_scale * W
    dy = (torch.rand(B, device=device, dtype=dtype, generator=generator) - 0.5) * 2 * translation_scale * H
    scale = 1.0 + (torch.rand(B, device=device, dtype=dtype, generator=generator) - 0.5) * 2 * scale_range
    inv_scale = scale.reciprocal()

    theta = torch.zeros(B, 2, 3, device=device, dtype=dtype)
    theta[:, 0, 0] = inv_scale
    theta[:, 1, 1] = inv_scale
    theta[:, 0, 2] = -2.0 * dx / max(W, 1)
    theta[:, 1, 2] = -2.0 * dy / max(H, 1)
    grid = F.affine_grid(theta, rendered.shape, align_corners=False)

    fill = rendered.new_tensor(
        [bg_color[0] / 255.0, bg_color[1] / 255.0, bg_color[2] / 255.0, 0.0],
    )[:C].view(1, C, 1, 1)
    rendered_aug = F.grid_sample(
        rendered - fill,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ) + fill

    foreground = F.grid_sample(
        targets["foreground"],
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )
    layer = _sample_index_target(targets["layer"], grid)
    part = _sample_index_target(targets["part"], grid)
    face = _sample_index_target(targets["face"], grid)
    uv = F.grid_sample(
        targets["uv"],
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )
    valid = (layer != IGNORE_INDEX).unsqueeze(1)
    uv = torch.where(valid, uv, torch.zeros_like(uv))
    foreground = foreground * valid.to(dtype=foreground.dtype)

    return rendered_aug, {
        "foreground": foreground,
        "layer": layer,
        "part": part,
        "face": face,
        "uv": uv,
    }


def randomize_render_background(rendered, probability=0.9, bg_color=(128, 128, 128)):
    """Replace rendered backgrounds while preserving the skin pixels and parser targets.

    The renderer produces RGBA images composited over a fixed gray background. The
    parser always receives RGB-style inputs (alpha fixed to one), with a random
    solid-color background for the requested fraction of samples.
    """
    if rendered.dim() != 4 or rendered.shape[1] != 4:
        raise ValueError(f"Expected RGBA render tensor as NCHW, got {tuple(rendered.shape)}.")

    B, _, H, W = rendered.shape
    device = rendered.device
    dtype = rendered.dtype
    alpha = rendered[:, 3:4].clamp(0.0, 1.0)
    source_bg = rendered.new_tensor(bg_color).view(1, 3, 1, 1) / 255.0
    probability = max(0.0, min(float(probability), 1.0))
    if probability > 0:
        active = torch.rand(B, device=device) < probability
        random_background = torch.rand(B, 3, 1, 1, device=device, dtype=dtype).expand(-1, -1, H, W)
        background = torch.where(active.view(B, 1, 1, 1), random_background, source_bg.expand(B, -1, H, W))
    else:
        background = source_bg.expand(B, -1, H, W)
    foreground_rgb = (rendered[:, :3] - (1.0 - alpha) * source_bg) / alpha.clamp_min(1e-4)
    composited_rgb = (alpha * foreground_rgb + (1.0 - alpha) * background).clamp(0.0, 1.0)

    return torch.cat([composited_rgb, torch.ones_like(alpha)], dim=1)


def splat_predictions_to_uv_conditioning(
    rendered,
    outputs,
    group_size=1,
    fg_threshold=0.5,
    bg_color=(128, 128, 128),
):
    """Splat parser predictions back to the 10-channel inverse_uv conditioning layout."""
    foreground_prob = torch.sigmoid(outputs["foreground"])[:, 0]
    fg = foreground_prob > fg_threshold
    fg = fg & (rendered[:, 3] > 1e-4)
    layer_prob = torch.softmax(outputs["layer"], dim=1)
    layer_confidence, layer = layer_prob.max(dim=1)
    flat_uv = prediction_flat_uv(outputs)
    confidence = foreground_prob * layer_confidence
    if "uv_x" in outputs and "uv_y" in outputs:
        uv_x_confidence = torch.softmax(outputs["uv_x"], dim=1).amax(dim=1)
        uv_y_confidence = torch.softmax(outputs["uv_y"], dim=1).amax(dim=1)
        confidence = confidence * torch.sqrt((uv_x_confidence * uv_y_confidence).clamp_min(1e-8))

    return splat_to_uv_conditioning(
        rendered,
        fg,
        layer,
        flat_uv,
        group_size=group_size,
        bg_color=bg_color,
        confidence=confidence,
    )


def prediction_flat_uv(outputs):
    if "uv_x" in outputs and "uv_y" in outputs:
        x = outputs["uv_x"].argmax(dim=1).clamp(0, UV_SIZE - 1)
        y = outputs["uv_y"].argmax(dim=1).clamp(0, UV_SIZE - 1)
    else:
        uv = outputs["uv"].clamp(0.0, 1.0)
        x = (uv[:, 0] * (UV_SIZE - 1)).round().long().clamp(0, UV_SIZE - 1)
        y = (uv[:, 1] * (UV_SIZE - 1)).round().long().clamp(0, UV_SIZE - 1)
    return y * UV_SIZE + x


def prediction_uv01(outputs):
    if "uv_x" in outputs and "uv_y" in outputs:
        x = outputs["uv_x"].argmax(dim=1).to(dtype=outputs["uv"].dtype) / (UV_SIZE - 1)
        y = outputs["uv_y"].argmax(dim=1).to(dtype=outputs["uv"].dtype) / (UV_SIZE - 1)
        return torch.stack([x, y], dim=1)
    return outputs["uv"].clamp(0.0, 1.0)


def splat_targets_to_uv_conditioning(rendered, targets, group_size=1, bg_color=(128, 128, 128)):
    if rendered.dim() != 4:
        raise ValueError(f"Expected rendered tensor as NCHW, got {tuple(rendered.shape)}.")
    layer = targets["layer"]
    fg = (targets["foreground"][:, 0] > 0.5) & (layer != IGNORE_INDEX) & (rendered[:, 3] > 1e-4)
    uv = targets["uv"].clamp(0.0, 1.0)
    x = (uv[:, 0] * (UV_SIZE - 1)).round().long().clamp(0, UV_SIZE - 1)
    y = (uv[:, 1] * (UV_SIZE - 1)).round().long().clamp(0, UV_SIZE - 1)
    flat_uv = y * UV_SIZE + x
    safe_layer = torch.where(layer == IGNORE_INDEX, torch.zeros_like(layer), layer)
    return splat_to_uv_conditioning(rendered, fg, safe_layer, flat_uv, group_size=group_size, bg_color=bg_color)


def splat_to_uv_conditioning(
    rendered,
    fg,
    layer,
    flat_uv,
    group_size=1,
    bg_color=(128, 128, 128),
    confidence=None,
):
    if rendered.dim() != 4:
        raise ValueError(f"Expected rendered tensor as NCHW, got {tuple(rendered.shape)}.")
    N, _, _, _ = rendered.shape
    if N % group_size != 0:
        raise ValueError(f"N={N} must be divisible by group_size={group_size}.")

    groups = N // group_size
    device = rendered.device
    dtype = rendered.dtype
    accum = rendered.new_zeros(groups, LAYER_CLASSES, 4, UV_SIZE * UV_SIZE)
    counts = rendered.new_zeros(groups, LAYER_CLASSES, 1, UV_SIZE * UV_SIZE)

    if confidence is None:
        confidence = rendered.new_ones(fg.shape)
        select_highest_confidence = False
    else:
        select_highest_confidence = True

    for group in range(groups):
        group_start = group * group_size
        group_end = group_start + group_size
        for layer_index in range(LAYER_CLASSES):
            candidate_values = []
            candidate_uv = []
            candidate_confidence = []
            for item in range(group_start, group_end):
                item_mask = fg[item] & (layer[item] == layer_index)
                if not item_mask.any():
                    continue
                candidate_values.append(rendered[item, :, item_mask])
                candidate_uv.append(flat_uv[item, item_mask])
                candidate_confidence.append(confidence[item, item_mask])
            if not candidate_uv:
                continue

            values = torch.cat(candidate_values, dim=1)
            target_uv = torch.cat(candidate_uv, dim=0)
            scores = torch.cat(candidate_confidence, dim=0).float()
            if select_highest_confidence:
                best_scores = scores.new_full((UV_SIZE * UV_SIZE,), -torch.inf)
                best_scores.scatter_reduce_(0, target_uv, scores, reduce="amax", include_self=True)
                keep = scores >= (best_scores[target_uv] - 1e-7)
                values = values[:, keep]
                target_uv = target_uv[keep]

            accum[group, layer_index].index_add_(1, target_uv, values)
            counts[group, layer_index, 0].index_add_(
                0,
                target_uv,
                torch.ones(target_uv.shape[0], dtype=dtype, device=device),
            )

    known = (counts > 0).to(dtype=dtype)
    aggregated = accum / counts.clamp_min(1.0)
    bg = rendered.new_tensor(bg_color, dtype=dtype).view(1, 1, 3, 1) / 255.0
    rgb = torch.where(
        known.expand(-1, -1, 3, -1) > 0,
        aggregated[:, :, :3],
        bg.expand(groups, LAYER_CLASSES, 3, UV_SIZE * UV_SIZE),
    )
    alpha = torch.where(known > 0, aggregated[:, :, 3:4], torch.zeros_like(aggregated[:, :, 3:4]))
    layers = torch.cat([rgb, alpha, known], dim=2).reshape(groups, -1, UV_SIZE, UV_SIZE)
    return layers.clamp(0.0, 1.0)


PART_PALETTE = (
    (239, 83, 80),
    (255, 202, 40),
    (102, 187, 106),
    (38, 166, 154),
    (66, 165, 245),
    (171, 71, 188),
)
LAYER_PALETTE = (
    (72, 169, 166),
    (255, 179, 71),
)


def bg_tensor(bg_color, reference):
    return reference.new_tensor(bg_color).view(1, 3, 1, 1) / 255.0


def colorize_labels(labels, palette, bg_color, reference):
    N, H, W = labels.shape
    bg = bg_tensor(bg_color, reference).expand(N, 3, H, W).clone()
    valid = labels != IGNORE_INDEX
    if not valid.any():
        return bg
    palette_tensor = reference.new_tensor(palette, dtype=reference.dtype) / 255.0
    safe_labels = labels.clamp(0, len(palette) - 1)
    out = bg.permute(0, 2, 3, 1)
    out[valid] = palette_tensor[safe_labels[valid]]
    return bg


def colorize_foreground(mask, bg_color, reference):
    N, H, W = mask.shape
    bg = bg_tensor(bg_color, reference).expand(N, 3, H, W)
    fg = reference.new_ones(N, 3, H, W)
    return torch.where(mask.unsqueeze(1), fg, bg)


def colorize_uv(uv, mask, bg_color):
    N, _, H, W = uv.shape
    bg = bg_tensor(bg_color, uv).expand(N, 3, H, W)
    zeros = uv.new_zeros(N, 1, H, W)
    uv_rgb = torch.cat([uv[:, 0:1], uv[:, 1:2], zeros], dim=1)
    return torch.where(mask.unsqueeze(1), uv_rgb, bg)

