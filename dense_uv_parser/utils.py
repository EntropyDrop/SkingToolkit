import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TOOLKIT_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from SkingToolkit.semantic_uv_reconstruction.losses import minecraft_layer_rects  # noqa: E402

IGNORE_INDEX = 255
PART_CLASSES = 6
FACE_CLASSES = 6
LAYER_CLASSES = 2
LAYER_FACE_CLASSES = LAYER_CLASSES * FACE_CLASSES
ROUTE_ROLE_CLASSES = 3
ROUTE_INNER_PRIMARY = 0
ROUTE_OUTER_PRIMARY = 1
ROUTE_SECONDARY = 2
UV_SIZE = 64
SPLAT_COLOR_AGGREGATIONS = ("exact_mode", "best")
SOLID_BACKGROUND_COLOR_TOLERANCE = 16.0 / 255.0
SOLID_BACKGROUND_MIN_CORNER_SUPPORT = 0.5
SECONDARY_TEXEL_MIN_PROBABILITY = 0.5


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


def _renderer_surface_cache(renderer):
    cache = getattr(renderer, "_dense_uv_surface_cache", None)
    if cache is None:
        cache = {}
        setattr(renderer, "_dense_uv_surface_cache", cache)
    return cache


def _surface_metadata(grids, is_outer, lookups):
    x, y = _grid_to_xy(grids)
    inner_part = lookups["inner_part"][y, x]
    inner_face = lookups["inner_face"][y, x]
    outer_part = lookups["outer_part"][y, x]
    outer_face = lookups["outer_face"][y, x]
    return {
        "flat_uv": y * UV_SIZE + x,
        "layer": is_outer.long(),
        "part": torch.where(is_outer, outer_part, inner_part),
        "face": torch.where(is_outer, outer_face, inner_face),
        "outer_part": outer_part,
    }


def build_static_surface_routing(renderer, view, device):
    """Build fixed UV candidates matching the renderer's direct/composite geometry paths."""
    device = torch.device(device)
    cache = _renderer_surface_cache(renderer)
    cache_key = (view, device.type, device.index)
    if cache_key in cache:
        return cache[cache_key]

    inner_grid = getattr(renderer, f"{view}_inner_grid").to(device=device)
    outer_grid = getattr(renderer, f"{view}_outer_grid").to(device=device)
    inner_mask = getattr(renderer, f"{view}_inner_mask").to(device=device).bool()
    outer_mask = getattr(renderer, f"{view}_outer_mask").to(device=device).bool()
    H, W = inner_mask.shape
    lookups = build_part_face_lookups(device=device)

    grids = [inner_grid, outer_grid]
    masks = [inner_mask, outer_mask]
    is_outer = [torch.zeros((H, W), dtype=torch.bool, device=device), torch.ones((H, W), dtype=torch.bool, device=device)]
    composite_count = 0
    geometry_count = 0

    composite_grid_name = f"{view}_composite_grid_layers"
    composite_decor_name = f"{view}_composite_is_decor_layers"
    if hasattr(renderer, composite_grid_name) and hasattr(renderer, composite_decor_name):
        composite_grids = getattr(renderer, composite_grid_name).to(device=device)
        composite_masks = getattr(renderer, f"{view}_composite_mask_layers").to(device=device).bool()
        composite_decor = getattr(renderer, composite_decor_name).to(device=device).bool()
        composite_count = composite_grids.shape[0]
        grids.extend(composite_grids.unbind(0))
        masks.extend(composite_masks.unbind(0))
        is_outer.extend(composite_decor.unbind(0))

    geometry_grid_name = f"{view}_geometry_grid_layers"
    if hasattr(renderer, geometry_grid_name):
        geometry_grids = getattr(renderer, geometry_grid_name).to(device=device)
        geometry_masks = getattr(renderer, f"{view}_geometry_mask_layers").to(device=device).bool()
        geometry_order = getattr(renderer, f"{view}_geometry_sort_indices").to(device=device).long()
        geometry_count = geometry_grids.shape[0]
        if geometry_count > 0:
            geometry_order = geometry_order.clamp(0, geometry_count - 1)
            sorted_grids = torch.gather(
                geometry_grids,
                0,
                geometry_order.unsqueeze(-1).expand(-1, -1, -1, 2),
            )
            sorted_masks = torch.gather(geometry_masks, 0, geometry_order)
            geometry_metadata = _surface_metadata(
                sorted_grids,
                torch.zeros_like(sorted_masks),
                lookups,
            )
            geometry_is_outer = geometry_metadata["outer_part"] != IGNORE_INDEX
            grids.extend(sorted_grids.unbind(0))
            masks.extend(sorted_masks.unbind(0))
            is_outer.extend(geometry_is_outer.unbind(0))

    surface_grids = torch.stack(grids, dim=0)
    surface_masks = torch.stack(masks, dim=0)
    surface_is_outer = torch.stack(is_outer, dim=0)
    metadata = _surface_metadata(surface_grids, surface_is_outer, lookups)
    metadata["layer_face"] = combine_layer_face(metadata["layer"], metadata["face"])
    static = {
        "grids": surface_grids,
        "masks": surface_masks,
        "flat_uv": metadata["flat_uv"],
        "layer": metadata["layer"],
        "part": metadata["part"],
        "face": metadata["face"],
        "layer_face": metadata["layer_face"],
        "direct_count": 2,
        "composite_count": composite_count,
        "geometry_count": geometry_count,
    }
    cache[cache_key] = static
    return static


def surface_class_count(renderer, views):
    device = renderer.bg_color.device
    return max(build_static_surface_routing(renderer, view, device)["grids"].shape[0] for view in parse_views(views))


def _select_static_surface(static, surface):
    B, H, W = surface.shape
    surface_count = static["grids"].shape[0]
    in_range = (surface >= 0) & (surface < surface_count)
    safe_surface = surface.clamp(0, max(surface_count - 1, 0))

    selected = {}
    for name in ("masks", "flat_uv", "layer", "part", "face"):
        values = static[name].unsqueeze(0).expand(B, -1, -1, -1)
        selected[name] = values.gather(1, safe_surface.unsqueeze(1)).squeeze(1)
    selected["valid"] = in_range & selected["masks"]
    return selected


def _sample_surface_alpha(skins, static):
    B = skins.shape[0]
    surface_count, H, W, _ = static["grids"].shape
    grids = static["grids"].unsqueeze(0).expand(B, -1, -1, -1, -1).reshape(B * surface_count, H, W, 2)
    alpha = skins[:, 3:4].unsqueeze(1).expand(-1, surface_count, -1, -1, -1)
    alpha = alpha.reshape(B * surface_count, 1, skins.shape[-2], skins.shape[-1])
    sampled = F.grid_sample(alpha, grids, mode="bilinear", padding_mode="zeros", align_corners=True)
    return sampled.reshape(B, surface_count, H, W) * static["masks"].unsqueeze(0)


def classify_route_role(static, selected_layer, selected_flat_uv, valid):
    """Classify visible pixels as direct inner/outer surfaces or deeper backfaces."""
    B, H, W = selected_layer.shape
    safe_layer = selected_layer.clamp(0, LAYER_CLASSES - 1)
    direct_masks = static["masks"][:LAYER_CLASSES].unsqueeze(0).expand(B, -1, H, W)
    direct_uv = static["flat_uv"][:LAYER_CLASSES].unsqueeze(0).expand(B, -1, H, W)
    direct_valid = direct_masks.gather(1, safe_layer.unsqueeze(1)).squeeze(1)
    expected_uv = direct_uv.gather(1, safe_layer.unsqueeze(1)).squeeze(1)
    primary = valid & direct_valid & (selected_flat_uv == expected_uv)
    route_role = torch.full_like(selected_layer, IGNORE_INDEX)
    route_role[primary] = selected_layer[primary]
    route_role[valid & ~primary] = ROUTE_SECONDARY
    return route_role


def build_dense_parser_batch(skins, renderer, view, alpha_threshold=0.5):
    """Render skins and label the exact fixed surface that produced every pixel."""
    device = skins.device
    dtype = skins.dtype
    B = skins.shape[0]

    rendered = renderer.forward_view(skins, view)
    static = build_static_surface_routing(renderer, view, device)
    _, _, H, W = rendered.shape
    sampled_alpha = _sample_surface_alpha(skins, static)

    surface = torch.full((B, H, W), IGNORE_INDEX, dtype=torch.long, device=device)
    direct_visible = sampled_alpha[:, :2] > alpha_threshold
    inner_visible = direct_visible[:, 0] & ~direct_visible[:, 1]
    outer_visible = direct_visible[:, 1]
    surface[inner_visible] = 0
    surface[outer_visible] = 1

    composite_start = static["direct_count"]
    composite_count = static["composite_count"]
    composite_visible = None
    if composite_count > 0:
        composite_visible = sampled_alpha[:, composite_start : composite_start + composite_count] > alpha_threshold
        composite_indices = torch.arange(composite_count, device=device).view(1, composite_count, 1, 1)
        first_composite = torch.where(
            composite_visible,
            composite_indices,
            torch.full_like(composite_indices, composite_count),
        ).amin(dim=1)
        has_composite = first_composite < composite_count
        composite_surface = composite_start + first_composite.clamp(max=composite_count - 1)
        composite_route = _select_static_surface(static, composite_surface)
        trust_composite = has_composite & (
            (composite_route["layer"] == 1) | (first_composite <= 1)
        )
        surface[trust_composite] = composite_surface[trust_composite]

        geometry_start = composite_start + composite_count
        geometry_count = static["geometry_count"]
        if geometry_count > 0:
            geometry_visible = sampled_alpha[:, geometry_start : geometry_start + geometry_count] > alpha_threshold
            geometry_indices = torch.arange(geometry_count, device=device).view(1, geometry_count, 1, 1)
            last_geometry = torch.where(
                geometry_visible,
                geometry_indices,
                torch.full_like(geometry_indices, -1),
            ).amax(dim=1)
            has_geometry = last_geometry >= 0
            geometry_surface = geometry_start + last_geometry.clamp_min(0)
            front_decor = static["masks"][composite_start] & (static["layer"][composite_start] == 1)
            geometry_fallback = (
                front_decor.unsqueeze(0)
                & ~composite_visible[:, 0]
                & has_geometry
            )
            surface[geometry_fallback] = geometry_surface[geometry_fallback]

    selected = _select_static_surface(static, surface)
    valid = selected["valid"]
    layer = torch.where(valid, selected["layer"], torch.full_like(surface, IGNORE_INDEX))
    route_role = classify_route_role(static, selected["layer"], selected["flat_uv"], valid)
    part = torch.where(valid, selected["part"], torch.full_like(surface, IGNORE_INDEX))
    face = torch.where(valid, selected["face"], torch.full_like(surface, IGNORE_INDEX))
    uv = flat_uv_to_uv01(selected["flat_uv"], dtype).masked_fill(~valid.unsqueeze(1), 0.0)

    targets = {
        "foreground": valid.unsqueeze(1).to(dtype=dtype),
        "layer": layer,
        "route_role": route_role,
        "part": part,
        "face": face,
        "surface": torch.where(valid, surface, torch.full_like(surface, IGNORE_INDEX)),
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
    translation_scale=0.0,
    scale_range=0.0,
    bg_color=(128, 128, 128),
    generator=None,
):
    B, _, H, W = rendered.shape
    device = rendered.device
    dtype = rendered.dtype
    if translation_scale <= 0 and scale_range <= 0:
        identity_targets = dict(targets)
        identity_targets["affine"] = rendered.new_zeros(B, 3)
        return rendered, identity_targets

    C = rendered.shape[1]

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
    route_role = _sample_index_target(targets["route_role"], grid)
    part = _sample_index_target(targets["part"], grid)
    face = _sample_index_target(targets["face"], grid)
    surface = _sample_index_target(targets["surface"], grid)
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

    affine = torch.stack(
        [
            2.0 * dx / max(W, 1),
            2.0 * dy / max(H, 1),
            scale.log(),
        ],
        dim=1,
    )
    return rendered_aug, {
        "foreground": foreground,
        "layer": layer,
        "route_role": route_role,
        "part": part,
        "face": face,
        "surface": surface,
        "uv": uv,
        # [tx, ty, log_scale] maps canonical output coordinates to the
        # augmented input. It is the inverse of the grid used above.
        "affine": affine,
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


def estimate_solid_background_color(
    rendered,
    color_tolerance=SOLID_BACKGROUND_COLOR_TOLERANCE,
    min_corner_support=SOLID_BACKGROUND_MIN_CORNER_SUPPORT,
):
    if rendered.dim() != 4 or rendered.shape[1] < 3:
        raise ValueError(f"Expected NCHW RGB(A), got {tuple(rendered.shape)}.")
    if not 0.0 <= color_tolerance <= 1.0:
        raise ValueError("color_tolerance must be in [0, 1].")
    if not 0.0 <= min_corner_support <= 1.0:
        raise ValueError("min_corner_support must be in [0, 1].")

    rgb = rendered[:, :3].float()
    batch, _, height, width = rgb.shape
    patch = max(2, min(32, min(height, width) // 16))
    corners = torch.cat(
        [
            rgb[:, :, :patch, :patch].flatten(2),
            rgb[:, :, :patch, -patch:].flatten(2),
            rgb[:, :, -patch:, :patch].flatten(2),
            rgb[:, :, -patch:, -patch:].flatten(2),
        ],
        dim=2,
    )
    background_rgb = corners.median(dim=2).values.view(batch, 3, 1, 1)
    corner_distance = (corners - background_rgb.flatten(2)).abs().amax(dim=1)
    solid_background = (corner_distance <= color_tolerance).float().mean(dim=1) >= min_corner_support
    return background_rgb, solid_background


def estimate_solid_background_foreground(
    rendered,
    color_tolerance=SOLID_BACKGROUND_COLOR_TOLERANCE,
    min_corner_support=SOLID_BACKGROUND_MIN_CORNER_SUPPORT,
):
    """Find non-background pixels without deleting matching colors enclosed by the subject."""
    if min(rendered.shape[-2:]) < 8:
        return torch.ones(
            rendered.shape[0],
            *rendered.shape[-2:],
            dtype=torch.bool,
            device=rendered.device,
        )

    rgb = rendered[:, :3].float()
    batch, _, height, width = rgb.shape
    background_rgb, solid_background = estimate_solid_background_color(
        rendered,
        color_tolerance=color_tolerance,
        min_corner_support=min_corner_support,
    )

    background_candidate = (rgb - background_rgb).abs().amax(dim=1) <= color_tolerance
    candidate_u8 = background_candidate.to(torch.uint8)
    connected_to_border = (
        candidate_u8.cumprod(dim=2).bool()
        | candidate_u8.flip(2).cumprod(dim=2).flip(2).bool()
        | candidate_u8.cumprod(dim=1).bool()
        | candidate_u8.flip(1).cumprod(dim=1).flip(1).bool()
    )
    connected_to_border &= solid_background.view(batch, 1, 1)
    max_steps = max(height, width)
    for _ in range(max_steps):
        expanded = F.max_pool2d(
            connected_to_border.unsqueeze(1).float(), 3, stride=1, padding=1
        )[:, 0] > 0.0
        updated = connected_to_border | (background_candidate & expanded)
        if torch.equal(updated, connected_to_border):
            break
        connected_to_border = updated
    return ~connected_to_border


def affine_to_canonical_grid(affine, output_shape):
    """Build a grid that samples an augmented render at canonical coordinates."""
    if affine.dim() != 2 or affine.shape[1] != 3:
        raise ValueError(f"Expected affine shape (N, 3), got {tuple(affine.shape)}.")
    if affine.shape[0] != output_shape[0]:
        raise ValueError(
            f"Affine batch size {affine.shape[0]} does not match output batch size {output_shape[0]}."
        )

    scale = affine[:, 2].exp()
    theta = affine.new_zeros(affine.shape[0], 2, 3)
    theta[:, 0, 0] = scale
    theta[:, 1, 1] = scale
    theta[:, 0, 2] = affine[:, 0]
    theta[:, 1, 2] = affine[:, 1]
    return F.affine_grid(theta, output_shape, align_corners=False)


def canonicalize_tensor(tensor, affine, mode="bilinear"):
    """Undo the parser's predicted global translation/scale for an NCHW tensor."""
    affine = affine.to(device=tensor.device, dtype=tensor.dtype)
    grid = affine_to_canonical_grid(affine, tensor.shape)
    return F.grid_sample(
        tensor,
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=False,
    )


def canonicalize_index_tensor(index, affine):
    """Undo a global transform while preserving IGNORE_INDEX outside valid labels."""
    valid = (index != IGNORE_INDEX).unsqueeze(1).float()
    values = index.masked_fill(index == IGNORE_INDEX, 0).float().unsqueeze(1)
    affine = affine.to(device=values.device, dtype=values.dtype)
    grid = affine_to_canonical_grid(affine, values.shape)
    sampled_valid = F.grid_sample(
        valid,
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    ).squeeze(1) > 0.5
    sampled_values = F.grid_sample(
        values,
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    ).squeeze(1).round().long()
    sampled_values[~sampled_valid] = IGNORE_INDEX
    return sampled_values


def canonicalize_dense_targets(targets):
    """Return parser targets in the fixed renderer coordinate system."""
    if "affine" not in targets:
        raise ValueError("Global-affine targets require an 'affine' tensor.")
    affine = targets["affine"]
    canonical = {
        "foreground": canonicalize_tensor(targets["foreground"], affine, mode="nearest"),
        "layer": canonicalize_index_tensor(targets["layer"], affine),
        "route_role": canonicalize_index_tensor(targets["route_role"], affine),
        "part": canonicalize_index_tensor(targets["part"], affine),
        "face": canonicalize_index_tensor(targets["face"], affine),
        "surface": canonicalize_index_tensor(targets["surface"], affine),
        "uv": canonicalize_tensor(targets["uv"], affine, mode="nearest"),
        "affine": affine,
    }
    valid = (canonical["layer"] != IGNORE_INDEX).unsqueeze(1)
    canonical["foreground"] = canonical["foreground"] * valid.to(canonical["foreground"].dtype)
    canonical["uv"] = torch.where(valid, canonical["uv"], torch.zeros_like(canonical["uv"]))
    return canonical


def canonicalize_parser_outputs(outputs):
    """Warp dense parser logits into the fixed renderer coordinate system."""
    if "affine" not in outputs:
        return outputs
    affine = outputs["affine"]
    canonical = {}
    for name, value in outputs.items():
        if name == "affine" or not torch.is_tensor(value) or value.dim() != 4:
            canonical[name] = value
        else:
            canonical[name] = canonicalize_tensor(value, affine, mode="bilinear")
    return canonical


def canonicalize_parser_render(rendered, outputs, mode="nearest", fill_color=None):
    """Undo the global transform for colors without blending Minecraft texels."""
    if "affine" not in outputs:
        return rendered
    if fill_color is None:
        return canonicalize_tensor(rendered, outputs["affine"], mode=mode)
    if fill_color.dim() == 2:
        fill_color = fill_color.unsqueeze(-1).unsqueeze(-1)
    if fill_color.shape != (rendered.shape[0], rendered.shape[1], 1, 1):
        raise ValueError(
            "fill_color must have shape "
            f"{(rendered.shape[0], rendered.shape[1], 1, 1)}, got {tuple(fill_color.shape)}."
        )
    fill_color = fill_color.to(device=rendered.device, dtype=rendered.dtype)
    return canonicalize_tensor(
        rendered - fill_color, outputs["affine"], mode=mode
    ) + fill_color


def _mask_moments(weights):
    weights = weights.float()
    N, H, W = weights.shape
    y = torch.arange(H, device=weights.device, dtype=weights.dtype).view(1, H, 1)
    x = torch.arange(W, device=weights.device, dtype=weights.dtype).view(1, 1, W)
    total = weights.sum(dim=(1, 2)).clamp_min(1e-6)
    mean_x = (weights * x).sum(dim=(1, 2)) / total
    mean_y = (weights * y).sum(dim=(1, 2)) / total
    var_x = (weights * (x - mean_x.view(N, 1, 1)).square()).sum(dim=(1, 2)) / total
    var_y = (weights * (y - mean_y.view(N, 1, 1)).square()).sum(dim=(1, 2)) / total
    return total, mean_x, mean_y, var_x.clamp_min(1e-6).sqrt(), var_y.clamp_min(1e-6).sqrt()


def _soft_mask_dice(probability, target, support):
    probability = probability.float() * support.float()
    target = target.float()
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (2.0 * intersection + 1e-6) / (denominator + 1e-6)


def refine_parser_affine(
    outputs,
    renderer,
    views,
    translation_radius_px=8.0,
    scale_radius=0.0,
    translation_step_px=0.5,
    observed_foreground=None,
):
    """Refine parser affine predictions against the fixed inner-layer silhouette."""
    if "affine" not in outputs or "foreground" not in outputs:
        return outputs.get("affine"), None
    if translation_radius_px < 0 or scale_radius < 0:
        raise ValueError("Affine refinement radii must be non-negative.")
    if translation_step_px <= 0:
        raise ValueError("translation_step_px must be positive.")

    views = parse_views(views)
    affine = outputs["affine"]
    if observed_foreground is None:
        foreground_prob = torch.sigmoid(outputs["foreground"])
    else:
        if observed_foreground.shape != outputs["foreground"].shape[:1] + outputs["foreground"].shape[-2:]:
            raise ValueError(
                "observed_foreground must match parser spatial output shape, got "
                f"{tuple(observed_foreground.shape)}."
            )
        foreground_prob = observed_foreground.to(
            device=outputs["foreground"].device,
            dtype=outputs["foreground"].dtype,
        ).unsqueeze(1)
    N, _, H, W = foreground_prob.shape
    if not views or N % len(views) != 0:
        raise ValueError(f"N={N} must be divisible by the number of views ({len(views)}).")

    target = foreground_prob.new_zeros(N, 1, H, W)
    for view_index, view in enumerate(views):
        static = build_static_surface_routing(renderer, view, foreground_prob.device)
        inner_mask = static["masks"][0]
        if inner_mask.shape != (H, W):
            raise ValueError(
                f"View {view!r} mapping shape {tuple(inner_mask.shape)} does not match parser shape {(H, W)}."
            )
        target[view_index::len(views), 0] = inner_mask.to(dtype=target.dtype)

    support_radius = max(math.ceil(translation_radius_px) + 2, 2)
    support = F.max_pool2d(
        target,
        kernel_size=support_radius * 2 + 1,
        stride=1,
        padding=support_radius,
    ).clamp(0.0, 1.0)

    base_canonical = canonicalize_tensor(foreground_prob, affine, mode="bilinear")
    base_score = _soft_mask_dice(base_canonical, target, support)
    target_stats = _mask_moments(target[:, 0])
    base_stats = _mask_moments(base_canonical[:, 0] * support[:, 0])
    target_total, _, _, target_std_x, target_std_y = target_stats
    base_total, _, _, base_std_x, base_std_y = base_stats
    enough_foreground = base_total > target_total * 0.25

    if scale_radius > 0:
        scale_delta = 0.5 * (
            (base_std_x / target_std_x).clamp_min(1e-6).log()
            + (base_std_y / target_std_y).clamp_min(1e-6).log()
        )
        scale_delta = scale_delta.clamp(-scale_radius, scale_radius)
        scale_affine = affine.clone()
        scale_affine[:, 2] = scale_affine[:, 2] + scale_delta.to(dtype=scale_affine.dtype)
        scale_canonical = canonicalize_tensor(foreground_prob, scale_affine, mode="bilinear")
    else:
        scale_delta = torch.zeros_like(base_score)
        scale_affine = affine
        scale_canonical = base_canonical
    _, pred_mean_x, pred_mean_y, _, _ = _mask_moments(scale_canonical[:, 0] * support[:, 0])
    _, target_mean_x, target_mean_y, _, _ = target_stats
    delta_x_px = (pred_mean_x - target_mean_x).clamp(-translation_radius_px, translation_radius_px)
    delta_y_px = (pred_mean_y - target_mean_y).clamp(-translation_radius_px, translation_radius_px)
    delta_x_px = (delta_x_px / translation_step_px).round() * translation_step_px
    delta_y_px = (delta_y_px / translation_step_px).round() * translation_step_px

    candidate_affine = scale_affine.clone()
    candidate_affine[:, 0] = candidate_affine[:, 0] + (2.0 * delta_x_px / W).to(candidate_affine.dtype)
    candidate_affine[:, 1] = candidate_affine[:, 1] + (2.0 * delta_y_px / H).to(candidate_affine.dtype)
    candidate_canonical = canonicalize_tensor(foreground_prob, candidate_affine, mode="bilinear")
    candidate_score = _soft_mask_dice(candidate_canonical, target, support)
    accepted = enough_foreground & (candidate_score > base_score + 1e-5)
    refined_affine = torch.where(accepted.unsqueeze(1), candidate_affine, affine)

    accepted_float = accepted.to(dtype=delta_x_px.dtype)
    details = {
        "accepted": accepted,
        "translation_px": torch.stack(
            [delta_x_px * accepted_float, delta_y_px * accepted_float],
            dim=1,
        ),
        "scale_percent": (scale_delta.exp() - 1.0) * 100.0 * accepted_float,
        "score_before": base_score,
        "score_after": torch.where(accepted, candidate_score, base_score),
    }
    return refined_affine, details


def _routing_from_affine_outputs(renderer, views, outputs, fg_threshold=0.5, semantic_gate=True):
    views = parse_views(views)
    if not views:
        raise ValueError("At least one renderer view is required for deterministic UV routing.")
    if "surface" not in outputs:
        raise ValueError("Affine parser outputs must include the static surface classifier.")
    foreground_prob = torch.sigmoid(outputs["foreground"])[:, 0]
    surface_prob = torch.softmax(outputs["surface"], dim=1)
    layer_prob = torch.softmax(outputs["layer"], dim=1)
    N, _, H, W = surface_prob.shape
    if N % len(views) != 0:
        raise ValueError(f"N={N} must be divisible by {len(views)} renderer views.")

    fg = torch.zeros_like(foreground_prob, dtype=torch.bool)
    layer = torch.zeros((N, H, W), dtype=torch.long, device=surface_prob.device)
    flat_uv = torch.zeros_like(layer)
    surface = torch.full_like(layer, IGNORE_INDEX)
    confidence = torch.zeros_like(foreground_prob)
    confidence_margin = torch.zeros_like(foreground_prob)
    confidence_margin_ratio = torch.zeros_like(foreground_prob)
    semantic_fallback = torch.zeros_like(foreground_prob, dtype=torch.bool)
    expected_part = torch.full_like(layer, IGNORE_INDEX)
    expected_face = torch.full_like(layer, IGNORE_INDEX)

    for view_index, view in enumerate(views):
        selection = slice(view_index, N, len(views))
        static = build_static_surface_routing(renderer, view, surface_prob.device)
        if static["masks"].shape[-2:] != (H, W):
            raise ValueError(
                f"View {view!r} mapping shape {tuple(static['masks'].shape[-2:])} does not match parser shape {(H, W)}."
            )
        view_surface_prob = surface_prob[selection]
        view_batch = view_surface_prob.shape[0]
        surface_count = static["masks"].shape[0]
        if view_surface_prob.shape[1] < surface_count:
            raise ValueError(
                f"Parser has {view_surface_prob.shape[1]} surface classes, but view {view!r} requires {surface_count}."
            )

        candidate_score = view_surface_prob[:, :surface_count]
        physical_valid = static["masks"].unsqueeze(0).expand(view_batch, -1, -1, -1)
        candidate_valid = physical_valid.clone()
        semantic_score = torch.ones_like(candidate_score)

        # Re-rank only surfaces that physically exist at this screen pixel. This
        # converts the semantic heads from a rejection gate into useful routing
        # evidence and avoids holes caused by a globally invalid surface argmax.
        semantic_names = (
            ("part", "layer_face")
            if "layer_face" in outputs
            else ("layer", "part", "face")
        )
        for name in semantic_names:
            if name not in outputs:
                continue
            probabilities = torch.softmax(outputs[name][selection], dim=1)
            expected = static[name]
            known_semantic = expected != IGNORE_INDEX
            expected_safe = expected.clamp(0, probabilities.shape[1] - 1)
            expected_probability = probabilities.gather(
                1,
                expected_safe.unsqueeze(0).expand(view_batch, -1, -1, -1),
            )
            semantic_score = semantic_score * torch.where(
                known_semantic.unsqueeze(0),
                expected_probability,
                torch.ones_like(expected_probability),
            )
            if semantic_gate:
                candidate_valid = candidate_valid & (
                    ~known_semantic.unsqueeze(0)
                    | (probabilities.argmax(dim=1).unsqueeze(1) == expected_safe.unsqueeze(0))
                )

        candidate_score = candidate_score * semantic_score.clamp_min(1e-12).sqrt()
        if "uv_x" in outputs and "uv_y" in outputs:
            uv_x_prob = torch.softmax(outputs["uv_x"][selection], dim=1)
            uv_y_prob = torch.softmax(outputs["uv_y"][selection], dim=1)
            candidate_x = static["flat_uv"].remainder(UV_SIZE)
            candidate_y = static["flat_uv"].div(UV_SIZE, rounding_mode="floor")
            candidate_x_prob = uv_x_prob.gather(
                1,
                candidate_x.unsqueeze(0).expand(view_batch, -1, -1, -1),
            )
            candidate_y_prob = uv_y_prob.gather(
                1,
                candidate_y.unsqueeze(0).expand(view_batch, -1, -1, -1),
            )
            candidate_score = candidate_score * (
                candidate_x_prob * candidate_y_prob
            ).clamp_min(1e-12).sqrt()
        gated_score = candidate_score.masked_fill(~candidate_valid, -1.0)
        has_gated_candidate = candidate_valid.any(dim=1, keepdim=True)
        if semantic_gate:
            fallback_score = candidate_score.masked_fill(~physical_valid, -1.0)
            candidate_score = torch.where(has_gated_candidate, gated_score, fallback_score)
            semantic_fallback[selection] = (
                ~has_gated_candidate.squeeze(1) & physical_valid.any(dim=1)
            )
        else:
            candidate_score = gated_score
        best_score, selected_surface = candidate_score.max(dim=1)
        if surface_count > 1:
            top_scores = candidate_score.topk(k=2, dim=1).values
            route_margin = (top_scores[:, 0] - top_scores[:, 1]).clamp_min(0.0)
            route_margin_ratio = (
                route_margin / top_scores[:, 0].abs().clamp_min(1e-8)
            ).clamp(0.0, 1.0)
        else:
            route_margin = best_score.clamp_min(0.0)
            route_margin_ratio = (best_score > 0.0).to(best_score.dtype)

        routed = _select_static_surface(static, selected_surface)
        selected_layer = routed["layer"]
        selected_part = routed["part"]
        selected_face = routed["face"]
        has_candidate = best_score >= 0.0
        base_silhouette = static["masks"][0].unsqueeze(0).expand(view_batch, -1, -1)
        routed_fg = (
            ((foreground_prob[selection] > fg_threshold) | base_silhouette)
            & has_candidate
            & routed["valid"]
        )
        fg[selection] = routed_fg
        layer[selection] = selected_layer
        flat_uv[selection] = routed["flat_uv"]
        surface[selection] = torch.where(
            has_candidate,
            selected_surface,
            torch.full_like(selected_surface, IGNORE_INDEX),
        )
        confidence[selection] = foreground_prob[selection] * best_score.clamp_min(0.0)
        confidence_margin[selection] = route_margin
        confidence_margin_ratio[selection] = route_margin_ratio
        expected_part[selection] = selected_part
        expected_face[selection] = selected_face

    return {
        "foreground": fg,
        "layer": layer,
        "flat_uv": flat_uv,
        "surface": surface,
        "confidence": confidence,
        "confidence_margin": confidence_margin,
        "confidence_margin_ratio": confidence_margin_ratio,
        "semantic_fallback": semantic_fallback,
        "part": expected_part,
        "face": expected_face,
    }


def _aggregate_role_probabilities_by_direct_texel(role_prob, static):
    """Average route-role probabilities within each projected Minecraft texel."""
    role_prob = role_prob.float()
    batch, role_count, height, width = role_prob.shape
    aggregated = []
    for layer_index in range(LAYER_CLASSES):
        valid = static["masks"][layer_index].reshape(1, 1, -1)
        flat_uv = static["flat_uv"][layer_index].reshape(1, 1, -1)
        sums = role_prob.new_zeros(batch, role_count, UV_SIZE * UV_SIZE)
        sums.scatter_add_(
            2,
            flat_uv.expand(batch, role_count, -1),
            role_prob.flatten(2) * valid,
        )
        counts = role_prob.new_zeros(UV_SIZE * UV_SIZE)
        counts.scatter_add_(
            0,
            flat_uv.reshape(-1),
            valid.reshape(-1).to(dtype=role_prob.dtype),
        )
        means = sums / counts.view(1, 1, -1).clamp_min(1.0)
        aggregated.append(
            means.gather(2, flat_uv.expand(batch, role_count, -1)).reshape(
                batch, role_count, height, width
            )
        )
    return torch.stack(aggregated, dim=1)


def _routing_from_geometry_outputs(
    renderer,
    views,
    outputs,
    fg_threshold=0.5,
    outer_threshold=0.5,
    texel_consensus=True,
):
    """Route a fitted Steve render through fixed inner/outer cuboid UV maps."""
    views = parse_views(views)
    if not views:
        raise ValueError("At least one renderer view is required for geometry routing.")
    foreground_prob = torch.sigmoid(outputs["foreground"])[:, 0]
    if outputs["layer"].shape[1] != ROUTE_ROLE_CLASSES:
        raise ValueError(
            "geometry_fit requires a three-class route-role head "
            "(inner_primary, outer_primary, secondary_backface). Retrain the parser."
        )
    role_prob = torch.softmax(outputs["layer"].float(), dim=1)
    raw_route_role = role_prob.argmax(dim=1)
    route_role = raw_route_role.clone()
    outer_prob = role_prob[:, ROUTE_OUTER_PRIMARY].clone()
    inner_prob = role_prob[:, ROUTE_INNER_PRIMARY].clone()
    top_prob = role_prob.topk(2, dim=1).values
    role_margin = top_prob[:, 0] - top_prob[:, 1]
    N, H, W = outer_prob.shape
    if N % len(views) != 0:
        raise ValueError(f"N={N} must be divisible by {len(views)} renderer views.")

    fg = torch.zeros_like(outer_prob, dtype=torch.bool)
    layer = torch.zeros((N, H, W), dtype=torch.long, device=outer_prob.device)
    flat_uv = torch.zeros_like(layer)
    surface = torch.full_like(layer, IGNORE_INDEX)
    confidence = torch.zeros_like(outer_prob)
    confidence_margin = torch.zeros_like(outer_prob)
    confidence_margin_ratio = torch.zeros_like(outer_prob)
    part = torch.full_like(layer, IGNORE_INDEX)
    face = torch.full_like(layer, IGNORE_INDEX)
    for view_index, view in enumerate(views):
        selection = slice(view_index, N, len(views))
        static = build_static_surface_routing(renderer, view, outer_prob.device)
        if static["masks"].shape[-2:] != (H, W):
            raise ValueError(
                f"View {view!r} mapping shape {tuple(static['masks'].shape[-2:])} "
                f"does not match parser shape {(H, W)}."
            )

        inner_valid = static["masks"][0].unsqueeze(0).expand(outer_prob[selection].shape[0], -1, -1)
        outer_valid = static["masks"][1].unsqueeze(0).expand_as(inner_valid)
        if texel_consensus:
            direct_scores = _aggregate_role_probabilities_by_direct_texel(
                role_prob[selection], static
            )
            unavailable = role_prob.new_tensor(-1.0)
            inner_score = torch.where(
                inner_valid,
                direct_scores[:, 0, ROUTE_INNER_PRIMARY],
                unavailable,
            )
            outer_score = torch.where(
                outer_valid,
                direct_scores[:, 1, ROUTE_OUTER_PRIMARY],
                unavailable,
            )
            secondary_score = torch.maximum(
                torch.where(
                    inner_valid,
                    direct_scores[:, 0, ROUTE_SECONDARY],
                    unavailable,
                ),
                torch.where(
                    outer_valid,
                    direct_scores[:, 1, ROUTE_SECONDARY],
                    unavailable,
                ),
            )
            primary_scores = torch.stack([inner_score, outer_score], dim=1)
            best_primary_score, best_primary_role = primary_scores.max(dim=1)
            secondary_supported = (
                (secondary_score >= SECONDARY_TEXEL_MIN_PROBABILITY)
                & (secondary_score > best_primary_score)
            )
            consensus_role = torch.where(
                secondary_supported,
                torch.full_like(best_primary_role, ROUTE_SECONDARY),
                best_primary_role,
            )
            consensus_scores = torch.cat(
                [primary_scores, secondary_score.unsqueeze(1)], dim=1
            )
            has_direct_candidate = inner_valid | outer_valid
            route_role[selection] = torch.where(
                has_direct_candidate,
                consensus_role,
                raw_route_role[selection],
            )
            inner_prob[selection] = torch.where(
                has_direct_candidate,
                inner_score.clamp_min(0.0),
                inner_prob[selection],
            )
            outer_prob[selection] = torch.where(
                has_direct_candidate,
                outer_score.clamp_min(0.0),
                outer_prob[selection],
            )
            consensus_top = consensus_scores.topk(2, dim=1).values
            role_margin[selection] = torch.where(
                has_direct_candidate,
                consensus_top[:, 0] - consensus_top[:, 1],
                role_margin[selection],
            )

        visible_outer = (
            (route_role[selection] == ROUTE_OUTER_PRIMARY)
            & (outer_prob[selection] >= outer_threshold)
        )
        # The Steve base layer is opaque and therefore supplies a complete
        # silhouette. Outer-only protrusions still need image foreground evidence.
        choose_outer = outer_valid & visible_outer & (
            inner_valid | (foreground_prob[selection] > fg_threshold)
        )
        choose_inner = inner_valid & (route_role[selection] == ROUTE_INNER_PRIMARY)
        selected_fg = choose_inner | choose_outer

        inner_flat_uv = static["flat_uv"][0].unsqueeze(0).expand_as(layer[selection])
        outer_flat_uv = static["flat_uv"][1].unsqueeze(0).expand_as(layer[selection])
        inner_part = static["part"][0].unsqueeze(0).expand_as(layer[selection])
        outer_part = static["part"][1].unsqueeze(0).expand_as(layer[selection])
        inner_face = static["face"][0].unsqueeze(0).expand_as(layer[selection])
        outer_face = static["face"][1].unsqueeze(0).expand_as(layer[selection])

        fg[selection] = selected_fg
        layer[selection] = choose_outer.long()
        flat_uv[selection] = torch.where(choose_outer, outer_flat_uv, inner_flat_uv)
        surface[selection] = torch.where(
            selected_fg,
            choose_outer.long(),
            torch.full_like(layer[selection], IGNORE_INDEX),
        )
        part[selection] = torch.where(
            selected_fg,
            torch.where(choose_outer, outer_part, inner_part),
            torch.full_like(layer[selection], IGNORE_INDEX),
        )
        face[selection] = torch.where(
            selected_fg,
            torch.where(choose_outer, outer_face, inner_face),
            torch.full_like(layer[selection], IGNORE_INDEX),
        )
        selected_confidence = torch.where(choose_outer, outer_prob[selection], inner_prob[selection])
        margin = role_margin[selection]
        confidence[selection] = selected_confidence
        confidence_margin[selection] = margin
        confidence_margin_ratio[selection] = (
            margin / selected_confidence.clamp_min(1e-8)
        ).clamp(0.0, 1.0)

    secondary = (route_role == ROUTE_SECONDARY) & (foreground_prob > fg_threshold)
    return {
        "foreground": fg,
        "layer": layer,
        "flat_uv": flat_uv,
        "surface": surface,
        "confidence": confidence,
        "confidence_margin": confidence_margin,
        "confidence_margin_ratio": confidence_margin_ratio,
        "semantic_fallback": torch.zeros_like(fg),
        "part": part,
        "face": face,
        "raw_route_role": raw_route_role,
        "route_role": route_role,
        "secondary": secondary,
    }


def _aggregate_surface_scores_by_candidate_texel(candidate_score, static):
    """Average each surface candidate inside its projected Minecraft UV texel."""
    candidate_score = candidate_score.float()
    batch, surface_count, height, width = candidate_score.shape
    flat_uv = static["flat_uv"][:surface_count].reshape(1, surface_count, -1)
    valid = static["masks"][:surface_count].reshape(1, surface_count, -1)
    indices = flat_uv.expand(batch, -1, -1)
    values = candidate_score.flatten(2) * valid.to(dtype=candidate_score.dtype)

    sums = candidate_score.new_zeros(batch, surface_count, UV_SIZE * UV_SIZE)
    sums.scatter_add_(2, indices, values)
    counts = candidate_score.new_zeros(1, surface_count, UV_SIZE * UV_SIZE)
    counts.scatter_add_(
        2,
        flat_uv,
        valid.to(dtype=candidate_score.dtype),
    )
    means = sums / counts.clamp_min(1.0)
    return means.gather(2, indices).reshape(batch, surface_count, height, width)


def _static_candidate_roles(static):
    """Return the Minecraft layer and route role implemented by every surface pixel."""
    candidate_layer = static["layer"].clamp(0, LAYER_CLASSES - 1)
    direct_masks = static["masks"][:LAYER_CLASSES].gather(0, candidate_layer)
    direct_uv = static["flat_uv"][:LAYER_CLASSES].gather(0, candidate_layer)
    primary = static["masks"] & direct_masks & (static["flat_uv"] == direct_uv)
    candidate_role = torch.where(
        primary,
        candidate_layer,
        torch.full_like(candidate_layer, ROUTE_SECONDARY),
    )
    return candidate_layer, candidate_role


def _routing_from_geometry_surface_outputs(
    renderer,
    views,
    outputs,
    fg_threshold=0.5,
    texel_consensus=True,
):
    """Route primary layers first, then use the surface head for secondary UVs.

    The route-role head owns the inner/outer/secondary decision.  The surface
    head only disambiguates renderer slots that implement the selected role.
    Keeping those decisions hierarchical is important: many composite and
    geometry slots overlap a direct cuboid face, so a strong wrong surface
    logit must not turn a primary pixel into a secondary/back-facing pixel.
    """
    views = parse_views(views)
    if not views:
        raise ValueError("At least one renderer view is required for geometry-surface routing.")
    if outputs["layer"].shape[1] != ROUTE_ROLE_CLASSES:
        raise ValueError("Geometry-surface routing requires the three-class route-role head.")
    if "surface" not in outputs:
        raise ValueError("Geometry-surface routing requires a static surface classifier.")

    foreground_prob = torch.sigmoid(outputs["foreground"])[:, 0]
    role_prob = torch.softmax(outputs["layer"].float(), dim=1)
    surface_prob = torch.softmax(outputs["surface"].float(), dim=1)
    raw_route_role = role_prob.argmax(dim=1)
    N, _, H, W = surface_prob.shape
    if N % len(views) != 0:
        raise ValueError(f"N={N} must be divisible by {len(views)} renderer views.")

    fg = torch.zeros((N, H, W), dtype=torch.bool, device=surface_prob.device)
    layer = torch.zeros((N, H, W), dtype=torch.long, device=surface_prob.device)
    flat_uv = torch.zeros_like(layer)
    surface = torch.full_like(layer, IGNORE_INDEX)
    route_role = torch.full_like(layer, IGNORE_INDEX)
    confidence = torch.zeros_like(foreground_prob)
    confidence_margin = torch.zeros_like(foreground_prob)
    confidence_margin_ratio = torch.zeros_like(foreground_prob)
    part = torch.full_like(layer, IGNORE_INDEX)
    face = torch.full_like(layer, IGNORE_INDEX)

    for view_index, view in enumerate(views):
        selection = slice(view_index, N, len(views))
        static = build_static_surface_routing(renderer, view, surface_prob.device)
        if static["masks"].shape[-2:] != (H, W):
            raise ValueError(
                f"View {view!r} mapping shape {tuple(static['masks'].shape[-2:])} "
                f"does not match parser shape {(H, W)}."
            )
        surface_count = static["masks"].shape[0]
        if surface_prob.shape[1] < surface_count:
            raise ValueError(
                f"Parser has {surface_prob.shape[1]} surface classes, but view {view!r} "
                f"requires {surface_count}."
            )

        candidate_layer, candidate_role = _static_candidate_roles(static)

        view_batch = surface_prob[selection].shape[0]
        view_role_prob = role_prob[selection]
        selected_role = raw_route_role[selection]
        selected_role_score = view_role_prob.gather(
            1, selected_role.unsqueeze(1)
        ).squeeze(1)
        role_top = view_role_prob.topk(2, dim=1).values
        role_margin = role_top[:, 0] - role_top[:, 1]

        if texel_consensus:
            direct_scores = _aggregate_role_probabilities_by_direct_texel(
                view_role_prob, static
            )
            inner_valid = static["masks"][0].unsqueeze(0).expand(view_batch, -1, -1)
            outer_valid = static["masks"][1].unsqueeze(0).expand_as(inner_valid)
            unavailable = view_role_prob.new_tensor(-1.0)
            inner_score = torch.where(
                inner_valid,
                direct_scores[:, 0, ROUTE_INNER_PRIMARY],
                unavailable,
            )
            outer_score = torch.where(
                outer_valid,
                direct_scores[:, 1, ROUTE_OUTER_PRIMARY],
                unavailable,
            )
            secondary_score = torch.maximum(
                torch.where(
                    inner_valid,
                    direct_scores[:, 0, ROUTE_SECONDARY],
                    unavailable,
                ),
                torch.where(
                    outer_valid,
                    direct_scores[:, 1, ROUTE_SECONDARY],
                    unavailable,
                ),
            )
            primary_scores = torch.stack([inner_score, outer_score], dim=1)
            best_primary_score, best_primary_role = primary_scores.max(dim=1)
            secondary_supported = (
                (secondary_score >= SECONDARY_TEXEL_MIN_PROBABILITY)
                & (secondary_score > best_primary_score)
            )
            consensus_role = torch.where(
                secondary_supported,
                torch.full_like(best_primary_role, ROUTE_SECONDARY),
                best_primary_role,
            )
            consensus_scores = torch.cat(
                [primary_scores, secondary_score.unsqueeze(1)], dim=1
            )
            has_direct_candidate = inner_valid | outer_valid
            selected_role = torch.where(
                has_direct_candidate,
                consensus_role,
                selected_role,
            )
            consensus_selected_score = consensus_scores.gather(
                1, consensus_role.unsqueeze(1)
            ).squeeze(1)
            selected_role_score = torch.where(
                has_direct_candidate,
                consensus_selected_score,
                selected_role_score,
            )
            consensus_top = consensus_scores.topk(2, dim=1).values
            role_margin = torch.where(
                has_direct_candidate,
                consensus_top[:, 0] - consensus_top[:, 1],
                role_margin,
            )

        candidate_score = surface_prob[selection, :surface_count]
        candidate_valid = static["masks"].unsqueeze(0).expand(
            view_batch, -1, -1, -1
        )
        candidate_valid = candidate_valid & (
            candidate_role.unsqueeze(0) == selected_role.unsqueeze(1)
        )
        if texel_consensus:
            consensus_score = _aggregate_surface_scores_by_candidate_texel(
                candidate_score,
                static,
            )
            local_scores = candidate_score.masked_fill(~candidate_valid, -1.0)
            local_top = local_scores.topk(k=min(2, surface_count), dim=1).values
            if surface_count > 1:
                local_margin_ratio = (
                    (local_top[:, 0] - local_top[:, 1]).clamp_min(0.0)
                    / local_top[:, 0].abs().clamp_min(1e-8)
                ).clamp(0.0, 1.0)
                consensus_weight = (1.0 - local_margin_ratio).unsqueeze(1)
                candidate_score = (
                    candidate_score * (1.0 - consensus_weight)
                    + consensus_score * consensus_weight
                )
        candidate_score = candidate_score.masked_fill(~candidate_valid, -1.0)
        best_surface_score, selected_surface = candidate_score.max(dim=1)
        selected_secondary = selected_role == ROUTE_SECONDARY
        # Primary routes already have an exact direct cuboid surface and UV.
        # Do not let an equivalent composite slot bypass direct-layer coverage
        # filtering; the learned surface head is needed only for secondary UVs.
        selected_surface = torch.where(
            selected_secondary,
            selected_surface,
            selected_role,
        )
        if surface_count > 1:
            top_scores = candidate_score.topk(k=2, dim=1).values
            surface_margin = (top_scores[:, 0] - top_scores[:, 1]).clamp_min(0.0)
            surface_margin_ratio = (
                surface_margin / top_scores[:, 0].abs().clamp_min(1e-8)
            ).clamp(0.0, 1.0)
        else:
            surface_margin = best_surface_score.clamp_min(0.0)
            surface_margin_ratio = (best_surface_score > 0.0).to(
                best_surface_score.dtype
            )

        routed = _select_static_surface(static, selected_surface)
        has_candidate = best_surface_score >= 0.0
        base_silhouette = static["masks"][0].unsqueeze(0).expand(view_batch, -1, -1)
        routed_fg = (
            ((foreground_prob[selection] > fg_threshold) | base_silhouette)
            & has_candidate
            & routed["valid"]
        )
        role_margin_ratio = (
            role_margin / selected_role_score.clamp_min(1e-8)
        ).clamp(0.0, 1.0)
        selected_confidence = torch.where(
            selected_secondary,
            (selected_role_score * best_surface_score.clamp_min(0.0)).sqrt(),
            selected_role_score,
        )
        if "route_confidence" in outputs:
            selected_confidence = selected_confidence * torch.sigmoid(
                outputs["route_confidence"][selection, 0].float()
            )
        selected_margin = torch.where(
            selected_secondary,
            torch.minimum(role_margin, surface_margin),
            role_margin,
        )
        selected_margin_ratio = torch.where(
            selected_secondary,
            torch.minimum(role_margin_ratio, surface_margin_ratio),
            role_margin_ratio,
        )

        fg[selection] = routed_fg
        layer[selection] = routed["layer"]
        flat_uv[selection] = routed["flat_uv"]
        surface[selection] = torch.where(
            routed_fg,
            selected_surface,
            torch.full_like(selected_surface, IGNORE_INDEX),
        )
        route_role[selection] = torch.where(
            routed_fg,
            selected_role,
            torch.full_like(selected_role, IGNORE_INDEX),
        )
        confidence[selection] = selected_confidence
        confidence_margin[selection] = selected_margin
        confidence_margin_ratio[selection] = selected_margin_ratio
        part[selection] = routed["part"]
        face[selection] = routed["face"]

    secondary = fg & (route_role == ROUTE_SECONDARY)
    return {
        "foreground": fg,
        "layer": layer,
        "flat_uv": flat_uv,
        "surface": surface,
        "confidence": confidence,
        "confidence_margin": confidence_margin,
        "confidence_margin_ratio": confidence_margin_ratio,
        "semantic_fallback": torch.zeros_like(fg),
        "part": part,
        "face": face,
        "raw_route_role": raw_route_role,
        "route_role": route_role,
        "secondary": secondary,
        "secondary_routed": secondary,
        "learned_trust": (
            torch.sigmoid(outputs["route_confidence"][:, 0].float())
            if "route_confidence" in outputs
            else torch.ones_like(foreground_prob)
        ),
    }


def _surface_aware_outer_coverage(routing, trusted, selected_outer, renderer, views):
    """Measure outer support within the selected view/surface/UV texel."""
    views = parse_views(views)
    views_per_group = len(views)
    group_count = trusted.shape[0] // views_per_group
    static_views = [
        build_static_surface_routing(renderer, view, trusted.device)
        for view in views
    ]
    surface_count = max(static["masks"].shape[0] for static in static_views)
    uv_count = UV_SIZE * UV_SIZE
    view_stride = surface_count * uv_count
    group_stride = views_per_group * view_stride
    expected = routing["confidence"].new_zeros(group_stride)

    for view_index, static in enumerate(static_views):
        static_surface_count = static["masks"].shape[0]
        surface_offsets = (
            torch.arange(static_surface_count, device=trusted.device).view(-1, 1, 1)
            * uv_count
        )
        expected_indices = (
            view_index * view_stride
            + surface_offsets
            + static["flat_uv"][:static_surface_count]
        )
        expected_mask = (
            static["masks"][:static_surface_count]
            & (static["layer"][:static_surface_count] == 1)
        )
        selected_indices = expected_indices[expected_mask]
        expected.scatter_add_(
            0,
            selected_indices,
            torch.ones(
                selected_indices.shape[0],
                device=trusted.device,
                dtype=expected.dtype,
            ),
        )

    item_indices = torch.arange(trusted.shape[0], device=trusted.device)
    item_groups = item_indices // views_per_group
    item_views = item_indices % views_per_group
    selected_surface = routing["surface"].clamp(0, surface_count - 1)
    route_index = (
        item_views.view(-1, 1, 1) * view_stride
        + selected_surface * uv_count
        + routing["flat_uv"]
    )
    grouped_route_index = route_index + item_groups.view(-1, 1, 1) * group_stride
    observed_mask = trusted & selected_outer
    observed = routing["confidence"].new_zeros(group_count * group_stride)
    observed_indices = grouped_route_index[observed_mask]
    observed.scatter_add_(
        0,
        observed_indices,
        torch.ones(
            observed_indices.shape[0],
            device=trusted.device,
            dtype=observed.dtype,
        ),
    )
    coverage = observed.reshape(group_count, group_stride) / expected.clamp_min(1.0)
    pixel_coverage = coverage[item_groups.view(-1, 1, 1), route_index]
    # Composite and geometry slots can be partially occluded by nearer surfaces,
    # so their static masks are not valid coverage denominators. Their exact slot
    # classifier and texel consensus already reject isolated fragments.
    return torch.where(
        selected_surface >= LAYER_CLASSES,
        torch.ones_like(pixel_coverage),
        pixel_coverage,
    )


def _scatter_soft_uv(rgb_sum, weight_sum, source_rgb, weight, flat_uv, valid):
    """Differentiably accumulate weighted render pixels into fixed UV indices."""
    if not valid.any():
        return rgb_sum, weight_sum
    batch = source_rgb.shape[0]
    indices = flat_uv[valid].reshape(1, -1).expand(batch, -1)
    selected_weight = weight[:, valid]
    selected_rgb = source_rgb[:, :, valid] * selected_weight.unsqueeze(1)
    weight_contribution = weight_sum.new_zeros(batch, UV_SIZE * UV_SIZE).scatter_add(
        1,
        indices,
        selected_weight,
    )
    rgb_contribution = rgb_sum.new_zeros(batch, 3, UV_SIZE * UV_SIZE).scatter_add(
        2,
        indices.unsqueeze(1).expand(-1, 3, -1),
        selected_rgb,
    )
    return rgb_sum + rgb_contribution, weight_sum + weight_contribution


def soft_splat_geometry_predictions_to_uv(
    rendered,
    outputs,
    renderer,
    views,
    group_size=None,
    temperature=1.0,
    canonicalize=True,
    eps=1e-6,
    return_details=False,
):
    """Build a differentiable provisional UV atlas from geometry-parser probabilities.

    Primary inner/outer probabilities are splatted through their direct cuboid
    maps. Secondary probability is distributed through the learned exact
    surface probabilities. UV scatter indices stay fixed while all weights and
    the predicted affine remain differentiable.
    """
    if rendered.dim() != 4:
        raise ValueError(f"Expected rendered NCHW tensor, got {tuple(rendered.shape)}.")
    views = parse_views(views)
    if not views:
        raise ValueError("At least one renderer view is required for soft geometry splatting.")
    if group_size is None:
        group_size = len(views)
    if group_size != len(views):
        raise ValueError(f"group_size={group_size} must equal the number of views ({len(views)}).")
    if rendered.shape[0] % group_size != 0:
        raise ValueError(f"N={rendered.shape[0]} must be divisible by group_size={group_size}.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if outputs["layer"].shape[1] != ROUTE_ROLE_CLASSES or "surface" not in outputs:
        raise ValueError(
            "Soft geometry splatting requires three route-role logits and exact surface logits."
        )

    if canonicalize:
        canonical_outputs = canonicalize_parser_outputs(outputs)
        canonical_rendered = canonicalize_parser_render(
            rendered,
            outputs,
            mode="bilinear",
        ).float()
    else:
        canonical_outputs = outputs
        canonical_rendered = rendered.float()
    role_prob = torch.softmax(canonical_outputs["layer"].float() / temperature, dim=1)
    surface_logits = canonical_outputs["surface"]
    surface_log_normalizer = torch.logsumexp(surface_logits / temperature, dim=1)
    foreground_prob = torch.sigmoid(canonical_outputs["foreground"][:, 0].float())
    groups = rendered.shape[0] // group_size
    reference = canonical_rendered
    rgb_sums = [
        reference.new_zeros(groups, 3, UV_SIZE * UV_SIZE)
        for _ in range(LAYER_CLASSES)
    ]
    weight_sums = [
        reference.new_zeros(groups, UV_SIZE * UV_SIZE)
        for _ in range(LAYER_CLASSES)
    ]
    expected = reference.new_zeros(LAYER_CLASSES, UV_SIZE * UV_SIZE)

    for view_index, view in enumerate(views):
        selection = slice(view_index, rendered.shape[0], group_size)
        static = build_static_surface_routing(renderer, view, rendered.device)
        if static["masks"].shape[-2:] != canonical_rendered.shape[-2:]:
            raise ValueError(
                f"View {view!r} mapping shape {tuple(static['masks'].shape[-2:])} "
                f"does not match parser shape {tuple(canonical_rendered.shape[-2:])}."
            )
        surface_count = static["masks"].shape[0]
        if surface_logits.shape[1] < surface_count:
            raise ValueError(
                f"Parser has {surface_logits.shape[1]} surface classes, but view {view!r} "
                f"requires {surface_count}."
            )

        source_rgb = canonical_rendered[selection, :3]
        view_foreground = foreground_prob[selection]
        view_role_prob = role_prob[selection]
        view_surface_logits = surface_logits[selection]
        view_surface_log_normalizer = surface_log_normalizer[selection]

        for layer_index in range(LAYER_CLASSES):
            valid = static["masks"][layer_index]
            flat_uv = static["flat_uv"][layer_index]
            weight = view_foreground * view_role_prob[:, layer_index]
            rgb_sums[layer_index], weight_sums[layer_index] = _scatter_soft_uv(
                rgb_sums[layer_index],
                weight_sums[layer_index],
                source_rgb,
                weight,
                flat_uv,
                valid,
            )
            expected[layer_index].scatter_add_(
                0,
                flat_uv[valid],
                torch.ones_like(flat_uv[valid], dtype=expected.dtype),
            )

        candidate_layer, candidate_role = _static_candidate_roles(static)
        for surface_index in range(LAYER_CLASSES, surface_count):
            secondary = (
                static["masks"][surface_index]
                & (candidate_role[surface_index] == ROUTE_SECONDARY)
            )
            if not secondary.any():
                continue
            surface_weight = (
                view_foreground
                * view_role_prob[:, ROUTE_SECONDARY]
                * torch.exp(
                    view_surface_logits[:, surface_index].float() / temperature
                    - view_surface_log_normalizer.float()
                )
            )
            for layer_index in range(LAYER_CLASSES):
                valid = secondary & (candidate_layer[surface_index] == layer_index)
                if not valid.any():
                    continue
                rgb_sums[layer_index], weight_sums[layer_index] = _scatter_soft_uv(
                    rgb_sums[layer_index],
                    weight_sums[layer_index],
                    source_rgb,
                    surface_weight,
                    static["flat_uv"][surface_index],
                    valid,
                )

    stacked_rgb_sum = torch.stack(rgb_sums, dim=1)
    stacked_weight_sum = torch.stack(weight_sums, dim=1).unsqueeze(2)
    expected = expected.view(1, LAYER_CLASSES, 1, UV_SIZE * UV_SIZE).clamp_min(1.0)
    layer_alpha = (stacked_weight_sum / expected).clamp(0.0, 1.0)
    total_rgb_sum = stacked_rgb_sum.sum(dim=1)
    total_weight = stacked_weight_sum.sum(dim=1)
    rgb = total_rgb_sum / total_weight.clamp_min(eps)
    alpha = layer_alpha.amax(dim=1)
    pred_uv = torch.cat([rgb, alpha], dim=1).reshape(groups, 4, UV_SIZE, UV_SIZE)
    support = total_weight.reshape(groups, 1, UV_SIZE, UV_SIZE)
    if return_details:
        return pred_uv, {
            "support": support,
            "layer_weight": stacked_weight_sum.reshape(
                groups, LAYER_CLASSES, UV_SIZE, UV_SIZE
            ),
            "layer_alpha": layer_alpha.reshape(
                groups, LAYER_CLASSES, UV_SIZE, UV_SIZE
            ),
            "canonical_rendered": canonical_rendered,
            "canonical_outputs": canonical_outputs,
        }
    return pred_uv


def render_direct_uv(skins, renderer, view):
    """Differentiably render the direct inner/outer cuboids without deep composites."""
    if skins.dim() != 4 or skins.shape[1:] != (4, UV_SIZE, UV_SIZE):
        raise ValueError(f"Expected skins shaped (N, 4, 64, 64), got {tuple(skins.shape)}.")
    batch = skins.shape[0]
    dtype = skins.dtype
    inner_grid = getattr(renderer, f"{view}_inner_grid").to(dtype=dtype)
    outer_grid = getattr(renderer, f"{view}_outer_grid").to(dtype=dtype)
    inner_mask = getattr(renderer, f"{view}_inner_mask").to(dtype=dtype)
    outer_mask = getattr(renderer, f"{view}_outer_mask").to(dtype=dtype)
    inner = F.grid_sample(
        skins,
        inner_grid.unsqueeze(0).expand(batch, -1, -1, -1),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ) * inner_mask.view(1, 1, *inner_mask.shape)
    outer = F.grid_sample(
        skins,
        outer_grid.unsqueeze(0).expand(batch, -1, -1, -1),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ) * outer_mask.view(1, 1, *outer_mask.shape)
    bg = renderer.bg_color.to(device=skins.device, dtype=dtype).view(1, 3, 1, 1)
    inner_rgb = inner[:, 3:4] * inner[:, :3] + (1.0 - inner[:, 3:4]) * bg
    rgb = outer[:, 3:4] * outer[:, :3] + (1.0 - outer[:, 3:4]) * inner_rgb
    alpha = outer[:, 3:4] + (1.0 - outer[:, 3:4]) * inner[:, 3:4]
    return torch.cat([rgb, alpha], dim=1)


def flat_uv_to_uv01(flat_uv, dtype):
    return torch.stack(
        [
            (flat_uv.remainder(UV_SIZE) / float(UV_SIZE - 1)).to(dtype),
            (flat_uv.div(UV_SIZE, rounding_mode="floor") / float(UV_SIZE - 1)).to(dtype),
        ],
        dim=1,
    )


def splat_parser_predictions_to_uv_conditioning(
    rendered,
    outputs,
    renderer=None,
    views=None,
    group_size=1,
    fg_threshold=0.5,
    bg_color=(128, 128, 128),
    semantic_gate=True,
    affine_refine=True,
    affine_refine_translation_px=8.0,
    affine_refine_scale=0.0,
    route_confidence_threshold=0.0,
    route_margin_threshold=0.0,
    outer_route_confidence_threshold=None,
    outer_route_margin_threshold=None,
    outer_uv_min_coverage=0.5,
    color_aggregation="exact_mode",
    geometry_outer_threshold=0.5,
    geometry_route_texel_consensus=True,
    observed_foreground=None,
    reject_semantic_fallback=False,
    reject_inner_semantic_fallback=False,
    include_confidence=False,
    return_details=False,
):
    """Route parser outputs to UV, using static mappings for affine-parser checkpoints."""
    if not 0.0 <= route_confidence_threshold <= 1.0:
        raise ValueError("route_confidence_threshold must be in [0, 1].")
    if not 0.0 <= route_margin_threshold <= 1.0:
        raise ValueError("route_margin_threshold must be in [0, 1].")
    if outer_route_confidence_threshold is None:
        outer_route_confidence_threshold = route_confidence_threshold
    if outer_route_margin_threshold is None:
        outer_route_margin_threshold = route_margin_threshold
    if not 0.0 <= outer_route_confidence_threshold <= 1.0:
        raise ValueError("outer_route_confidence_threshold must be in [0, 1].")
    if not 0.0 <= outer_route_margin_threshold <= 1.0:
        raise ValueError("outer_route_margin_threshold must be in [0, 1].")
    if not 0.0 <= outer_uv_min_coverage <= 1.0:
        raise ValueError("outer_uv_min_coverage must be in [0, 1].")
    if "affine" not in outputs:
        conditioning = splat_predictions_to_uv_conditioning(
            rendered,
            outputs,
            group_size=group_size,
            fg_threshold=fg_threshold,
            bg_color=bg_color,
        )
        if return_details:
            return conditioning, {"rendered": rendered, "outputs": outputs, "routing": None, "alignment": None}
        return conditioning

    if renderer is None or views is None:
        raise ValueError("Affine parser routing requires both renderer and views.")
    views = parse_views(views)
    if group_size != len(views):
        raise ValueError(f"group_size={group_size} must equal the number of views ({len(views)}).")

    if observed_foreground is None:
        observed_foreground = estimate_solid_background_foreground(rendered)
    else:
        if observed_foreground.shape != rendered.shape[:1] + rendered.shape[-2:]:
            raise ValueError(
                "observed_foreground must have shape "
                f"{rendered.shape[:1] + rendered.shape[-2:]}, got {tuple(observed_foreground.shape)}."
            )
        observed_foreground = observed_foreground.to(
            device=rendered.device, dtype=torch.bool
        )
    source_background_rgb, _ = estimate_solid_background_color(rendered)
    source_fill = rendered.new_ones(rendered.shape[0], rendered.shape[1], 1, 1)
    source_fill[:, :3] = source_background_rgb.to(dtype=rendered.dtype)

    alignment = None
    routing_outputs = outputs
    if affine_refine:
        refined_affine, alignment = refine_parser_affine(
            outputs,
            renderer,
            views,
            translation_radius_px=affine_refine_translation_px,
            scale_radius=affine_refine_scale,
            observed_foreground=observed_foreground,
        )
        routing_outputs = dict(outputs)
        routing_outputs["affine"] = refined_affine

    canonical_rendered = canonicalize_parser_render(
        rendered,
        routing_outputs,
        mode="nearest",
        fill_color=source_fill,
    )
    canonical_observed_foreground = canonicalize_tensor(
        observed_foreground.unsqueeze(1).to(dtype=rendered.dtype),
        routing_outputs["affine"],
        mode="nearest",
    )[:, 0] > 0.5
    canonical_outputs = canonicalize_parser_outputs(routing_outputs)
    if "surface" in canonical_outputs and "part" not in canonical_outputs:
        routing = _routing_from_geometry_surface_outputs(
            renderer,
            views,
            canonical_outputs,
            fg_threshold=fg_threshold,
            texel_consensus=geometry_route_texel_consensus,
        )
    elif "surface" not in canonical_outputs and "part" not in canonical_outputs:
        routing = _routing_from_geometry_outputs(
            renderer,
            views,
            canonical_outputs,
            fg_threshold=fg_threshold,
            outer_threshold=geometry_outer_threshold,
            texel_consensus=geometry_route_texel_consensus,
        )
    else:
        routing = _routing_from_affine_outputs(
            renderer,
            views,
            canonical_outputs,
            fg_threshold=fg_threshold,
            semantic_gate=semantic_gate,
        )
    raw_foreground = routing["foreground"]
    routing["secondary"] = routing.get(
        "secondary", torch.zeros_like(raw_foreground)
    ) & canonical_observed_foreground
    selected_outer = routing["layer"] == 1
    confidence_threshold = torch.where(
        selected_outer,
        routing["confidence"].new_tensor(outer_route_confidence_threshold),
        routing["confidence"].new_tensor(route_confidence_threshold),
    )
    margin_threshold = torch.where(
        selected_outer,
        routing["confidence_margin_ratio"].new_tensor(outer_route_margin_threshold),
        routing["confidence_margin_ratio"].new_tensor(route_margin_threshold),
    )
    trusted = (
        raw_foreground
        & canonical_observed_foreground
        & (routing["confidence"] >= confidence_threshold)
        & (routing["confidence_margin_ratio"] >= margin_threshold)
    )
    if reject_semantic_fallback:
        rejected_fallback = routing["semantic_fallback"] & (
            selected_outer | reject_inner_semantic_fallback
        )
        trusted = trusted & ~rejected_fallback
    outer_uv_coverage = torch.ones_like(routing["confidence"])
    if outer_uv_min_coverage > 0.0:
        views_per_group = len(views)
        group_count = trusted.shape[0] // views_per_group
        if "surface" in canonical_outputs:
            pixel_coverage = _surface_aware_outer_coverage(
                routing,
                trusted,
                selected_outer,
                renderer,
                views,
            )
        else:
            expected = routing["confidence"].new_zeros(UV_SIZE * UV_SIZE)
            for view in views:
                static = build_static_surface_routing(renderer, view, trusted.device)
                expected_uv = static["flat_uv"][1][static["masks"][1]]
                expected.scatter_add_(
                    0,
                    expected_uv,
                    torch.ones(expected_uv.shape[0], device=trusted.device, dtype=expected.dtype),
                )

            item_groups = torch.arange(trusted.shape[0], device=trusted.device) // views_per_group
            grouped_uv = routing["flat_uv"] + item_groups.view(-1, 1, 1) * (UV_SIZE * UV_SIZE)
            observed_mask = trusted & selected_outer
            observed = routing["confidence"].new_zeros(group_count * UV_SIZE * UV_SIZE)
            observed_uv = grouped_uv[observed_mask]
            observed.scatter_add_(
                0,
                observed_uv,
                torch.ones(observed_uv.shape[0], device=trusted.device, dtype=observed.dtype),
            )
            coverage = observed.reshape(group_count, UV_SIZE * UV_SIZE) / expected.clamp_min(1.0)
            pixel_coverage = coverage[item_groups.view(-1, 1, 1), routing["flat_uv"]]
        outer_uv_coverage = torch.where(selected_outer, pixel_coverage, outer_uv_coverage)
        trusted = trusted & (~selected_outer | (pixel_coverage >= outer_uv_min_coverage))
    routing["raw_foreground"] = raw_foreground
    routing["observed_foreground"] = canonical_observed_foreground
    routing["background_rejected"] = raw_foreground & ~canonical_observed_foreground
    routing["rejected"] = raw_foreground & ~trusted
    routing["foreground"] = trusted
    if "secondary_routed" in routing:
        routing["secondary_rejected"] = routing["secondary"] & ~trusted
        routing["secondary_routed"] = routing["secondary"] & trusted
    routing["outer_uv_coverage"] = outer_uv_coverage
    conditioning = splat_to_uv_conditioning(
        canonical_rendered,
        routing["foreground"],
        routing["layer"],
        routing["flat_uv"],
        group_size=group_size,
        bg_color=bg_color,
        confidence=routing["confidence"],
        color_aggregation=color_aggregation,
        include_confidence=include_confidence,
    )
    if return_details:
        return conditioning, {
            "rendered": canonical_rendered,
            "outputs": canonical_outputs,
            "routing": routing,
            "alignment": alignment,
        }
    return conditioning


def splat_deterministic_targets_to_uv_conditioning(
    rendered,
    targets,
    renderer,
    views,
    group_size=1,
    bg_color=(128, 128, 128),
):
    """Splat ground-truth labels through the same fixed mapping used by affine mode."""
    views = parse_views(views)
    if group_size != len(views):
        raise ValueError(f"group_size={group_size} must equal the number of views ({len(views)}).")
    canonical_rendered = canonicalize_tensor(rendered, targets["affine"], mode="nearest")
    canonical_targets = canonicalize_dense_targets(targets)
    requested_surface = canonical_targets["surface"]
    N, H, W = requested_surface.shape
    fg = torch.zeros_like(requested_surface, dtype=torch.bool)
    routed_layer = torch.zeros_like(requested_surface)
    flat_uv = torch.zeros_like(requested_surface)

    for view_index, view in enumerate(views):
        selection = slice(view_index, N, len(views))
        static = build_static_surface_routing(renderer, view, requested_surface.device)
        if static["masks"].shape[-2:] != (H, W):
            raise ValueError(
                f"View {view!r} mapping shape {tuple(static['masks'].shape[-2:])} does not match target shape {(H, W)}."
            )
        routed = _select_static_surface(static, requested_surface[selection])
        fg[selection] = (
            (canonical_targets["foreground"][selection, 0] > 0.5)
            & (requested_surface[selection] != IGNORE_INDEX)
            & routed["valid"]
        )
        routed_layer[selection] = routed["layer"]
        flat_uv[selection] = routed["flat_uv"]

    return splat_to_uv_conditioning(
        canonical_rendered,
        fg,
        routed_layer,
        flat_uv,
        group_size=group_size,
        bg_color=bg_color,
    )


def splat_predictions_to_uv_conditioning(
    rendered,
    outputs,
    group_size=1,
    fg_threshold=0.5,
    bg_color=(128, 128, 128),
):
    """Splat parser predictions back to the 10-channel semantic_uv_reconstruction conditioning layout."""
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


def _select_exact_mode_candidates(values, target_uv, scores):
    """Select one real source pixel from the most frequent 8-bit RGB value per UV texel."""
    if target_uv.numel() == 0:
        return values, target_uv, scores

    rgb8 = (values[:3].clamp(0.0, 1.0) * 255.0).round().long()
    color_code = rgb8[0] | (rgb8[1] << 8) | (rgb8[2] << 16)
    color_count = 1 << 24
    combined_key = target_uv.long() * color_count + color_code
    unique_keys, inverse = torch.unique(combined_key, sorted=False, return_inverse=True)
    votes = scores.new_zeros(unique_keys.shape[0])
    votes.index_add_(0, inverse, torch.ones_like(scores))

    positions = torch.arange(target_uv.shape[0], device=target_uv.device)
    first_position = torch.full(
        (unique_keys.shape[0],),
        target_uv.shape[0],
        dtype=torch.long,
        device=target_uv.device,
    )
    first_position.scatter_reduce_(0, inverse, positions, reduce="amin", include_self=True)

    unique_uv = unique_keys.div(color_count, rounding_mode="floor")
    best_votes = votes.new_full((UV_SIZE * UV_SIZE,), -torch.inf)
    best_votes.scatter_reduce_(0, unique_uv, votes, reduce="amax", include_self=True)
    tied_winner = votes >= (best_votes[unique_uv] - 1e-7)
    earliest_winner = torch.full(
        (UV_SIZE * UV_SIZE,),
        target_uv.shape[0],
        dtype=torch.long,
        device=target_uv.device,
    )
    earliest_winner.scatter_reduce_(
        0,
        unique_uv[tied_winner],
        first_position[tied_winner],
        reduce="amin",
        include_self=True,
    )
    winning_unique = tied_winner & (first_position == earliest_winner[unique_uv])
    winning_key = torch.full(
        (UV_SIZE * UV_SIZE,),
        -1,
        dtype=torch.long,
        device=target_uv.device,
    )
    winning_key[unique_uv[winning_unique]] = unique_keys[winning_unique]
    in_winning_color = combined_key == winning_key[target_uv]

    values = values[:, in_winning_color]
    target_uv = target_uv[in_winning_color]
    scores = scores[in_winning_color]
    best_scores = scores.new_full((UV_SIZE * UV_SIZE,), -torch.inf)
    best_scores.scatter_reduce_(0, target_uv, scores, reduce="amax", include_self=True)
    tied_representative = scores >= (best_scores[target_uv] - 1e-7)
    positions = torch.arange(target_uv.shape[0], device=target_uv.device)
    first_representative = torch.full(
        (UV_SIZE * UV_SIZE,),
        target_uv.shape[0],
        dtype=torch.long,
        device=target_uv.device,
    )
    first_representative.scatter_reduce_(
        0,
        target_uv[tied_representative],
        positions[tied_representative],
        reduce="amin",
        include_self=True,
    )
    selected = tied_representative & (positions == first_representative[target_uv])
    return values[:, selected], target_uv[selected], scores[selected]


def splat_to_uv_conditioning(
    rendered,
    fg,
    layer,
    flat_uv,
    group_size=1,
    bg_color=(128, 128, 128),
    confidence=None,
    color_aggregation="exact_mode",
    include_confidence=False,
):
    if rendered.dim() != 4:
        raise ValueError(f"Expected rendered tensor as NCHW, got {tuple(rendered.shape)}.")
    N, _, _, _ = rendered.shape
    if N % group_size != 0:
        raise ValueError(f"N={N} must be divisible by group_size={group_size}.")
    if color_aggregation not in SPLAT_COLOR_AGGREGATIONS:
        raise ValueError(
            f"Unknown color_aggregation={color_aggregation!r}; "
            f"expected one of {SPLAT_COLOR_AGGREGATIONS}."
        )

    groups = N // group_size
    device = rendered.device
    dtype = rendered.dtype
    accum = rendered.new_zeros(groups, LAYER_CLASSES, 4, UV_SIZE * UV_SIZE)
    counts = rendered.new_zeros(groups, LAYER_CLASSES, 1, UV_SIZE * UV_SIZE)
    confidence_map = rendered.new_zeros(
        groups, LAYER_CLASSES, 1, UV_SIZE * UV_SIZE
    )

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
            if color_aggregation == "exact_mode":
                values, target_uv, scores = _select_exact_mode_candidates(
                    values, target_uv, scores
                )
            if color_aggregation == "best" and select_highest_confidence:
                best_scores = scores.new_full((UV_SIZE * UV_SIZE,), -torch.inf)
                best_scores.scatter_reduce_(0, target_uv, scores, reduce="amax", include_self=True)
                is_best = scores >= (best_scores[target_uv] - 1e-7)
                candidate_indices = torch.arange(target_uv.shape[0], device=device)
                first_best = torch.full(
                    (UV_SIZE * UV_SIZE,),
                    target_uv.shape[0],
                    dtype=torch.long,
                    device=device,
                )
                first_best.scatter_reduce_(
                    0,
                    target_uv[is_best],
                    candidate_indices[is_best],
                    reduce="amin",
                    include_self=True,
                )
                selected_indices = first_best[first_best < target_uv.shape[0]]
                values = values[:, selected_indices]
                target_uv = target_uv[selected_indices]
                scores = scores[selected_indices]

            accum[group, layer_index].index_add_(1, target_uv, values)
            counts[group, layer_index, 0].index_add_(
                0,
                target_uv,
                torch.ones(target_uv.shape[0], dtype=dtype, device=device),
            )
            confidence_map[group, layer_index, 0].scatter_reduce_(
                0,
                target_uv,
                scores.to(dtype=dtype),
                reduce="amax",
                include_self=True,
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
    layer_channels = [rgb, alpha, known]
    if include_confidence:
        layer_channels.append(confidence_map * known)
    layers = torch.cat(layer_channels, dim=2).reshape(
        groups, -1, UV_SIZE, UV_SIZE
    )
    return layers.clamp(0.0, 1.0)


def conditioning_to_pred_uv(conditioning):
    """Merge two-layer parser conditioning into a preliminary RGBA skin atlas.

    Conditioning stores either ``RGBA + evidence`` (legacy 10-channel input)
    or ``RGBA + evidence + confidence`` (12-channel input) for each layer.
    Their Minecraft UV rectangles normally do not overlap, but outer data wins
    if both layers mark the same texel as observed.
    Unknown RGB retains the conditioning background as a useful placeholder;
    unknown alpha remains transparent until Minecraft base-alpha finalization.
    """
    squeeze_batch = conditioning.dim() == 3
    if squeeze_batch:
        conditioning = conditioning.unsqueeze(0)
    if conditioning.dim() != 4 or conditioning.shape[1] not in (10, 12):
        raise ValueError(
            "Expected 10- or 12-channel parser conditioning, "
            f"got {tuple(conditioning.shape)}."
        )

    inner_rgba = conditioning[:, 0:4]
    inner_known = conditioning[:, 4:5] > 0.5
    outer_offset = 6 if conditioning.shape[1] == 12 else 5
    outer_rgba = conditioning[:, outer_offset : outer_offset + 4]
    outer_known = conditioning[:, outer_offset + 4 : outer_offset + 5] > 0.5
    known = inner_known | outer_known

    rgba = torch.where(outer_known.expand_as(outer_rgba), outer_rgba, inner_rgba)
    rgba[:, 3:4] = torch.where(
        known,
        rgba[:, 3:4],
        torch.zeros_like(rgba[:, 3:4]),
    )
    rgba = rgba.clamp(0.0, 1.0)
    return rgba[0] if squeeze_batch else rgba


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
ROUTE_ROLE_PALETTE = (
    (72, 169, 166),
    (255, 179, 71),
    (218, 112, 232),
)
FACE_NAMES = ("front", "back", "left", "right", "top", "bottom")
FACE_PALETTE = (
    (239, 83, 80),
    (171, 71, 188),
    (66, 165, 245),
    (38, 166, 154),
    (255, 202, 40),
    (117, 117, 117),
)
LAYER_FACE_PALETTE = (
    (183, 28, 28),
    (106, 27, 154),
    (21, 101, 192),
    (0, 121, 107),
    (245, 166, 35),
    (84, 84, 84),
    (255, 128, 125),
    (218, 112, 232),
    (126, 208, 255),
    (102, 221, 203),
    (255, 224, 130),
    (210, 210, 210),
)


def combine_layer_face(layer, face):
    """Encode inner/outer x six cube faces as 12 visualization classes."""
    valid = (layer != IGNORE_INDEX) & (face != IGNORE_INDEX)
    combined = torch.full_like(layer, IGNORE_INDEX)
    combined[valid] = layer[valid] * FACE_CLASSES + face[valid]
    return combined


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


def build_geometry_grid_debug(renderer, views, item_count, reference, bg_color=(128, 128, 128)):
    """Build fitted inner/outer cuboid face maps with projected UV texel boundaries."""
    views = parse_views(views)
    if not views:
        raise ValueError("At least one view is required for geometry-grid debug output.")
    colors = [[], []]
    masks = [[], []]
    edges = [[], []]
    for item_index in range(item_count):
        static = build_static_surface_routing(renderer, views[item_index % len(views)], reference.device)
        for layer_index in range(LAYER_CLASSES):
            mask = static["masks"][layer_index]
            face = static["face"][layer_index]
            flat_uv = static["flat_uv"][layer_index]
            labels = combine_layer_face(
                torch.full_like(face, layer_index),
                torch.where(mask, face, torch.full_like(face, IGNORE_INDEX)),
            ).unsqueeze(0)
            color = colorize_labels(labels, LAYER_FACE_PALETTE, bg_color, reference[:1])[0]

            edge = torch.zeros_like(mask)
            different_x = (flat_uv[:, 1:] != flat_uv[:, :-1]) | (face[:, 1:] != face[:, :-1])
            different_y = (flat_uv[1:, :] != flat_uv[:-1, :]) | (face[1:, :] != face[:-1, :])
            edge[:, 1:] |= mask[:, 1:] & mask[:, :-1] & different_x
            edge[1:, :] |= mask[1:, :] & mask[:-1, :] & different_y
            interior = -F.max_pool2d(-mask.float().unsqueeze(0).unsqueeze(0), 3, 1, 1)[0, 0]
            edge |= mask & (interior < 0.5)
            edge_color = reference.new_tensor((0.08, 0.08, 0.08)).view(3, 1, 1)
            color = torch.where(edge.unsqueeze(0), edge_color, color)

            colors[layer_index].append(color)
            masks[layer_index].append(mask)
            edges[layer_index].append(edge)

    return (
        torch.stack(colors[0]),
        torch.stack(colors[1]),
        torch.stack(masks[0]),
        torch.stack(masks[1]),
        torch.stack(edges[0]),
        torch.stack(edges[1]),
    )


def fill_geometry_grid_debug(rendered, foreground, layer, geometry_debug, bg_color=(128, 128, 128)):
    """Show only source RGB pixels actually routed to each predicted layer."""
    del geometry_debug
    rgb = rendered[:, :3]
    bg = rgb.new_tensor(bg_color).view(1, 3, 1, 1) / 255.0

    def fill(layer_index):
        classified = (foreground & (layer == layer_index)).unsqueeze(1)
        return torch.where(classified, rgb, bg.expand_as(rgb))

    return fill(0), fill(1)


def overlay_geometry_grid_debug(
    rendered,
    geometry_debug,
    tint_alpha=0.12,
    base_images=None,
):
    """Overlay fitted inner/outer UV texel grids on the canonical source image."""
    if not 0.0 <= tint_alpha <= 1.0:
        raise ValueError(f"tint_alpha must be in [0, 1], got {tint_alpha}.")

    rgb = rendered[:, :3]
    if base_images is None:
        base_images = (rgb, rgb)
    if len(base_images) != LAYER_CLASSES:
        raise ValueError(
            f"base_images must contain {LAYER_CLASSES} tensors, got {len(base_images)}."
        )
    masks = geometry_debug[2:4]
    edges = geometry_debug[4:6]
    fill_colors = (
        rgb.new_tensor((0.10, 0.85, 0.95)),
        rgb.new_tensor((1.00, 0.65, 0.10)),
    )
    edge_colors = (
        rgb.new_tensor((0.00, 1.00, 1.00)),
        rgb.new_tensor((1.00, 0.25, 0.85)),
    )

    overlays = []
    for base, mask, edge, fill_color, edge_color in zip(
        base_images, masks, edges, fill_colors, edge_colors
    ):
        if base.shape != rgb.shape:
            raise ValueError(
                f"Geometry overlay base shape {tuple(base.shape)} does not match RGB {tuple(rgb.shape)}."
            )
        fill_color = fill_color.view(1, 3, 1, 1)
        edge_color = edge_color.view(1, 3, 1, 1)
        tinted = base * (1.0 - tint_alpha) + fill_color * tint_alpha
        overlay = torch.where(mask.unsqueeze(1), tinted, base)
        overlay = torch.where(edge.unsqueeze(1), edge_color, overlay)
        overlays.append(overlay)
    return tuple(overlays)


def colorize_surface(labels, bg_color, reference):
    """Colorize fixed renderer surface slots without assuming a fixed slot count."""
    N, H, W = labels.shape
    bg = bg_tensor(bg_color, reference).expand(N, 3, H, W).clone()
    valid = labels != IGNORE_INDEX
    if not valid.any():
        return bg
    surface = labels.clamp_min(0).to(dtype=reference.dtype)
    colors = torch.stack(
        [
            (surface.mul(47.0).remainder(191.0) + 32.0) / 255.0,
            (surface.mul(83.0).remainder(191.0) + 32.0) / 255.0,
            (surface.mul(131.0).remainder(191.0) + 32.0) / 255.0,
        ],
        dim=1,
    )
    return torch.where(valid.unsqueeze(1), colors, bg)


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
