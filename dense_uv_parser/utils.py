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
    static = {
        "grids": surface_grids,
        "masks": surface_masks,
        "flat_uv": metadata["flat_uv"],
        "layer": metadata["layer"],
        "part": metadata["part"],
        "face": metadata["face"],
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
    part = torch.where(valid, selected["part"], torch.full_like(surface, IGNORE_INDEX))
    face = torch.where(valid, selected["face"], torch.full_like(surface, IGNORE_INDEX))
    uv = flat_uv_to_uv01(selected["flat_uv"], dtype).masked_fill(~valid.unsqueeze(1), 0.0)

    targets = {
        "foreground": valid.unsqueeze(1).to(dtype=dtype),
        "layer": layer,
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
    translation_scale=0.03,
    scale_range=0.03,
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
        if name in {"affine", "uv", "uv_x", "uv_y"} or not torch.is_tensor(value) or value.dim() != 4:
            canonical[name] = value
        else:
            canonical[name] = canonicalize_tensor(value, affine, mode="bilinear")
    return canonical


def canonicalize_parser_render(rendered, outputs):
    if "affine" not in outputs:
        return rendered
    return canonicalize_tensor(rendered, outputs["affine"], mode="bilinear")


def _routing_from_affine_outputs(renderer, views, outputs, fg_threshold=0.5, semantic_gate=True):
    views = parse_views(views)
    if not views:
        raise ValueError("At least one renderer view is required for deterministic UV routing.")
    if "surface" not in outputs:
        raise ValueError("Affine parser outputs must include the static surface classifier.")
    foreground_prob = torch.sigmoid(outputs["foreground"])[:, 0]
    surface_prob = torch.softmax(outputs["surface"], dim=1)
    requested_surface = surface_prob.argmax(dim=1)
    layer_prob = torch.softmax(outputs["layer"], dim=1)
    N, H, W = requested_surface.shape
    if N % len(views) != 0:
        raise ValueError(f"N={N} must be divisible by {len(views)} renderer views.")

    fg = torch.zeros_like(foreground_prob, dtype=torch.bool)
    layer = torch.zeros_like(requested_surface)
    flat_uv = torch.zeros_like(requested_surface)
    surface = torch.full_like(requested_surface, IGNORE_INDEX)
    confidence = torch.zeros_like(foreground_prob)
    expected_part = torch.full_like(requested_surface, IGNORE_INDEX)
    expected_face = torch.full_like(requested_surface, IGNORE_INDEX)

    for view_index, view in enumerate(views):
        selection = slice(view_index, N, len(views))
        static = build_static_surface_routing(renderer, view, requested_surface.device)
        if static["masks"].shape[-2:] != (H, W):
            raise ValueError(
                f"View {view!r} mapping shape {tuple(static['masks'].shape[-2:])} does not match parser shape {(H, W)}."
            )
        selected_surface = requested_surface[selection]
        routed = _select_static_surface(static, selected_surface)
        selected_layer = routed["layer"]
        selected_part = routed["part"]
        selected_face = routed["face"]
        surface_score = surface_prob[selection].gather(1, selected_surface.unsqueeze(1)).squeeze(1)
        semantic_valid = torch.ones_like(routed["valid"], dtype=torch.bool)
        semantic_score = torch.ones_like(surface_score)

        for name, expected in (("layer", selected_layer), ("part", selected_part), ("face", selected_face)):
            if name not in outputs:
                continue
            probabilities = torch.softmax(outputs[name][selection], dim=1)
            known_semantic = expected != IGNORE_INDEX
            expected_safe = expected.clamp(0, probabilities.shape[1] - 1)
            expected_probability = probabilities.gather(1, expected_safe.unsqueeze(1)).squeeze(1)
            semantic_score = semantic_score * torch.where(
                known_semantic,
                expected_probability,
                torch.ones_like(expected_probability),
            )
            if semantic_gate:
                semantic_valid = semantic_valid & (
                    ~known_semantic | (probabilities.argmax(dim=1) == expected_safe)
                )

        routed_fg = (
            (foreground_prob[selection] > fg_threshold)
            & routed["valid"]
            & semantic_valid
        )
        fg[selection] = routed_fg
        layer[selection] = selected_layer
        flat_uv[selection] = routed["flat_uv"]
        surface[selection] = selected_surface
        confidence[selection] = foreground_prob[selection] * surface_score * semantic_score.sqrt()
        expected_part[selection] = selected_part
        expected_face[selection] = selected_face

    return {
        "foreground": fg,
        "layer": layer,
        "flat_uv": flat_uv,
        "surface": surface,
        "confidence": confidence,
        "part": expected_part,
        "face": expected_face,
    }


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
    return_details=False,
):
    """Route parser outputs to UV, using static mappings for affine-parser checkpoints."""
    if "affine" not in outputs:
        conditioning = splat_predictions_to_uv_conditioning(
            rendered,
            outputs,
            group_size=group_size,
            fg_threshold=fg_threshold,
            bg_color=bg_color,
        )
        if return_details:
            return conditioning, {"rendered": rendered, "outputs": outputs, "routing": None}
        return conditioning

    if renderer is None or views is None:
        raise ValueError("Affine parser routing requires both renderer and views.")
    views = parse_views(views)
    if group_size != len(views):
        raise ValueError(f"group_size={group_size} must equal the number of views ({len(views)}).")

    canonical_rendered = canonicalize_parser_render(rendered, outputs)
    canonical_outputs = canonicalize_parser_outputs(outputs)
    routing = _routing_from_affine_outputs(
        renderer,
        views,
        canonical_outputs,
        fg_threshold=fg_threshold,
        semantic_gate=semantic_gate,
    )
    conditioning = splat_to_uv_conditioning(
        canonical_rendered,
        routing["foreground"],
        routing["layer"],
        routing["flat_uv"],
        group_size=group_size,
        bg_color=bg_color,
        confidence=routing["confidence"],
    )
    if return_details:
        return conditioning, {
            "rendered": canonical_rendered,
            "outputs": canonical_outputs,
            "routing": routing,
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
    canonical_rendered = canonicalize_tensor(rendered, targets["affine"], mode="bilinear")
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
