import argparse
import json
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TOOLKIT_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from SkingToolkit.dense_uv_parser.losses import DenseUVParserLoss  # noqa: E402
from SkingToolkit.dense_uv_parser.model import DenseUVParserNet, count_parameters  # noqa: E402
from SkingToolkit.dense_uv_parser.semantic import (  # noqa: E402
    attach_siglip_runtime,
    build_siglip_runtime,
    cached_semantic_batch,
)
from SkingToolkit.dense_uv_parser.utils import (  # noqa: E402
    FACE_PALETTE,
    IGNORE_INDEX,
    LAYER_FACE_PALETTE,
    LAYER_PALETTE,
    PART_PALETTE,
    ROUTE_ROLE_PALETTE,
    SPLAT_COLOR_AGGREGATIONS,
    UV_SIZE,
    augment_dense_batch,
    build_dense_parser_batch,
    build_geometry_grid_debug,
    canonicalize_dense_targets,
    colorize_foreground,
    colorize_labels,
    colorize_surface,
    colorize_uv,
    combine_layer_face,
    flat_uv_to_uv01,
    fill_geometry_grid_debug,
    overlay_geometry_grid_debug,
    parse_views,
    prediction_uv01,
    randomize_render_background,
    render_direct_uv,
    soft_splat_geometry_predictions_to_uv,
    splat_deterministic_targets_to_uv_conditioning,
    splat_parser_predictions_to_uv_conditioning,
    splat_predictions_to_uv_conditioning,
    splat_targets_to_uv_conditioning,
    surface_class_count,
)
from SkingToolkit.semantic_uv_reconstruction.dataset import UVInpaintingDataset  # noqa: E402
from SkingToolkit.semantic_uv_reconstruction.semantic_dataset import (  # noqa: E402
    SigLIPGlobalCache,
)
from SkingToolkit.semantic_uv_reconstruction.semantic_losses import (  # noqa: E402
    build_part_layer_masks,
    build_semantic_attribute_targets,
)
from SkingToolkit.semantic_uv_reconstruction.train import get_device  # noqa: E402
from SkingToolkit.renderer import DifferentiableRenderer  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def autocast_context(device, precision):
    if precision == "no" or device.type == "cpu":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def build_grad_scaler(device, precision):
    if precision != "fp16" or device.type != "cuda":
        return None
    return torch.amp.GradScaler("cuda")


def configure_torch(args, device):
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(args.matmul_precision)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = args.cudnn_benchmark


def learning_rate_for_epoch(base_lr, epoch, epochs, schedule="cosine", min_lr_ratio=0.05):
    """Return an absolute-epoch LR so old checkpoints can resume without scheduler state."""
    if base_lr <= 0:
        raise ValueError("base_lr must be positive.")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1].")
    if schedule == "constant" or epochs <= 1:
        return float(base_lr)
    if schedule != "cosine":
        raise ValueError(f"Unsupported learning-rate schedule {schedule!r}.")
    progress = min(max((epoch - 1) / max(epochs - 1, 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(base_lr) * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def format_metrics(
    metric_sums,
    count,
    inner_recall_weight=0.5,
    outer_precision_weight=0.75,
    outer_recall_weight=0.75,
    outer_iou_weight=0.5,
    hard_rgb_selection_weight=1.0,
):
    result = {}
    for name, value in metric_sums.items():
        denominator = 1 if name.startswith("count_") else max(count, 1)
        averaged = value / denominator
        result[name] = (
            float(averaged.detach().cpu())
            if torch.is_tensor(averaged)
            else averaged
        )

    for family in ("", "hard_"):
        for role_name in ("inner", "outer"):
            count_names = tuple(
                f"count_{family}{role_name}_{suffix}"
                for suffix in ("tp", "fp", "fn")
            )
            if all(name in result for name in count_names):
                role_tp = result[f"count_{family}{role_name}_tp"]
                role_fp = result[f"count_{family}{role_name}_fp"]
                role_fn = result[f"count_{family}{role_name}_fn"]
                result[f"{family}precision_{role_name}"] = role_tp / max(
                    role_tp + role_fp, 1.0
                )
                result[f"{family}recall_{role_name}"] = role_tp / max(
                    role_tp + role_fn, 1.0
                )
                result[f"{family}iou_{role_name}"] = role_tp / max(
                    role_tp + role_fp + role_fn, 1.0
                )
                rgb_abs_name = f"count_{family}{role_name}_rgb_abs"
                rgb_values_name = f"count_{family}{role_name}_rgb_values"
                if rgb_abs_name in result and rgb_values_name in result:
                    result[f"{family}rgb_mae_{role_name}"] = (
                        result[rgb_abs_name]
                        / max(result[rgb_values_name], 1.0)
                    )

    if "recall_outer" in result:
        if "loss_geometry" in result:
            result["loss_outer_selection"] = (
                result["loss_geometry"]
                + inner_recall_weight * (1.0 - result.get("recall_inner", 1.0))
                + outer_precision_weight * (1.0 - result["precision_outer"])
                + outer_recall_weight * (1.0 - result["recall_outer"])
                + outer_iou_weight * (1.0 - result["iou_outer"])
            )
    if "hard_recall_outer" in result:
        result["loss_hard_uv_selection"] = (
            inner_recall_weight * (1.0 - result["hard_iou_inner"])
            + outer_precision_weight * (1.0 - result["hard_precision_outer"])
            + outer_recall_weight * (1.0 - result["hard_recall_outer"])
            + outer_iou_weight * (1.0 - result["hard_iou_outer"])
        )
        if "hard_rgb_mae_inner" in result and "hard_rgb_mae_outer" in result:
            result["loss_hard_uv_color_selection"] = (
                result["loss_hard_uv_selection"]
                + hard_rgb_selection_weight
                * 0.5
                * (
                    result["hard_rgb_mae_inner"]
                    + result["hard_rgb_mae_outer"]
                )
            )
    return result


def move_batch(batch, device):
    return {
        "uv": batch["uv"].to(device, non_blocking=True),
        "path": batch["path"],
    }


def stack_view_targets(targets_by_view):
    result = {}
    for key in targets_by_view[0]:
        stacked = torch.stack([targets[key] for targets in targets_by_view], dim=1)
        if stacked.dim() == 5:
            B, V, C, H, W = stacked.shape
            result[key] = stacked.reshape(B * V, C, H, W)
        elif stacked.dim() == 4:
            B, V, H, W = stacked.shape
            result[key] = stacked.reshape(B * V, H, W)
        elif stacked.dim() == 3:
            B, V, C = stacked.shape
            result[key] = stacked.reshape(B * V, C)
        else:
            raise ValueError(f"Unexpected target shape for {key}: {tuple(stacked.shape)}")
    return result


def build_parser_inputs(batch_uv, renderer, views, train, args, apply_augment=None, augment_generator=None):
    rendered_by_view = []
    targets_by_view = []
    with torch.no_grad():
        for view in views:
            rendered, targets = build_dense_parser_batch(
                batch_uv,
                renderer,
                view,
                alpha_threshold=args.target_alpha_threshold,
            )
            should_augment = train and args.augment if apply_augment is None else apply_augment
            rendered, targets = augment_dense_batch(
                rendered,
                targets,
                translation_scale=args.translation_scale if should_augment else 0.0,
                scale_range=args.scale_range if should_augment else 0.0,
                bg_color=args.bg_color,
                generator=augment_generator,
            )
            background_probability = args.background_augment_prob if train and args.background_augment else 0.0
            rendered = randomize_render_background(
                rendered,
                probability=background_probability,
                bg_color=args.bg_color,
            )
            rendered_by_view.append(rendered)
            targets_by_view.append(targets)

    rendered = torch.stack(rendered_by_view, dim=1)
    B, V, C, H, W = rendered.shape
    rendered = rendered.reshape(B * V, C, H, W)
    view_ids = torch.arange(V, device=rendered.device).view(1, V).expand(B, -1).reshape(B * V)
    targets = stack_view_targets(targets_by_view)
    return rendered, targets, V, view_ids


def visible_target_layer_uv_counts(targets, layer_index, group_size):
    """Count visible GT projections of one skin layer in each grouped UV texel."""
    layer = targets["layer"]
    uv = targets["uv"]
    if layer.dim() != 3 or uv.dim() != 4 or uv.shape[1] != 2:
        raise ValueError("Expected layer NHW and UV N2HW targets.")
    if group_size <= 0:
        raise ValueError("group_size must be positive.")
    if layer.shape[0] % group_size != 0:
        raise ValueError(
            f"Target batch {layer.shape[0]} must be divisible by group_size={group_size}."
        )

    group_count = layer.shape[0] // group_size
    x = (uv[:, 0].clamp(0.0, 1.0) * (UV_SIZE - 1)).round().long()
    y = (uv[:, 1].clamp(0.0, 1.0) * (UV_SIZE - 1)).round().long()
    flat_uv = y * UV_SIZE + x
    item_group = (
        torch.arange(layer.shape[0], device=layer.device)
        .div(group_size, rounding_mode="floor")
        .view(-1, 1, 1)
    )
    grouped_uv = flat_uv + item_group * (UV_SIZE * UV_SIZE)
    visible = layer == layer_index
    counts = uv.new_zeros(group_count * UV_SIZE * UV_SIZE)
    selected_uv = grouped_uv[visible]
    counts.scatter_add_(
        0,
        selected_uv,
        torch.ones(selected_uv.shape[0], device=uv.device, dtype=uv.dtype),
    )
    return counts.reshape(group_count, 1, UV_SIZE, UV_SIZE)


def visible_target_layer_uv_mask(targets, layer_index, group_size):
    return (
        visible_target_layer_uv_counts(targets, layer_index, group_size) > 0
    ).to(dtype=targets["uv"].dtype)


def focused_visible_layer_recall_loss(
    predicted_weight,
    visible_counts,
    hard_fraction=0.10,
    hard_weight=0.50,
):
    """Combine mean recall with the worst visible texels for small-detail protection."""
    if not 0.0 < hard_fraction <= 1.0:
        raise ValueError("hard_fraction must be in (0, 1].")
    if not 0.0 <= hard_weight <= 1.0:
        raise ValueError("hard_weight must be in [0, 1].")
    if predicted_weight.shape != visible_counts.shape:
        raise ValueError(
            "predicted_weight and visible_counts must have identical shapes, got "
            f"{tuple(predicted_weight.shape)} and {tuple(visible_counts.shape)}."
        )

    visible = visible_counts > 0
    visible_probability = (
        predicted_weight / visible_counts.clamp_min(1.0)
    ).clamp(0.0, 1.0)
    deficit = 1.0 - visible_probability
    mean_losses = []
    hard_losses = []
    for item in range(deficit.shape[0]):
        item_deficit = deficit[item][visible[item]]
        if item_deficit.numel() == 0:
            zero = predicted_weight[item].sum() * 0.0
            mean_losses.append(zero)
            hard_losses.append(zero)
            continue
        mean_losses.append(item_deficit.mean())
        hard_count = max(1, math.ceil(item_deficit.numel() * hard_fraction))
        hard_losses.append(item_deficit.topk(hard_count, sorted=False).values.mean())

    mean_loss = torch.stack(mean_losses).mean()
    hard_loss = torch.stack(hard_losses).mean()
    combined = (1.0 - hard_weight) * mean_loss + hard_weight * hard_loss
    return combined, mean_loss, hard_loss


def differentiable_geometry_losses(
    rendered,
    gt_uv,
    outputs,
    targets,
    renderer,
    views,
    temperature=1.0,
    canonicalize=True,
    recall_hard_fraction=0.10,
    recall_hard_weight=0.50,
):
    """Photometric UV and multi-view render losses for geometry-parser logits."""
    views = parse_views(views)
    pred_uv, soft_details = soft_splat_geometry_predictions_to_uv(
        rendered,
        outputs,
        renderer=renderer,
        views=views,
        group_size=len(views),
        temperature=temperature,
        canonicalize=canonicalize,
        return_details=True,
    )
    gt_uv = gt_uv.float()
    support = (pred_uv[:, 3:4].detach() > 0.05).to(dtype=pred_uv.dtype)
    gt_alpha = (gt_uv[:, 3:4] > 0.5).to(dtype=pred_uv.dtype)
    rgb_mask = support * gt_alpha
    rgb_denom = (rgb_mask.sum(dim=(1, 2, 3)) * 3.0).clamp_min(1.0)
    loss_soft_uv_rgb = (
        ((pred_uv[:, :3] - gt_uv[:, :3]).abs() * rgb_mask)
        .sum(dim=(1, 2, 3))
        .div(rgb_denom)
        .mean()
    )
    transparent_support = support * (1.0 - gt_alpha)
    alpha_denom = transparent_support.sum(dim=(1, 2, 3)).clamp_min(1.0)
    loss_soft_uv_alpha = (
        (pred_uv[:, 3:4] * transparent_support)
        .sum(dim=(1, 2, 3))
        .div(alpha_denom)
        .mean()
    )
    visible_inner_counts = visible_target_layer_uv_counts(
        targets,
        layer_index=0,
        group_size=len(views),
    ).to(dtype=pred_uv.dtype)
    visible_inner_counts = visible_inner_counts * gt_alpha
    visible_inner_uv = (visible_inner_counts > 0).to(dtype=pred_uv.dtype)
    (
        loss_soft_uv_inner_recall,
        loss_soft_uv_inner_recall_mean,
        loss_soft_uv_inner_recall_hard,
    ) = focused_visible_layer_recall_loss(
        soft_details["layer_weight"][:, 0:1],
        visible_inner_counts,
        hard_fraction=recall_hard_fraction,
        hard_weight=recall_hard_weight,
    )

    visible_outer_counts = visible_target_layer_uv_counts(
        targets,
        layer_index=1,
        group_size=len(views),
    ).to(dtype=pred_uv.dtype)
    visible_outer_counts = visible_outer_counts * gt_alpha
    visible_outer_uv = (visible_outer_counts > 0).to(dtype=pred_uv.dtype)
    (
        loss_soft_uv_outer_recall,
        loss_soft_uv_outer_recall_mean,
        loss_soft_uv_outer_recall_hard,
    ) = focused_visible_layer_recall_loss(
        soft_details["layer_weight"][:, 1:2],
        visible_outer_counts,
        hard_fraction=recall_hard_fraction,
        hard_weight=recall_hard_weight,
    )

    render_rgb_total = pred_uv.new_zeros(())
    render_alpha_total = pred_uv.new_zeros(())
    for view in views:
        pred_render = render_direct_uv(pred_uv, renderer, view)
        with torch.no_grad():
            gt_render = render_direct_uv(gt_uv, renderer, view)
        foreground = gt_render[:, 3:4].detach()
        rgb_denom = (foreground.sum(dim=(1, 2, 3)) * 3.0).clamp_min(1.0)
        render_rgb_total = render_rgb_total + (
            ((pred_render[:, :3] - gt_render[:, :3]).abs() * foreground)
            .sum(dim=(1, 2, 3))
            .div(rgb_denom)
            .mean()
        )
        alpha_support = torch.maximum(
            pred_render[:, 3:4], gt_render[:, 3:4]
        ).detach()
        alpha_denom = alpha_support.sum(dim=(1, 2, 3)).clamp_min(1.0)
        render_alpha_total = render_alpha_total + (
            ((pred_render[:, 3:4] - gt_render[:, 3:4]).abs() * alpha_support)
            .sum(dim=(1, 2, 3))
            .div(alpha_denom)
            .mean()
        )

    view_count = max(len(views), 1)
    return {
        "loss_soft_uv_rgb": loss_soft_uv_rgb,
        "loss_soft_uv_alpha": loss_soft_uv_alpha,
        "loss_soft_uv_inner_recall": loss_soft_uv_inner_recall,
        "loss_soft_uv_inner_recall_mean": loss_soft_uv_inner_recall_mean,
        "loss_soft_uv_inner_recall_hard": loss_soft_uv_inner_recall_hard,
        "loss_soft_uv_outer_recall": loss_soft_uv_outer_recall,
        "loss_soft_uv_outer_recall_mean": loss_soft_uv_outer_recall_mean,
        "loss_soft_uv_outer_recall_hard": loss_soft_uv_outer_recall_hard,
        "loss_render_rgb": render_rgb_total / view_count,
        "loss_render_alpha": render_alpha_total / view_count,
        "visible_inner_uv_percent": visible_inner_uv.float().mean() * 100.0,
        "visible_outer_uv_percent": visible_outer_uv.float().mean() * 100.0,
        "soft_uv_known_percent": (
            (soft_details["support"] > 0.05).float().mean() * 100.0
        ),
    }


def hard_uv_conditioning_metrics(
    rendered,
    outputs,
    targets,
    renderer,
    views,
    args,
):
    """Measure known-texel precision/recall through the exact hard inference route."""
    group_size = len(views)
    hard_outputs = {
        name: (
            value.to(dtype=rendered.dtype)
            if torch.is_tensor(value) and value.is_floating_point()
            else value
        )
        for name, value in outputs.items()
    }
    if "affine" in outputs:
        predicted = splat_parser_predictions_to_uv_conditioning(
            rendered,
            hard_outputs,
            renderer=renderer,
            views=views,
            group_size=group_size,
            fg_threshold=args.splat_fg_threshold,
            bg_color=args.bg_color,
            semantic_gate=args.semantic_gate,
            affine_refine=args.affine_refine,
            affine_refine_translation_px=args.affine_refine_translation_px,
            affine_refine_scale=args.affine_refine_scale,
            route_confidence_threshold=args.route_confidence_threshold,
            route_margin_threshold=args.route_margin_threshold,
            outer_route_confidence_threshold=args.outer_route_confidence_threshold,
            outer_route_margin_threshold=args.outer_route_margin_threshold,
            outer_uv_min_coverage=args.outer_uv_min_coverage,
            outer_uv_min_source_pixels=args.outer_uv_min_source_pixels,
            outer_geometry_rescue=getattr(args, "outer_geometry_rescue", False),
            outer_rescue_confidence_threshold=getattr(
                args, "outer_rescue_confidence_threshold", 0.60
            ),
            outer_rescue_margin_threshold=getattr(
                args, "outer_rescue_margin_threshold", 0.25
            ),
            outer_rescue_min_coverage=getattr(
                args, "outer_rescue_min_coverage", 0.10
            ),
            color_aggregation=args.splat_color_aggregation,
            geometry_route_texel_consensus=getattr(
                args, "geometry_route_texel_consensus", False
            ),
            observed_foreground=None,
            background_color_tolerance=args.background_color_tolerance,
            color_background_tolerance=getattr(
                args, "color_background_tolerance", 8.0 / 255.0
            ),
            color_foreground_inset=getattr(args, "color_foreground_inset", 1),
            reject_semantic_fallback=not args.allow_semantic_fallback,
        )
        expected = splat_deterministic_targets_to_uv_conditioning(
            rendered,
            targets,
            renderer=renderer,
            views=views,
            group_size=group_size,
            bg_color=args.bg_color,
            color_aggregation=args.splat_color_aggregation,
        )
    else:
        predicted = splat_predictions_to_uv_conditioning(
            rendered,
            hard_outputs,
            group_size=group_size,
            fg_threshold=args.splat_fg_threshold,
            bg_color=args.bg_color,
        )
        expected = splat_targets_to_uv_conditioning(
            rendered,
            targets,
            group_size=group_size,
            bg_color=args.bg_color,
        )

    metrics = {}
    for role_name, rgba_start, known_channel in (
        ("inner", 0, 4),
        ("outer", 5, 9),
    ):
        predicted_known = predicted[:, known_channel] > 0.5
        expected_known = expected[:, known_channel] > 0.5
        matched = predicted_known & expected_known
        metrics[f"count_hard_{role_name}_tp"] = matched.sum().float()
        metrics[f"count_hard_{role_name}_fp"] = (
            predicted_known & ~expected_known
        ).sum().float()
        metrics[f"count_hard_{role_name}_fn"] = (
            ~predicted_known & expected_known
        ).sum().float()
        rgb_error = (
            predicted[:, rgba_start : rgba_start + 3]
            - expected[:, rgba_start : rgba_start + 3]
        ).abs()
        metrics[f"count_hard_{role_name}_rgb_abs"] = (
            rgb_error * matched.unsqueeze(1)
        ).sum().float()
        metrics[f"count_hard_{role_name}_rgb_values"] = (
            matched.sum().float() * 3.0
        )
    return metrics


def run_epoch(
    model,
    criterion,
    renderer,
    loader,
    optimizer,
    scaler,
    device,
    precision,
    args,
    train=True,
    compute_hard_metrics=False,
    semantic_cache=None,
    semantic_masks=None,
):
    model.train(train)
    views = parse_views(args.views)
    metric_sums = {}
    sample_count = 0
    iterator = tqdm(loader, leave=False, file=sys.__stderr__ or sys.stderr) if tqdm is not None else loader
    val_generator = None
    apply_augment = train and args.augment
    if not train and args.augment_validation:
        val_generator = torch.Generator(device=device)
        val_generator.manual_seed(args.seed + 1009)
        apply_augment = True

    for batch in iterator:
        batch = move_batch(batch, device)
        rendered, targets, _, view_ids = build_parser_inputs(
            batch["uv"],
            renderer,
            views,
            train=train,
            args=args,
            apply_augment=apply_augment,
            augment_generator=val_generator,
        )
        parser_samples = rendered.shape[0]
        semantic_features = cached_semantic_batch(
            semantic_cache, batch["path"], device
        )

        with torch.set_grad_enabled(train):
            with autocast_context(device, precision):
                outputs = (
                    model(
                        rendered,
                        view_ids=view_ids,
                        semantic_features=semantic_features,
                    )
                    if semantic_features is not None
                    else model(rendered, view_ids=view_ids)
                )
                losses = criterion(outputs, targets)
                zero = losses["loss_total"].new_zeros(())
                if semantic_masks is not None and "outer_presence_logits" in outputs:
                    inner_part_masks, outer_part_masks = semantic_masks
                    attributes = build_semantic_attribute_targets(
                        batch["uv"], inner_part_masks, outer_part_masks
                    )
                    loss_semantic_presence = F.binary_cross_entropy_with_logits(
                        outputs["outer_presence_logits"].float(),
                        attributes["outer_presence"],
                    )
                    loss_semantic_coverage = F.smooth_l1_loss(
                        outputs["outer_coverage"].float(),
                        attributes["outer_coverage"],
                    )
                    loss_semantic_attributes = (
                        args.lambda_semantic_presence * loss_semantic_presence
                        + args.lambda_semantic_coverage * loss_semantic_coverage
                    )
                    losses["loss_semantic_presence"] = loss_semantic_presence
                    losses["loss_semantic_coverage"] = loss_semantic_coverage
                    losses["loss_semantic_attributes"] = loss_semantic_attributes
                    losses["loss_total"] = losses["loss_total"] + loss_semantic_attributes
                    losses["loss_routing"] = losses["loss_routing"] + loss_semantic_attributes
                else:
                    losses["loss_semantic_presence"] = zero
                    losses["loss_semantic_coverage"] = zero
                    losses["loss_semantic_attributes"] = zero
                auxiliary_enabled = args.parser_mode == "geometry_fit" and any(
                    weight > 0
                    for weight in (
                        args.lambda_soft_uv_rgb,
                        args.lambda_soft_uv_alpha,
                        args.lambda_soft_uv_inner_recall,
                        args.lambda_soft_uv_outer_recall,
                        args.lambda_render_rgb,
                        args.lambda_render_alpha,
                    )
                )
                if auxiliary_enabled:
                    auxiliary = differentiable_geometry_losses(
                        rendered,
                        batch["uv"],
                        outputs,
                        targets,
                        renderer,
                        views,
                        temperature=args.render_softmax_temperature,
                        canonicalize=apply_augment,
                        recall_hard_fraction=args.soft_uv_recall_hard_fraction,
                        recall_hard_weight=args.soft_uv_recall_hard_weight,
                    )
                else:
                    auxiliary = {
                        "loss_soft_uv_rgb": zero,
                        "loss_soft_uv_alpha": zero,
                        "loss_soft_uv_inner_recall": zero,
                        "loss_soft_uv_inner_recall_mean": zero,
                        "loss_soft_uv_inner_recall_hard": zero,
                        "loss_soft_uv_outer_recall": zero,
                        "loss_soft_uv_outer_recall_mean": zero,
                        "loss_soft_uv_outer_recall_hard": zero,
                        "loss_render_rgb": zero,
                        "loss_render_alpha": zero,
                        "soft_uv_known_percent": zero,
                        "visible_inner_uv_percent": zero,
                        "visible_outer_uv_percent": zero,
                    }
                weighted_auxiliary = (
                    args.lambda_soft_uv_rgb * auxiliary["loss_soft_uv_rgb"]
                    + args.lambda_soft_uv_alpha * auxiliary["loss_soft_uv_alpha"]
                    + args.lambda_soft_uv_inner_recall
                    * auxiliary["loss_soft_uv_inner_recall"]
                    + args.lambda_soft_uv_outer_recall
                    * auxiliary["loss_soft_uv_outer_recall"]
                    + args.lambda_render_rgb * auxiliary["loss_render_rgb"]
                    + args.lambda_render_alpha * auxiliary["loss_render_alpha"]
                )
                losses.update(auxiliary)
                losses["loss_differentiable"] = weighted_auxiliary
                losses["loss_total"] = losses["loss_total"] + weighted_auxiliary
                losses["loss_geometry"] = losses["loss_geometry"] + weighted_auxiliary
                losses["loss_routing"] = losses["loss_routing"] + weighted_auxiliary
                loss = losses["loss_total"]

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

        if compute_hard_metrics:
            with torch.no_grad():
                losses.update(
                    hard_uv_conditioning_metrics(
                        rendered,
                        outputs,
                        targets,
                        renderer,
                        views,
                        args,
                    )
                )

        sample_count += parser_samples
        for name, value in losses.items():
            detached = value.detach()
            metric_weight = 1 if name.startswith("count_") else parser_samples
            metric_sums[name] = (
                metric_sums.get(name, detached.new_zeros(()))
                + detached * metric_weight
            )

        if tqdm is not None and args.log_every > 0 and sample_count % (args.log_every * parser_samples) == 0:
            avg = format_metrics(
                metric_sums,
                sample_count,
                inner_recall_weight=args.inner_selection_recall_weight,
                outer_precision_weight=args.outer_selection_precision_weight,
                outer_recall_weight=args.outer_selection_recall_weight,
                outer_iou_weight=args.outer_selection_iou_weight,
                hard_rgb_selection_weight=args.hard_rgb_selection_weight,
            )
            postfix = {
                "total": f"{avg['loss_total']:.4f}",
                "fg": f"{avg.get('recall_foreground', avg['acc_foreground']):.3f}",
            }
            if args.parser_mode in ("global_affine", "geometry_fit"):
                postfix["align"] = f"{avg.get('err_affine_translation_px', 0.0):.2f}px"
                postfix["scale"] = f"{avg.get('err_affine_scale_pct', 0.0):.2f}%"
                if args.parser_mode == "geometry_fit":
                    postfix["outer_p"] = f"{avg.get('precision_outer', 0.0):.3f}"
                    postfix["outer_r"] = f"{avg.get('recall_outer', 0.0):.3f}"
                    postfix["render"] = f"{avg.get('loss_render_rgb', 0.0):.3f}"
                else:
                    postfix["surface"] = f"{avg.get('acc_surface', 0.0):.3f}"
            else:
                postfix["uv"] = f"{avg.get('loss_uv_l1_px', avg['loss_uv']):.2f}px"
                postfix["uv1"] = f"{avg.get('acc_uv_within1', 0.0):.3f}"
            iterator.set_postfix(
                **postfix,
            )

    return format_metrics(
        metric_sums,
        sample_count,
        inner_recall_weight=args.inner_selection_recall_weight,
        outer_precision_weight=args.outer_selection_precision_weight,
        outer_recall_weight=args.outer_selection_recall_weight,
        outer_iou_weight=args.outer_selection_iou_weight,
        hard_rgb_selection_weight=args.hard_rgb_selection_weight,
    )





def save_preview(
    model,
    renderer,
    loader,
    device,
    args,
    output_path,
    max_items=2,
    semantic_cache=None,
):
    model.eval()
    views = parse_views(args.views)
    batch = move_batch(next(iter(loader)), device)
    rendered, targets, view_count, view_ids = build_parser_inputs(
        batch["uv"], renderer, views, train=False, args=args, apply_augment=False
    )
    semantic_features = cached_semantic_batch(
        semantic_cache, batch["path"], device
    )
    with torch.no_grad():
        outputs = (
            model(
                rendered,
                view_ids=view_ids,
                semantic_features=semantic_features,
            )
            if semantic_features is not None
            else model(rendered, view_ids=view_ids)
        )
        if "affine" in outputs:
            pred_conditioning, routing_details = splat_parser_predictions_to_uv_conditioning(
                rendered,
                outputs,
                renderer=renderer,
                views=views,
                group_size=view_count,
                fg_threshold=args.splat_fg_threshold,
                bg_color=args.bg_color,
                semantic_gate=args.semantic_gate,
                affine_refine=args.affine_refine,
                affine_refine_translation_px=args.affine_refine_translation_px,
                affine_refine_scale=args.affine_refine_scale,
                route_confidence_threshold=args.route_confidence_threshold,
                route_margin_threshold=args.route_margin_threshold,
                outer_route_confidence_threshold=args.outer_route_confidence_threshold,
                outer_route_margin_threshold=args.outer_route_margin_threshold,
                outer_uv_min_coverage=args.outer_uv_min_coverage,
                outer_uv_min_source_pixels=getattr(
                    args, "outer_uv_min_source_pixels", 3
                ),
                outer_geometry_rescue=getattr(args, "outer_geometry_rescue", False),
                outer_rescue_confidence_threshold=getattr(
                    args, "outer_rescue_confidence_threshold", 0.60
                ),
                outer_rescue_margin_threshold=getattr(
                    args, "outer_rescue_margin_threshold", 0.25
                ),
                outer_rescue_min_coverage=getattr(
                    args, "outer_rescue_min_coverage", 0.10
                ),
                color_aggregation=args.splat_color_aggregation,
                geometry_route_texel_consensus=getattr(
                    args, "geometry_route_texel_consensus", False
                ),
                observed_foreground=None,
                background_color_tolerance=getattr(
                    args, "background_color_tolerance", 0.25
                ),
                color_background_tolerance=getattr(
                    args, "color_background_tolerance", 8.0 / 255.0
                ),
                color_foreground_inset=getattr(
                    args, "color_foreground_inset", 1
                ),
                reject_semantic_fallback=not args.allow_semantic_fallback,
                return_details=True,
            )
            gt_conditioning = splat_deterministic_targets_to_uv_conditioning(
                rendered,
                targets,
                renderer=renderer,
                views=views,
                group_size=view_count,
                bg_color=args.bg_color,
            )
        else:
            routing_details = None
            pred_conditioning = splat_predictions_to_uv_conditioning(
                rendered,
                outputs,
                group_size=view_count,
                fg_threshold=args.splat_fg_threshold,
                bg_color=args.bg_color,
            )
            gt_conditioning = splat_targets_to_uv_conditioning(
                rendered,
                targets,
                group_size=view_count,
                bg_color=args.bg_color,
            )

    count = min(max_items, pred_conditioning.shape[0])
    pred_conditioning = pred_conditioning[:count].detach().cpu()
    gt_conditioning = gt_conditioning[:count].detach().cpu()
    preview = torch.cat(
        [
            pred_conditioning[:, 0:3],
            pred_conditioning[:, 5:8],
            gt_conditioning[:, 0:3],
            gt_conditioning[:, 5:8],
        ],
        dim=0,
    )
    save_image(preview.clamp(0.0, 1.0), output_path, nrow=count)

    debug_count = min(count * view_count, rendered.shape[0])
    if routing_details is not None:
        debug_outputs = {key: value[:debug_count] for key, value in routing_details["outputs"].items()}
        rendered_debug = routing_details["rendered"][:debug_count]
        targets_debug = {
            key: value[:debug_count]
            for key, value in canonicalize_dense_targets(targets).items()
        }
        routing_debug = {
            key: value[:debug_count]
            for key, value in routing_details["routing"].items()
        }
        pred_fg = routing_debug["foreground"]
        pred_layer_values = routing_debug["layer"]
        pred_uv = flat_uv_to_uv01(routing_debug["flat_uv"], rendered_debug.dtype)
    else:
        debug_outputs = {key: value[:debug_count] for key, value in outputs.items()}
        rendered_debug = rendered[:debug_count]
        targets_debug = {key: value[:debug_count] for key, value in targets.items()}
        pred_fg = torch.sigmoid(debug_outputs["foreground"])[:, 0] > args.splat_fg_threshold
        pred_layer_values = debug_outputs["layer"].argmax(dim=1)
        pred_uv = prediction_uv01(debug_outputs)
    gt_fg = targets_debug["foreground"][:, 0] > 0.5
    raw_fg = torch.sigmoid(debug_outputs["foreground"])[:, 0] > args.splat_fg_threshold
    pred_part_values = (
        debug_outputs["part"].argmax(dim=1)
        if "part" in debug_outputs
        else routing_debug["part"]
    )
    pred_part = torch.where(pred_fg, pred_part_values, torch.full_like(pred_part_values, IGNORE_INDEX))
    pred_layer = torch.where(
        pred_fg,
        pred_layer_values,
        torch.full_like(debug_outputs["layer"].argmax(dim=1), IGNORE_INDEX),
    )
    pred_face_values = (
        debug_outputs["face"].argmax(dim=1)
        if "face" in debug_outputs
        else routing_debug["face"]
    )
    pred_face = torch.where(
        pred_fg,
        pred_face_values,
        torch.full_like(pred_face_values, IGNORE_INDEX),
    )
    gt_face = torch.where(
        gt_fg,
        targets_debug["face"],
        torch.full_like(targets_debug["face"], IGNORE_INDEX),
    )
    debug_images = [
        rendered_debug[:, :3],
        colorize_foreground(pred_fg, args.bg_color, rendered_debug),
        colorize_foreground(gt_fg, args.bg_color, rendered_debug),
        colorize_labels(pred_part, PART_PALETTE, args.bg_color, rendered_debug),
        colorize_labels(targets_debug["part"], PART_PALETTE, args.bg_color, rendered_debug),
        colorize_labels(pred_layer, LAYER_PALETTE, args.bg_color, rendered_debug),
        colorize_labels(targets_debug["layer"], LAYER_PALETTE, args.bg_color, rendered_debug),
        colorize_labels(pred_face, FACE_PALETTE, args.bg_color, rendered_debug),
        colorize_labels(gt_face, FACE_PALETTE, args.bg_color, rendered_debug),
        colorize_labels(
            (
                torch.where(
                    pred_fg,
                    debug_outputs["layer_face"].argmax(dim=1),
                    torch.full_like(debug_outputs["layer_face"].argmax(dim=1), IGNORE_INDEX),
                )
                if "layer_face" in debug_outputs
                else combine_layer_face(pred_layer, pred_face)
            ),
            LAYER_FACE_PALETTE,
            args.bg_color,
            rendered_debug,
        ),
        colorize_labels(
            combine_layer_face(targets_debug["layer"], gt_face),
            LAYER_FACE_PALETTE,
            args.bg_color,
            rendered_debug,
        ),
    ]
    if "route_role" in targets_debug and debug_outputs["layer"].shape[1] == 3:
        pred_route_role = torch.where(
            raw_fg,
            debug_outputs["layer"].argmax(dim=1),
            torch.full_like(debug_outputs["layer"].argmax(dim=1), IGNORE_INDEX),
        )
        debug_images.extend(
            [
                colorize_labels(pred_route_role, ROUTE_ROLE_PALETTE, args.bg_color, rendered_debug),
                colorize_labels(targets_debug["route_role"], ROUTE_ROLE_PALETTE, args.bg_color, rendered_debug),
            ]
        )
        if "route_role_prior" in debug_outputs:
            prior_route_role = torch.where(
                gt_fg,
                debug_outputs["route_role_prior"].argmax(dim=1),
                torch.full_like(targets_debug["route_role"], IGNORE_INDEX),
            )
            debug_images.append(
                colorize_labels(
                    prior_route_role,
                    ROUTE_ROLE_PALETTE,
                    args.bg_color,
                    rendered_debug,
                )
            )
    if routing_details is not None:
        geometry_debug = build_geometry_grid_debug(
            renderer,
            views,
            rendered_debug.shape[0],
            rendered_debug,
            bg_color=args.bg_color,
        )
        inner_fill, outer_fill = fill_geometry_grid_debug(
            rendered_debug,
            pred_fg,
            pred_layer_values,
            geometry_debug,
            bg_color=args.bg_color,
        )
        inner_overlay, outer_overlay = overlay_geometry_grid_debug(
            rendered_debug,
            geometry_debug,
        )
        inner_routed_overlay, outer_routed_overlay = overlay_geometry_grid_debug(
            rendered_debug,
            geometry_debug,
            base_images=(inner_fill, outer_fill),
        )
        debug_images.extend(
            [
                inner_overlay,
                outer_overlay,
                inner_routed_overlay,
                outer_routed_overlay,
                geometry_debug[0],
                geometry_debug[1],
                inner_fill,
                outer_fill,
            ]
        )
    if routing_details is not None:
        pred_surface = torch.where(
            pred_fg,
            routing_debug["surface"],
            torch.full_like(routing_debug["surface"], IGNORE_INDEX),
        )
        debug_images.extend(
            [
                colorize_surface(pred_surface, args.bg_color, rendered_debug),
                colorize_surface(targets_debug["surface"], args.bg_color, rendered_debug),
            ]
        )
    debug_images.extend(
        [
            colorize_uv(pred_uv, pred_fg, args.bg_color),
            colorize_uv(targets_debug["uv"], gt_fg, args.bg_color),
        ]
    )
    debug_preview = torch.cat(debug_images, dim=0)
    debug_path = output_path.with_name(f"{output_path.stem}_debug{output_path.suffix}")
    save_image(debug_preview.clamp(0.0, 1.0).detach().cpu(), debug_path, nrow=view_count)


def save_checkpoint(path, model, optimizer, scaler, epoch, args, metrics, best_metric=None):
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "args": vars(args),
        "metrics": metrics,
        "best_metric": best_metric,
        "model_config": {
            "input_channels": 4,
            "base_channels": args.base_channels,
            "uv_size": UV_SIZE,
            "uv_classification": model.uv_classification,
            "view_classes": len(parse_views(args.views)),
            "parser_mode": args.parser_mode,
            "predict_affine": model.predict_affine,
            "affine_translation_scale": model.affine_translation_scale,
            "affine_scale_range": model.affine_scale_range,
            "surface_classes": model.surface_classes,
            "layer_classes": model.layer_classes,
            "route_role_classes": model.layer_classes if model.geometry_only else 0,
            "layer_face_classes": 0 if model.geometry_only else 12,
            "geometry_only": model.geometry_only,
            "feature_dropout": model.feature_dropout_probability,
            "semantic_backbone": args.semantic_backbone,
            "siglip_model": args.siglip_model if args.semantic_backbone == "siglip2" else None,
            "semantic_feature_dim": model.semantic_feature_dim,
            "semantic_channels": model.semantic_channels,
            "semantic_attention_heads": model.semantic_attention_heads,
            "semantic_layers": model.semantic_layers,
            "semantic_dropout": model.semantic_dropout,
            "predict_confidence": model.predict_confidence,
            "route_role_spatial_prior": model.route_role_spatial_prior,
            "route_prior_height": model.route_prior_height,
            "route_prior_width": model.route_prior_width,
            "route_prior_logit_cap": model.route_prior_logit_cap,
            "route_prior_dropout": model.route_prior_dropout,
            "arm_model": "steve",
        },
    }
    path = Path(path)
    temporary_path = path.with_name(f".{path.name}.tmp")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train a dense render-pixel to Minecraft UV parser.")
    parser.add_argument("--data_dir", default="../skins")
    parser.add_argument("--output_dir", default="runs/dense_uv_parser")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume; --epochs remains the final epoch number.")
    parser.add_argument("--mappings_dir", default=None)
    parser.add_argument("--views", default="walk_front_both_layer_ortho,walk_back_both_layer_ortho")
    parser.add_argument(
        "--parser_mode",
        choices=["geometry_fit", "global_affine", "dense"],
        default="geometry_fit",
        help="geometry_fit learns alignment, route role, and exact renderer surface-slot routing.",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--feature_dropout", type=float, default=0.10)
    parser.add_argument(
        "--route_role_spatial_prior",
        dest="route_role_spatial_prior",
        action="store_true",
        default=True,
        help=(
            "Learn a bounded per-view canonical route-role prior for common "
            "fixed-view patterns such as bangs, hat brims, and back hair."
        ),
    )
    parser.add_argument(
        "--no_route_role_spatial_prior",
        dest="route_role_spatial_prior",
        action="store_false",
    )
    parser.add_argument("--route_prior_height", type=int, default=32)
    parser.add_argument("--route_prior_width", type=int, default=16)
    parser.add_argument("--route_prior_logit_cap", type=float, default=1.5)
    parser.add_argument("--route_prior_dropout", type=float, default=0.10)
    parser.add_argument(
        "--semantic_backbone",
        choices=["none", "siglip2"],
        default="none",
        help="Frozen global semantic context used to condition route-role parsing.",
    )
    parser.add_argument("--siglip_model", default="google/siglip2-base-patch16-224")
    parser.add_argument("--siglip_cache_dir", default=None)
    parser.add_argument("--siglip_local_files_only", action="store_true")
    parser.add_argument("--semantic_channels", type=int, default=128)
    parser.add_argument("--semantic_attention_heads", type=int, default=4)
    parser.add_argument("--semantic_layers", type=int, default=1)
    parser.add_argument("--semantic_dropout", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--lr_schedule",
        choices=["constant", "cosine"],
        default="cosine",
        help="Absolute-epoch schedule; cosine is resume-safe and reduces late-epoch drift.",
    )
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--matmul_precision", choices=["highest", "high", "medium"], default="high")
    parser.add_argument("--cudnn_benchmark", dest="cudnn_benchmark", action="store_true", default=True)
    parser.add_argument("--no_cudnn_benchmark", dest="cudnn_benchmark", action="store_false")
    parser.add_argument("--target_alpha_threshold", type=float, default=0.5)
    parser.add_argument("--splat_fg_threshold", type=float, default=0.5)
    parser.add_argument("--affine_refine", dest="affine_refine", action="store_true", default=False)
    parser.add_argument("--no_affine_refine", dest="affine_refine", action="store_false")
    parser.add_argument("--affine_refine_translation_px", type=float, default=0.0)
    parser.add_argument("--affine_refine_scale", type=float, default=0.0)
    parser.add_argument("--route_confidence_threshold", type=float, default=0.0)
    parser.add_argument("--route_margin_threshold", type=float, default=0.0)
    parser.add_argument(
        "--background_color_tolerance", type=float, default=0.25
    )
    parser.add_argument(
        "--color_background_tolerance", type=float, default=8.0 / 255.0
    )
    parser.add_argument("--color_foreground_inset", type=int, default=1)
    parser.add_argument("--outer_route_confidence_threshold", type=float, default=0.80)
    parser.add_argument("--outer_route_margin_threshold", type=float, default=0.55)
    parser.add_argument("--outer_uv_min_coverage", type=float, default=0.25)
    parser.add_argument("--outer_uv_min_source_pixels", type=int, default=3)
    parser.add_argument(
        "--outer_geometry_rescue",
        dest="outer_geometry_rescue",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_outer_geometry_rescue",
        dest="outer_geometry_rescue",
        action="store_false",
    )
    parser.add_argument("--outer_rescue_confidence_threshold", type=float, default=0.60)
    parser.add_argument("--outer_rescue_margin_threshold", type=float, default=0.25)
    parser.add_argument("--outer_rescue_min_coverage", type=float, default=0.10)
    parser.add_argument(
        "--geometry_route_texel_consensus",
        dest="geometry_route_texel_consensus",
        action="store_true",
        default=True,
        help="Let fixed projected UV cells override local semantic route-role predictions.",
    )
    parser.add_argument(
        "--no_geometry_route_texel_consensus",
        dest="geometry_route_texel_consensus",
        action="store_false",
    )
    parser.add_argument(
        "--splat_color_aggregation",
        choices=SPLAT_COLOR_AGGREGATIONS,
        default="texel_center",
    )
    parser.add_argument("--allow_semantic_fallback", action="store_true")
    parser.add_argument("--augment", dest="augment", action="store_true", default=False)
    parser.add_argument("--no_augment", dest="augment", action="store_false")
    parser.add_argument("--augment_validation", dest="augment_validation", action="store_true", default=False)
    parser.add_argument("--no_augment_validation", dest="augment_validation", action="store_false")
    parser.add_argument("--translation_scale", type=float, default=0.0)
    parser.add_argument("--scale_range", type=float, default=0.0)
    parser.add_argument("--background_augment", dest="background_augment", action="store_true", default=True)
    parser.add_argument("--no_background_augment", dest="background_augment", action="store_false")
    parser.add_argument("--background_augment_prob", type=float, default=0.9)
    parser.add_argument("--semantic_gate", dest="semantic_gate", action="store_true", default=True)
    parser.add_argument("--no_semantic_gate", dest="semantic_gate", action="store_false")
    parser.add_argument("--lambda_foreground", type=float, default=1.0)
    parser.add_argument("--lambda_layer", type=float, default=1.0)
    parser.add_argument("--lambda_part", type=float, default=0.5)
    parser.add_argument("--lambda_face", type=float, default=0.5)
    parser.add_argument("--lambda_layer_face", type=float, default=1.0)
    parser.add_argument("--lambda_uv", type=float, default=0.25)
    parser.add_argument("--lambda_uv_class", type=float, default=1.0)
    parser.add_argument("--lambda_affine", type=float, default=1.0)
    parser.add_argument("--lambda_surface", type=float, default=1.0)
    parser.add_argument("--lambda_outer_false_positive", type=float, default=0.75)
    parser.add_argument("--lambda_outer_false_negative", type=float, default=0.75)
    parser.add_argument("--lambda_route_confidence", type=float, default=0.25)
    parser.add_argument("--lambda_primary_route_swap", type=float, default=1.0)
    parser.add_argument(
        "--lambda_route_texel_consistency", type=float, default=0.25
    )
    parser.add_argument(
        "--lambda_route_prior_regularization", type=float, default=0.001
    )
    parser.add_argument("--lambda_semantic_presence", type=float, default=0.25)
    parser.add_argument("--lambda_semantic_coverage", type=float, default=0.25)
    parser.add_argument("--outer_false_positive_gamma", type=float, default=2.0)
    parser.add_argument("--outer_false_negative_gamma", type=float, default=2.0)
    parser.add_argument("--primary_route_swap_gamma", type=float, default=2.0)
    parser.add_argument("--route_prior_tv_weight", type=float, default=1.0)
    parser.add_argument("--route_class_weight_floor", type=float, default=0.75)
    parser.add_argument("--route_outer_class_weight_cap", type=float, default=1.0)
    parser.add_argument("--lambda_soft_uv_rgb", type=float, default=0.25)
    parser.add_argument("--lambda_soft_uv_alpha", type=float, default=0.35)
    parser.add_argument("--lambda_soft_uv_inner_recall", type=float, default=0.50)
    parser.add_argument("--lambda_soft_uv_outer_recall", type=float, default=0.50)
    parser.add_argument("--soft_uv_recall_hard_fraction", type=float, default=0.10)
    parser.add_argument("--soft_uv_recall_hard_weight", type=float, default=0.50)
    parser.add_argument("--lambda_render_rgb", type=float, default=0.20)
    parser.add_argument("--lambda_render_alpha", type=float, default=0.25)
    parser.add_argument("--outer_selection_precision_weight", type=float, default=1.50)
    parser.add_argument("--outer_selection_recall_weight", type=float, default=0.50)
    parser.add_argument("--outer_selection_iou_weight", type=float, default=0.5)
    parser.add_argument("--inner_selection_recall_weight", type=float, default=0.5)
    parser.add_argument("--hard_rgb_selection_weight", type=float, default=1.0)
    parser.add_argument(
        "--render_softmax_temperature",
        type=float,
        default=1.0,
        help="Temperature for the differentiable route/surface probabilities.",
    )
    parser.add_argument("--uv_classification", dest="uv_classification", action="store_true", default=True)
    parser.add_argument("--no_uv_classification", dest="uv_classification", action="store_false")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--preview_every", type=int, default=1)
    parser.add_argument(
        "--best_metric",
        choices=[
            "loss_total",
            "loss_geometry",
            "loss_routing",
            "loss_surface",
            "loss_uv_class",
            "loss_differentiable",
            "loss_outer_selection",
            "loss_hard_uv_selection",
            "loss_hard_uv_color_selection",
        ],
        default="loss_hard_uv_color_selection",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    args.bg_color = (128, 128, 128)
    if args.scale_range < 0 or args.scale_range >= 1:
        raise ValueError("--scale_range must be in [0, 1).")
    if not 0.0 <= args.feature_dropout < 1.0:
        raise ValueError("--feature_dropout must be in [0, 1).")
    if args.route_prior_height < 1 or args.route_prior_width < 1:
        raise ValueError("Route-prior dimensions must be positive.")
    if args.route_prior_logit_cap <= 0:
        raise ValueError("--route_prior_logit_cap must be positive.")
    if not 0.0 <= args.route_prior_dropout < 1.0:
        raise ValueError("--route_prior_dropout must be in [0, 1).")
    if args.route_role_spatial_prior and args.parser_mode != "geometry_fit":
        raise ValueError(
            "--route_role_spatial_prior is supported only by geometry_fit."
        )
    if (
        args.route_role_spatial_prior
        and args.augment
        and (args.translation_scale > 0.0 or args.scale_range > 0.0)
    ):
        raise ValueError(
            "The canonical route prior requires fixed geometry; disable geometric "
            "augmentation or disable the spatial prior."
        )
    differentiable_weights = (
        args.lambda_soft_uv_rgb,
        args.lambda_soft_uv_alpha,
        args.lambda_soft_uv_inner_recall,
        args.lambda_soft_uv_outer_recall,
        args.lambda_render_rgb,
        args.lambda_render_alpha,
    )
    if any(weight < 0 for weight in differentiable_weights):
        raise ValueError("Differentiable parser loss weights must be non-negative.")
    if args.hard_rgb_selection_weight < 0:
        raise ValueError("--hard_rgb_selection_weight must be non-negative.")
    if not 0.0 <= args.background_color_tolerance <= 1.0:
        raise ValueError("--background_color_tolerance must be in [0, 1].")
    if not 0.0 <= args.color_background_tolerance <= 1.0:
        raise ValueError("--color_background_tolerance must be in [0, 1].")
    if args.color_foreground_inset < 0:
        raise ValueError("--color_foreground_inset must be non-negative.")
    if args.outer_uv_min_source_pixels < 1:
        raise ValueError("--outer_uv_min_source_pixels must be positive.")
    if args.lr <= 0:
        raise ValueError("--lr must be positive.")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("--min_lr_ratio must be in [0, 1].")
    if not 0.0 < args.soft_uv_recall_hard_fraction <= 1.0:
        raise ValueError("--soft_uv_recall_hard_fraction must be in (0, 1].")
    if not 0.0 <= args.soft_uv_recall_hard_weight <= 1.0:
        raise ValueError("--soft_uv_recall_hard_weight must be in [0, 1].")
    if args.lambda_outer_false_positive < 0:
        raise ValueError("--lambda_outer_false_positive must be non-negative.")
    if args.lambda_outer_false_negative < 0:
        raise ValueError("--lambda_outer_false_negative must be non-negative.")
    if min(
        args.lambda_route_confidence,
        args.lambda_primary_route_swap,
        args.lambda_route_texel_consistency,
        args.lambda_route_prior_regularization,
        args.lambda_semantic_presence,
        args.lambda_semantic_coverage,
    ) < 0:
        raise ValueError("Route, semantic, and confidence loss weights must be non-negative.")
    if args.semantic_channels < 1 or args.semantic_layers < 1:
        raise ValueError("Semantic channels and layers must be positive.")
    if args.semantic_attention_heads < 1:
        raise ValueError("--semantic_attention_heads must be positive.")
    if args.semantic_channels % args.semantic_attention_heads != 0:
        raise ValueError("Semantic channels must be divisible by attention heads.")
    if not 0.0 <= args.semantic_dropout < 1.0:
        raise ValueError("--semantic_dropout must be in [0, 1).")
    if args.outer_false_positive_gamma < 0:
        raise ValueError("--outer_false_positive_gamma must be non-negative.")
    if args.outer_false_negative_gamma < 0:
        raise ValueError("--outer_false_negative_gamma must be non-negative.")
    if args.primary_route_swap_gamma < 0:
        raise ValueError("--primary_route_swap_gamma must be non-negative.")
    if args.route_prior_tv_weight < 0:
        raise ValueError("--route_prior_tv_weight must be non-negative.")
    if not 0 <= args.route_class_weight_floor <= 4.0:
        raise ValueError("--route_class_weight_floor must be in [0, 4].")
    if args.route_outer_class_weight_cap <= 0:
        raise ValueError("--route_outer_class_weight_cap must be positive.")
    if (
        args.outer_selection_precision_weight < 0
        or args.outer_selection_recall_weight < 0
        or args.outer_selection_iou_weight < 0
        or args.inner_selection_recall_weight < 0
    ):
        raise ValueError("Outer checkpoint-selection weights must be non-negative.")
    if args.render_softmax_temperature <= 0:
        raise ValueError("--render_softmax_temperature must be positive.")
    device = get_device(args.device)
    configure_torch(args, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "previews").mkdir(exist_ok=True)

    dataset = UVInpaintingDataset(
        data_dir=args.data_dir,
        mappings_dir=args.mappings_dir,
        views=args.views,
        max_samples=args.max_samples,
    )
    semantic_cache = None
    runtime_semantic_backbone = None
    semantic_feature_dim = 0
    semantic_masks = None
    if args.semantic_backbone == "siglip2":
        if args.siglip_cache_dir:
            semantic_cache = SigLIPGlobalCache(
                args.siglip_cache_dir,
                expected_views=parse_views(args.views),
                expected_model=args.siglip_model,
                expected_data_dir=args.data_dir,
            )
            missing_semantics = [
                Path(path).name
                for path in dataset.skin_paths
                if Path(path).name not in semantic_cache.filename_to_index
            ]
            if missing_semantics:
                raise ValueError(
                    f"SigLIP cache is missing {len(missing_semantics)} selected skins; "
                    f"first missing: {missing_semantics[0]}."
                )
            semantic_feature_dim = int(semantic_cache.metadata["feature_dim"])
        else:
            runtime_semantic_backbone = build_siglip_runtime(
                args.siglip_model,
                device,
                semantic_channels=args.semantic_channels,
                local_files_only=args.siglip_local_files_only,
            )
            semantic_feature_dim = runtime_semantic_backbone.raw_feature_dim
        semantic_masks = tuple(
            mask.to(device) for mask in build_part_layer_masks()
        )
    val_count = int(len(dataset) * args.val_split)
    train_count = len(dataset) - val_count
    generator = torch.Generator().manual_seed(args.seed)
    if val_count > 0:
        train_dataset, val_dataset = random_split(dataset, [train_count, val_count], generator=generator)
    else:
        train_dataset, val_dataset = dataset, None

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs) if val_dataset is not None else None

    renderer = DifferentiableRenderer(
        mappings_dir=args.mappings_dir,
        bg_color=tuple(channel / 255.0 for channel in args.bg_color),
    ).to(device)
    missing_views = [view for view in parse_views(args.views) if view not in renderer.views]
    if missing_views:
        raise ValueError(f"Unknown renderer views {missing_views}. Available views: {', '.join(renderer.views)}")

    affine_mode = args.parser_mode in ("geometry_fit", "global_affine")
    geometry_only = args.parser_mode == "geometry_fit"
    surface_classes = (
        surface_class_count(renderer, parse_views(args.views))
        if args.parser_mode in ("geometry_fit", "global_affine")
        else 0
    )
    model = DenseUVParserNet(
        base_channels=args.base_channels,
        uv_size=UV_SIZE,
        uv_classification=args.uv_classification and not geometry_only,
        view_classes=len(parse_views(args.views)),
        predict_affine=affine_mode,
        affine_translation_scale=args.translation_scale,
        affine_scale_range=args.scale_range,
        surface_classes=surface_classes,
        geometry_only=geometry_only,
        feature_dropout=args.feature_dropout,
        semantic_feature_dim=semantic_feature_dim,
        semantic_channels=args.semantic_channels,
        semantic_attention_heads=args.semantic_attention_heads,
        semantic_layers=args.semantic_layers,
        semantic_dropout=args.semantic_dropout,
        predict_confidence=args.semantic_backbone != "none",
        route_role_spatial_prior=(
            geometry_only and args.route_role_spatial_prior
        ),
        route_prior_height=args.route_prior_height,
        route_prior_width=args.route_prior_width,
        route_prior_logit_cap=args.route_prior_logit_cap,
        route_prior_dropout=args.route_prior_dropout,
    ).to(device)
    if runtime_semantic_backbone is not None:
        attach_siglip_runtime(
            model,
            args.siglip_model,
            device,
            local_files_only=args.siglip_local_files_only,
            backbone=runtime_semantic_backbone,
        )
    criterion = DenseUVParserLoss(
        lambda_foreground=args.lambda_foreground,
        lambda_layer=args.lambda_layer,
        lambda_part=args.lambda_part,
        lambda_face=args.lambda_face,
        lambda_layer_face=args.lambda_layer_face,
        lambda_uv=args.lambda_uv,
        lambda_uv_class=args.lambda_uv_class,
        lambda_affine=args.lambda_affine,
        lambda_surface=args.lambda_surface,
        lambda_outer_false_positive=args.lambda_outer_false_positive,
        lambda_outer_false_negative=args.lambda_outer_false_negative,
        lambda_route_confidence=args.lambda_route_confidence,
        lambda_primary_route_swap=args.lambda_primary_route_swap,
        lambda_route_texel_consistency=args.lambda_route_texel_consistency,
        lambda_route_prior_regularization=(
            args.lambda_route_prior_regularization
        ),
        outer_false_positive_gamma=args.outer_false_positive_gamma,
        outer_false_negative_gamma=args.outer_false_negative_gamma,
        primary_route_swap_gamma=args.primary_route_swap_gamma,
        route_prior_tv_weight=args.route_prior_tv_weight,
        route_class_weight_floor=args.route_class_weight_floor,
        route_outer_class_weight_cap=args.route_outer_class_weight_cap,
        uv_size=UV_SIZE,
        use_uv=(not affine_mode) or model.uv_classification,
        affine_translation_limit=model.affine_translation_limit if affine_mode else 1.0,
        affine_log_scale_limit=model.affine_log_scale_limit if affine_mode else 1.0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = build_grad_scaler(device, args.mixed_precision)

    start_epoch = 1
    best_metric = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        checkpoint_mode = checkpoint.get("model_config", {}).get(
            "parser_mode", checkpoint.get("args", {}).get("parser_mode", "dense")
        )
        if checkpoint_mode != args.parser_mode:
            raise ValueError(
                f"Cannot resume parser_mode={checkpoint_mode!r} as {args.parser_mode!r}. Start a new run."
            )
        checkpoint_semantic_backbone = checkpoint.get("model_config", {}).get(
            "semantic_backbone", "none"
        )
        if checkpoint_semantic_backbone != args.semantic_backbone:
            raise ValueError(
                "Cannot change semantic backbone while resuming: "
                f"checkpoint={checkpoint_semantic_backbone!r}, "
                f"requested={args.semantic_backbone!r}. Start a new run."
            )
        if args.semantic_backbone != "none":
            checkpoint_config = checkpoint.get("model_config", {})
            checkpoint_siglip_model = checkpoint_config.get(
                "siglip_model", checkpoint.get("args", {}).get("siglip_model")
            )
            if checkpoint_siglip_model != args.siglip_model:
                raise ValueError(
                    "Cannot change SigLIP model while resuming: "
                    f"checkpoint={checkpoint_siglip_model!r}, "
                    f"requested={args.siglip_model!r}. Start a new run."
                )
            semantic_resume_fields = {
                "semantic_feature_dim": model.semantic_feature_dim,
                "semantic_channels": model.semantic_channels,
                "semantic_attention_heads": model.semantic_attention_heads,
                "semantic_layers": model.semantic_layers,
            }
            mismatches = {
                name: (checkpoint_config.get(name), expected)
                for name, expected in semantic_resume_fields.items()
                if checkpoint_config.get(name) != expected
            }
            if mismatches:
                raise ValueError(
                    "Cannot change semantic adapter shape while resuming: "
                    f"{mismatches}. Start a new run."
                )
        checkpoint_layer_classes = checkpoint.get("model_config", {}).get("layer_classes", 2)
        if geometry_only and checkpoint_layer_classes != model.layer_classes:
            raise ValueError(
                "This geometry checkpoint predates the secondary/backface route class. "
                "Start a new parser run instead of resuming it."
            )
        checkpoint_surface_classes = checkpoint.get("model_config", {}).get("surface_classes", 0)
        if geometry_only and checkpoint_surface_classes != model.surface_classes:
            raise ValueError(
                "This geometry checkpoint predates exact surface-slot routing. "
                "Start a new parser run instead of resuming it."
            )
        checkpoint_config = checkpoint.get("model_config", {})
        checkpoint_route_prior = checkpoint_config.get(
            "route_role_spatial_prior",
            "route_role_prior" in checkpoint["model"],
        )
        if geometry_only and checkpoint_route_prior != model.route_role_spatial_prior:
            raise ValueError(
                "Cannot add or remove the learned route-role spatial prior while "
                "resuming. Start a new parser run, or explicitly disable the prior "
                "to continue a legacy checkpoint."
            )
        if checkpoint_route_prior:
            prior_shape = tuple(checkpoint["model"]["route_role_prior"].shape[-2:])
            expected_prior_shape = (
                model.route_prior_height,
                model.route_prior_width,
            )
            if prior_shape != expected_prior_shape:
                raise ValueError(
                    "Route-prior grid mismatch while resuming: "
                    f"checkpoint={prior_shape}, requested={expected_prior_shape}."
                )
        if args.parser_mode == "global_affine" and not any(
            key.startswith("layer_face.") for key in checkpoint["model"]
        ):
            raise ValueError("This global-affine checkpoint predates the joint layer-face head.")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr
            param_group["weight_decay"] = args.weight_decay
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])

        checkpoint_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = checkpoint_epoch + 1
        checkpoint_args = checkpoint.get("args", {})
        checkpoint_best_metric = checkpoint_args.get("best_metric", "loss_total")
        if checkpoint_best_metric != args.best_metric:
            print(
                "Resetting best-metric history because checkpoint selection changed: "
                f"checkpoint={checkpoint_best_metric!r}, requested={args.best_metric!r}."
            )
            best_metric = float("inf")
        else:
            best_metric = float(checkpoint.get("best_metric", float("inf")))
        if start_epoch > args.epochs:
            raise ValueError(
                f"Checkpoint is already at epoch {checkpoint_epoch}; set --epochs above {checkpoint_epoch}."
            )
        print(
            json.dumps(
                {
                    "resume": str(args.resume),
                    "checkpoint_epoch": checkpoint_epoch,
                    "start_epoch": start_epoch,
                    "target_epoch": args.epochs,
                    "lr": args.lr,
                    "best_metric": args.best_metric,
                    "best_metric_value": best_metric,
                },
                indent=2,
            )
        )

    metadata = {
        "num_samples": len(dataset),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset) if val_dataset is not None else 0,
        "views": parse_views(args.views),
        "parameters": count_parameters(model),
        "feature_dropout": args.feature_dropout,
        "semantic_backbone": args.semantic_backbone,
        "semantic_feature_dim": semantic_feature_dim,
        "semantic_channels": args.semantic_channels,
        "semantic_layers": args.semantic_layers,
        "predict_confidence": model.predict_confidence,
        "route_role_spatial_prior": model.route_role_spatial_prior,
        "route_prior_height": model.route_prior_height,
        "route_prior_width": model.route_prior_width,
        "route_prior_logit_cap": model.route_prior_logit_cap,
        "route_prior_dropout": model.route_prior_dropout,
        "device": str(device),
        "parser_mode": args.parser_mode,
        "uv_classification": model.uv_classification,
        "view_classes": len(parse_views(args.views)),
        "surface_classes": surface_classes,
        "layer_classes": model.layer_classes,
        "route_role_classes": model.layer_classes if geometry_only else 0,
        "geometry_only": geometry_only,
        "arm_model": "steve",
        "best_metric": args.best_metric,
        "base_learning_rate": args.lr,
        "lr_schedule": args.lr_schedule,
        "min_lr_ratio": args.min_lr_ratio,
        "augment": args.augment,
        "augment_validation": args.augment_validation,
        "background_augment": args.background_augment,
        "background_augment_prob": args.background_augment_prob,
        "semantic_gate": args.semantic_gate,
        "geometry_route_texel_consensus": args.geometry_route_texel_consensus,
        "outer_uv_min_source_pixels": args.outer_uv_min_source_pixels,
        "background_color_tolerance": args.background_color_tolerance,
        "color_background_tolerance": args.color_background_tolerance,
        "color_foreground_inset": args.color_foreground_inset,
        "affine_refine": args.affine_refine,
        "affine_refine_translation_px": args.affine_refine_translation_px,
        "affine_refine_scale": args.affine_refine_scale,
        "lambda_soft_uv_rgb": args.lambda_soft_uv_rgb,
        "lambda_soft_uv_alpha": args.lambda_soft_uv_alpha,
        "lambda_soft_uv_inner_recall": args.lambda_soft_uv_inner_recall,
        "lambda_soft_uv_outer_recall": args.lambda_soft_uv_outer_recall,
        "soft_uv_recall_hard_fraction": args.soft_uv_recall_hard_fraction,
        "soft_uv_recall_hard_weight": args.soft_uv_recall_hard_weight,
        "lambda_render_rgb": args.lambda_render_rgb,
        "lambda_render_alpha": args.lambda_render_alpha,
        "lambda_outer_false_positive": args.lambda_outer_false_positive,
        "lambda_outer_false_negative": args.lambda_outer_false_negative,
        "lambda_route_confidence": args.lambda_route_confidence,
        "lambda_primary_route_swap": args.lambda_primary_route_swap,
        "lambda_route_texel_consistency": args.lambda_route_texel_consistency,
        "lambda_route_prior_regularization": (
            args.lambda_route_prior_regularization
        ),
        "lambda_semantic_presence": args.lambda_semantic_presence,
        "lambda_semantic_coverage": args.lambda_semantic_coverage,
        "outer_false_positive_gamma": args.outer_false_positive_gamma,
        "outer_false_negative_gamma": args.outer_false_negative_gamma,
        "primary_route_swap_gamma": args.primary_route_swap_gamma,
        "route_prior_tv_weight": args.route_prior_tv_weight,
        "route_class_weight_floor": args.route_class_weight_floor,
        "route_outer_class_weight_cap": args.route_outer_class_weight_cap,
        "outer_selection_precision_weight": args.outer_selection_precision_weight,
        "outer_selection_recall_weight": args.outer_selection_recall_weight,
        "outer_selection_iou_weight": args.outer_selection_iou_weight,
        "inner_selection_recall_weight": args.inner_selection_recall_weight,
        "hard_rgb_selection_weight": args.hard_rgb_selection_weight,
        "render_softmax_temperature": args.render_softmax_temperature,
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump({"args": vars(args), "metadata": metadata}, handle, indent=2)
    print(json.dumps(metadata, indent=2))

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_lr = learning_rate_for_epoch(
            args.lr,
            epoch,
            args.epochs,
            schedule=args.lr_schedule,
            min_lr_ratio=args.min_lr_ratio,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = epoch_lr
        train_metrics = run_epoch(
            model,
            criterion,
            renderer,
            train_loader,
            optimizer,
            scaler,
            device,
            args.mixed_precision,
            args,
            train=True,
            compute_hard_metrics=(
                val_loader is None
                and args.best_metric
                in ("loss_hard_uv_selection", "loss_hard_uv_color_selection")
            ),
            semantic_cache=semantic_cache,
            semantic_masks=semantic_masks,
        )
        train_metrics["learning_rate"] = epoch_lr
        metrics = {"train": train_metrics}
        metric_source = train_metrics
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(
                    model,
                    criterion,
                    renderer,
                    val_loader,
                    optimizer,
                    scaler,
                    device,
                    args.mixed_precision,
                    args,
                    train=False,
                    compute_hard_metrics=True,
                    semantic_cache=semantic_cache,
                    semantic_masks=semantic_masks,
                )
            metrics["val"] = val_metrics
            metric_source = val_metrics

        if args.best_metric not in metric_source:
            raise ValueError(f"Best metric {args.best_metric!r} is not available in epoch metrics.")
        metric = metric_source[args.best_metric]
        print(f"epoch={epoch} metrics={json.dumps(metrics, sort_keys=True)}")

        is_best = metric < best_metric
        if is_best:
            best_metric = metric
        if epoch % args.save_every == 0:
            save_checkpoint(
                output_dir / "latest.pt", model, optimizer, scaler, epoch, args, metrics, best_metric=best_metric
            )
        if is_best:
            save_checkpoint(
                output_dir / "best.pt", model, optimizer, scaler, epoch, args, metrics, best_metric=best_metric
            )
        if epoch % args.preview_every == 0:
            save_preview(
                model,
                renderer,
                val_loader or train_loader,
                device,
                args,
                output_dir / "previews" / f"epoch_{epoch:04d}.png",
                semantic_cache=semantic_cache,
            )


if __name__ == "__main__":
    main()
