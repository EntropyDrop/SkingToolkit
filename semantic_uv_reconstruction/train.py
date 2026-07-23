import argparse
import copy
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TOOLKIT_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from SkingToolkit.semantic_uv_reconstruction.dataset import UVInpaintingDataset, finalize_minecraft_alpha, RenderAugmenter, parse_views  # noqa: E402
from SkingToolkit.semantic_uv_reconstruction.losses import UVInpaintingLoss  # noqa: E402
from SkingToolkit.semantic_uv_reconstruction.model import UVInpaintingNet, PatchGANDiscriminator, count_parameters  # noqa: E402
from SkingToolkit.semantic_uv_reconstruction.topology_model import TopologyAwareUVCompletionNet  # noqa: E402
from SkingToolkit.dense_uv_parser.model import DenseUVParserNet  # noqa: E402
from SkingToolkit.dense_uv_parser.semantic import attach_semantic_runtime  # noqa: E402
from SkingToolkit.dense_uv_parser.utils import (  # noqa: E402
    SPLAT_COLOR_AGGREGATIONS,
    randomize_render_background,
    splat_parser_predictions_to_uv_conditioning,
    surface_class_count,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def get_device(device_arg):
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_context(device, precision):
    if precision == "no" or device.type == "cpu":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def configure_torch(args, device):
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(args.matmul_precision)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = args.cudnn_benchmark


def move_batch(batch, device):
    return {
        "uv": batch["uv"].to(device, non_blocking=True),
        "path": batch["path"],
    }


def save_preview(pred_uv, gt_uv, output_path, max_items=4):
    count = min(max_items, pred_uv.shape[0])
    pred = finalize_minecraft_alpha(pred_uv[:count].detach().cpu())
    gt = finalize_minecraft_alpha(gt_uv[:count].detach().cpu())
    preview = torch.cat([pred, gt], dim=0)
    save_image(preview.clamp(0.0, 1.0), output_path, nrow=count)


def format_losses(loss_sums, count):
    formatted = {}
    for name, value in loss_sums.items():
        avg = value / max(count, 1)
        if torch.is_tensor(avg):
            avg = float(avg.detach().cpu())
        formatted[name] = avg
    return formatted


def current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def build_scheduler(optimizer, args, start_epoch):
    if args.scheduler == "none":
        return None
    remaining_epochs = max(args.epochs - start_epoch + 1, 1)
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=remaining_epochs,
            eta_min=args.min_lr,
        )
    raise ValueError(f"Unsupported scheduler={args.scheduler!r}.")


def load_dense_parser(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_args = checkpoint.get("args", {})
    model_config = checkpoint.get("model_config", {})
    if model_config.get("arm_model", "steve") != "steve":
        raise ValueError("Geometry parser only supports standard Steve arms.")
    state_dict = checkpoint["model"]
    has_uv_classification = any(key.startswith("uv_x.") or key.startswith("uv_y.") for key in state_dict)
    has_layer_face = any(key.startswith("layer_face.") for key in state_dict)
    has_route_prior = "route_role_prior" in state_dict
    has_outer_uv_occupancy = any(
        key.startswith("outer_uv_occupancy_head.") for key in state_dict
    )
    uv_classification = model_config.get("uv_classification", has_uv_classification)
    parser_mode = model_config.get("parser_mode", checkpoint_args.get("parser_mode", "dense"))
    predict_affine = model_config.get("predict_affine", parser_mode in ("global_affine", "geometry_fit"))
    geometry_only = model_config.get("geometry_only", parser_mode == "geometry_fit")
    layer_classes = model_config.get("layer_classes", 2)
    if geometry_only and layer_classes != 3:
        raise ValueError(
            "This geometry parser predates the secondary/backface route class. "
            "Train a new dense_uv_parser checkpoint before semantic_uv_reconstruction training."
        )
    model = DenseUVParserNet(
        base_channels=model_config.get("base_channels", checkpoint_args.get("base_channels", 32)),
        uv_size=model_config.get("uv_size", 64),
        uv_classification=uv_classification,
        layer_classes=layer_classes,
        layer_face_classes=model_config.get("layer_face_classes", 12 if has_layer_face else 0),
        view_classes=model_config.get("view_classes", 0),
        predict_affine=predict_affine,
        affine_translation_scale=model_config.get(
            "affine_translation_scale", checkpoint_args.get("translation_scale", 0.03)
        ),
        affine_scale_range=model_config.get("affine_scale_range", checkpoint_args.get("scale_range", 0.03)),
        surface_classes=model_config.get(
            "surface_classes",
            checkpoint_args.get("surface_classes", 0 if geometry_only else 2 if predict_affine else 0),
        ),
        geometry_only=geometry_only,
        feature_dropout=model_config.get(
            "feature_dropout", checkpoint_args.get("feature_dropout", 0.0)
        ),
        semantic_feature_dim=model_config.get("semantic_feature_dim", 0),
        semantic_channels=model_config.get("semantic_channels", 128),
        semantic_attention_heads=model_config.get("semantic_attention_heads", 4),
        semantic_layers=model_config.get("semantic_layers", 1),
        semantic_dropout=model_config.get("semantic_dropout", 0.05),
        semantic_spatial_feature_dim=model_config.get(
            "semantic_spatial_feature_dim", 0
        ),
        semantic_spatial_channels=model_config.get(
            "semantic_spatial_channels", 64
        ),
        predict_confidence=model_config.get(
            "predict_confidence",
            any(key.startswith("route_confidence.") for key in state_dict),
        ),
        route_role_spatial_prior=model_config.get(
            "route_role_spatial_prior", has_route_prior
        ),
        route_prior_height=model_config.get(
            "route_prior_height",
            state_dict["route_role_prior"].shape[-2] if has_route_prior else 32,
        ),
        route_prior_width=model_config.get(
            "route_prior_width",
            state_dict["route_role_prior"].shape[-1] if has_route_prior else 16,
        ),
        route_prior_logit_cap=model_config.get("route_prior_logit_cap", 1.5),
        route_prior_dropout=model_config.get("route_prior_dropout", 0.0),
        predict_outer_uv_occupancy=model_config.get(
            "predict_outer_uv_occupancy", has_outer_uv_occupancy
        ),
    ).to(device)
    model.load_state_dict(state_dict)
    if (
        model.semantic_feature_dim > 0
        or model.semantic_spatial_feature_dim > 0
    ):
        semantic_backbone = model_config.get("semantic_backbone", "siglip2")
        semantic_model = model_config.get(
            "semantic_model",
            model_config.get(
                "tipsv2_model"
                if semantic_backbone == "tipsv2"
                else "siglip_model",
                checkpoint_args.get(
                    "tipsv2_model"
                    if semantic_backbone == "tipsv2"
                    else "siglip_model",
                    "google/tipsv2-b14"
                    if semantic_backbone == "tipsv2"
                    else "google/siglip2-base-patch16-224",
                ),
            ),
        )
        attach_semantic_runtime(
            model,
            semantic_backbone,
            semantic_model,
            device,
            local_files_only=bool(
                checkpoint_args.get(
                    "tipsv2_local_files_only"
                    if semantic_backbone == "tipsv2"
                    else "siglip_local_files_only",
                    False,
                )
            ),
            runtime_batch_size=int(
                checkpoint_args.get("semantic_runtime_batch_size", 32)
            ),
        )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, checkpoint_args


def build_dense_parser_conditioning(
    skin,
    renderer,
    views,
    dense_parser,
    augmenter=None,
    parser_background_augment=False,
    parser_background_augment_prob=0.9,
    fg_threshold=0.5,
    semantic_gate=True,
    affine_refine=True,
    affine_refine_translation_px=8.0,
    affine_refine_scale=0.0,
    route_confidence_threshold=0.0,
    route_margin_threshold=0.0,
    outer_route_confidence_threshold=0.55,
    outer_route_margin_threshold=0.35,
    outer_uv_min_coverage=0.0,
    outer_uv_min_source_pixels=15,
    outer_geometry_rescue=True,
    outer_rescue_confidence_threshold=0.60,
    outer_rescue_margin_threshold=0.25,
    outer_rescue_min_coverage=0.10,
    color_aggregation="grid_mode",
    geometry_route_texel_consensus=False,
    reject_semantic_fallback=True,
    confidence_aware_conditioning=False,
    bg_color=(128, 128, 128),
    return_renders=False,
):
    """Build semantic_uv_reconstruction conditioning through the same parser+splat path used at inference."""
    is_batched = skin.dim() == 4
    skin_batch = skin if is_batched else skin.unsqueeze(0)

    rendered_by_view = []
    observed_foreground_by_view = []
    gt_renders = {} if return_renders else None
    with torch.no_grad():
        for view in views:
            clean_render = renderer.forward_view(skin_batch, view)
            if return_renders:
                gt_renders[view] = clean_render if is_batched else clean_render.squeeze(0)
            parser_render = augmenter(clean_render) if augmenter is not None else clean_render
            observed_foreground_by_view.append(parser_render[:, 3] > 0.5)
            parser_render = randomize_render_background(
                parser_render,
                probability=parser_background_augment_prob if parser_background_augment else 0.0,
                bg_color=bg_color,
            )
            rendered_by_view.append(parser_render)

        rendered = torch.stack(rendered_by_view, dim=1)
        observed_foreground = torch.stack(observed_foreground_by_view, dim=1)
        B, V, C, H, W = rendered.shape
        rendered = rendered.reshape(B * V, C, H, W)
        observed_foreground = observed_foreground.reshape(B * V, H, W)
        view_ids = torch.arange(V, device=rendered.device).view(1, V).expand(B, -1).reshape(B * V)
        outputs = dense_parser(
            rendered,
            view_ids=view_ids,
            semantic_foreground=observed_foreground,
        )
        conditioning = splat_parser_predictions_to_uv_conditioning(
            rendered,
            outputs,
            renderer=renderer,
            views=views,
            group_size=V,
            fg_threshold=fg_threshold,
            bg_color=bg_color,
            semantic_gate=semantic_gate,
            affine_refine=affine_refine,
            affine_refine_translation_px=affine_refine_translation_px,
            affine_refine_scale=affine_refine_scale,
            route_confidence_threshold=route_confidence_threshold,
            route_margin_threshold=route_margin_threshold,
            outer_route_confidence_threshold=outer_route_confidence_threshold,
            outer_route_margin_threshold=outer_route_margin_threshold,
            outer_uv_min_coverage=outer_uv_min_coverage,
            outer_uv_min_source_pixels=outer_uv_min_source_pixels,
            outer_geometry_rescue=outer_geometry_rescue,
            outer_rescue_confidence_threshold=outer_rescue_confidence_threshold,
            outer_rescue_margin_threshold=outer_rescue_margin_threshold,
            outer_rescue_min_coverage=outer_rescue_min_coverage,
            color_aggregation=color_aggregation,
            geometry_route_texel_consensus=geometry_route_texel_consensus,
            observed_foreground=observed_foreground,
            reject_semantic_fallback=reject_semantic_fallback,
            include_confidence=confidence_aware_conditioning,
        )

    if not is_batched:
        conditioning = conditioning.squeeze(0)
    if return_renders:
        return conditioning, gt_renders
    return conditioning


def build_training_conditioning(
    skin,
    renderer,
    views,
    dense_parser,
    augmenter=None,
    parser_background_augment=False,
    parser_background_augment_prob=0.9,
    parser_splat_fg_threshold=0.5,
    parser_semantic_gate=True,
    parser_affine_refine=True,
    parser_affine_refine_translation_px=8.0,
    parser_affine_refine_scale=0.0,
    parser_route_confidence_threshold=0.0,
    parser_route_margin_threshold=0.0,
    parser_outer_route_confidence_threshold=0.55,
    parser_outer_route_margin_threshold=0.35,
    parser_outer_uv_min_coverage=0.0,
    parser_outer_uv_min_source_pixels=15,
    parser_outer_geometry_rescue=True,
    parser_outer_rescue_confidence_threshold=0.60,
    parser_outer_rescue_margin_threshold=0.25,
    parser_outer_rescue_min_coverage=0.10,
    parser_splat_color_aggregation="grid_mode",
    parser_geometry_route_texel_consensus=False,
    parser_reject_semantic_fallback=True,
    confidence_aware_conditioning=False,
    bg_color=(128, 128, 128),
    return_renders=False,
):
    if dense_parser is None:
        raise ValueError("Dense parser conditioning requires --parser_checkpoint.")
    return build_dense_parser_conditioning(
        skin,
        renderer,
        views,
        dense_parser,
        augmenter=augmenter,
        parser_background_augment=parser_background_augment,
        parser_background_augment_prob=parser_background_augment_prob,
        fg_threshold=parser_splat_fg_threshold,
        semantic_gate=parser_semantic_gate,
        affine_refine=parser_affine_refine,
        affine_refine_translation_px=parser_affine_refine_translation_px,
        affine_refine_scale=parser_affine_refine_scale,
        route_confidence_threshold=parser_route_confidence_threshold,
        route_margin_threshold=parser_route_margin_threshold,
        outer_route_confidence_threshold=parser_outer_route_confidence_threshold,
        outer_route_margin_threshold=parser_outer_route_margin_threshold,
        outer_uv_min_coverage=parser_outer_uv_min_coverage,
        outer_uv_min_source_pixels=parser_outer_uv_min_source_pixels,
        outer_geometry_rescue=parser_outer_geometry_rescue,
        outer_rescue_confidence_threshold=parser_outer_rescue_confidence_threshold,
        outer_rescue_margin_threshold=parser_outer_rescue_margin_threshold,
        outer_rescue_min_coverage=parser_outer_rescue_min_coverage,
        color_aggregation=parser_splat_color_aggregation,
        geometry_route_texel_consensus=parser_geometry_route_texel_consensus,
        reject_semantic_fallback=parser_reject_semantic_fallback,
        confidence_aware_conditioning=confidence_aware_conditioning,
        bg_color=bg_color,
        return_renders=return_renders,
    )


def run_epoch(
    model,
    criterion,
    loader,
    optimizer,
    scaler,
    device,
    precision,
    train=True,
    d_optimizer=None,
    views=None,
    augmenter=None,
    dense_parser=None,
    parser_background_augment=False,
    parser_background_augment_prob=0.9,
    parser_splat_fg_threshold=0.5,
    parser_semantic_gate=True,
    parser_affine_refine=True,
    parser_affine_refine_translation_px=8.0,
    parser_affine_refine_scale=0.0,
    parser_route_confidence_threshold=0.0,
    parser_route_margin_threshold=0.0,
    parser_outer_route_confidence_threshold=0.55,
    parser_outer_route_margin_threshold=0.35,
    parser_outer_uv_min_coverage=0.0,
    parser_outer_uv_min_source_pixels=15,
    parser_outer_geometry_rescue=True,
    parser_outer_rescue_confidence_threshold=0.60,
    parser_outer_rescue_margin_threshold=0.25,
    parser_outer_rescue_min_coverage=0.10,
    parser_splat_color_aggregation="grid_mode",
    parser_geometry_route_texel_consensus=False,
    parser_reject_semantic_fallback=True,
    bg_color=(128, 128, 128),
    log_every=50,
    topology_drop_known_min=0.1,
    topology_drop_known_max=0.5,
    topology_teacher_reveal_unknown=0.1,
    lambda_rgb_token=1.0,
    lambda_rgb_distribution=2.0,
    lambda_alpha_token=0.5,
    ignore_covered_inner=True,
    covered_inner_alpha_threshold=0.1,
):
    model.train(train)
    if criterion.discriminator is not None:
        criterion.discriminator.train(train)
    loss_sums = {}
    sample_count = 0
    step_count = 0
    total_steps = len(loader) if hasattr(loader, "__len__") else None
    iterator = tqdm(loader, leave=False, file=sys.__stderr__ or sys.stderr) if tqdm is not None else loader

    for batch in iterator:
        step_count += 1
        batch = move_batch(batch, device)
        batch_size = batch["uv"].shape[0]

        # Build conditioning on GPU (fast grid_sample vs CPU in DataLoader workers)
        with torch.no_grad():
            batch_augmenter = augmenter
            result = build_training_conditioning(
                batch["uv"],
                criterion.renderer,
                views,
                dense_parser=dense_parser,
                augmenter=batch_augmenter,
                parser_background_augment=train and parser_background_augment,
                parser_background_augment_prob=parser_background_augment_prob,
                parser_splat_fg_threshold=parser_splat_fg_threshold,
                parser_semantic_gate=parser_semantic_gate,
                parser_affine_refine=parser_affine_refine,
                parser_affine_refine_translation_px=parser_affine_refine_translation_px,
                parser_affine_refine_scale=parser_affine_refine_scale,
                parser_route_confidence_threshold=parser_route_confidence_threshold,
                parser_route_margin_threshold=parser_route_margin_threshold,
                parser_outer_route_confidence_threshold=parser_outer_route_confidence_threshold,
                parser_outer_route_margin_threshold=parser_outer_route_margin_threshold,
                parser_outer_uv_min_coverage=parser_outer_uv_min_coverage,
                parser_outer_uv_min_source_pixels=parser_outer_uv_min_source_pixels,
                parser_outer_geometry_rescue=parser_outer_geometry_rescue,
                parser_outer_rescue_confidence_threshold=parser_outer_rescue_confidence_threshold,
                parser_outer_rescue_margin_threshold=parser_outer_rescue_margin_threshold,
                parser_outer_rescue_min_coverage=parser_outer_rescue_min_coverage,
                parser_splat_color_aggregation=parser_splat_color_aggregation,
                parser_geometry_route_texel_consensus=parser_geometry_route_texel_consensus,
                parser_reject_semantic_fallback=parser_reject_semantic_fallback,
                bg_color=bg_color,
                confidence_aware_conditioning=(
                    getattr(model, "input_channels", 10) == 12
                ),
                return_renders=True,
            )
            conditioning, gt_renders = result
            if train and hasattr(model, "augment_training_conditioning"):
                conditioning = model.augment_training_conditioning(
                    conditioning,
                    batch["uv"],
                    drop_known_min=topology_drop_known_min,
                    drop_known_max=topology_drop_known_max,
                    teacher_reveal_unknown=topology_teacher_reveal_unknown,
                )

        with torch.set_grad_enabled(train):
            with autocast_context(device, precision):
                model_outputs = None
                if hasattr(model, "masked_token_loss"):
                    model_outputs = model(conditioning, return_logits=True)
                    pred_uv = model_outputs["uv"]
                else:
                    pred_uv = model(conditioning)
                losses = criterion(pred_uv, batch["uv"], gt_renders=gt_renders)
                if model_outputs is not None:
                    token_losses = model.masked_token_loss(
                        model_outputs,
                        batch["uv"],
                        lambda_rgb_token=lambda_rgb_token,
                        lambda_rgb_distribution=lambda_rgb_distribution,
                        lambda_alpha_token=lambda_alpha_token,
                        ignore_covered_inner=ignore_covered_inner,
                        covered_inner_alpha_threshold=covered_inner_alpha_threshold,
                    )
                    continuous_recon = losses["loss_recon_total"]
                    losses["loss_recon_continuous"] = continuous_recon
                    losses["loss_recon_total"] = continuous_recon + token_losses["loss_token"]
                    losses["loss_total"] = losses["loss_total"] + token_losses["loss_token"]
                    losses.update(token_losses)
                loss = losses["loss_total"]

        if train:
            loss_d = losses.get("loss_d", None)
            has_d = d_optimizer is not None and loss_d is not None

            # Zero both optimizers before any backward
            if has_d:
                d_optimizer.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)

            # Accumulate discriminator gradients (retain graph for generator's GAN loss)
            if has_d:
                if scaler is not None:
                    scaler.scale(loss_d).backward(retain_graph=True)
                else:
                    loss_d.backward(retain_graph=True)

            # Accumulate generator gradients
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Step discriminator
            if has_d:
                if scaler is not None:
                    scaler.unscale_(d_optimizer)
                torch.nn.utils.clip_grad_norm_(criterion.discriminator.parameters(), 1.0)
                if scaler is not None:
                    scaler.step(d_optimizer)
                else:
                    d_optimizer.step()

            # Step generator
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        sample_count += batch_size
        for name, value in losses.items():
            detached = value.detach()
            loss_sums[name] = loss_sums.get(name, detached.new_zeros(())) + detached * batch_size
        should_log = (
            tqdm is not None
            and log_every > 0
            and (step_count == 1 or step_count % log_every == 0 or step_count == total_steps)
        )
        if should_log:
            avg = format_losses(loss_sums, sample_count)
            iterator.set_postfix(
                total=f"{avg['loss_total']:.4f}",
                recon=f"{avg.get('loss_recon_total', avg['loss_total']):.4f}",
                rgb=f"{avg['loss_rgb']:.4f}",
            )

    return format_losses(loss_sums, sample_count)


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    args,
    input_channels,
    metrics,
    discriminator=None,
    d_optimizer=None,
    scheduler=None,
    d_scheduler=None,
    best_metric=None,
):
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "input_channels": input_channels,
        "metrics": metrics,
    }
    if hasattr(model, "checkpoint_config"):
        checkpoint["model_config"] = model.checkpoint_config()
    else:
        checkpoint["model_config"] = {
            "model_type": "unet",
            "input_channels": input_channels,
            "base_channels": getattr(args, "base_channels", 64),
            "preserve_known": getattr(args, "preserve_known", True),
            "arm_model": "steve",
        }
    if best_metric is not None:
        checkpoint["best_metric"] = best_metric
    if discriminator is not None:
        checkpoint["discriminator"] = discriminator.state_dict()
    if d_optimizer is not None:
        checkpoint["d_optimizer"] = d_optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    if d_scheduler is not None:
        checkpoint["d_scheduler"] = d_scheduler.state_dict()
    torch.save(checkpoint, path)


def build_grad_scaler(device, precision):
    enabled = device.type == "cuda" and precision == "fp16"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train the Minecraft skin Semantic UV reconstruction model.")
    parser.add_argument("--data_dir", required=True, help="Folder containing GT 64x64 RGBA skin PNGs.")
    parser.add_argument("--output_dir", default="semantic_uv_reconstruction_runs/default", help="Checkpoint/output folder.")
    parser.add_argument(
        "--preserve_known",
        dest="preserve_known",
        action="store_true",
        default=True,
        help="Hard-copy trusted known conditioning texels to the output.",
    )
    parser.add_argument("--no_preserve_known", dest="preserve_known", action="store_false")
    parser.add_argument("--mappings_dir", default=None, help="Renderer mappings directory.")
    parser.add_argument("--views", default="static_front,static_back", help="Comma-separated render views.")
    parser.add_argument(
        "--render_size",
        type=int,
        default=256,
        help="Deprecated compatibility option; UV unprojection uses each view's native mapping size.",
    )
    parser.add_argument(
        "--include_alpha",
        action="store_true",
        help="Deprecated compatibility option; UV unprojection always builds RGBA plus mask conditioning.",
    )
    parser.add_argument(
        "--completion_model",
        choices=["unet", "topology_maskgit"],
        default="unet",
        help="UV completion architecture. topology_maskgit uses cuboid topology and discrete masked generation.",
    )
    parser.add_argument("--base_channels", type=int, default=64, help="Base channel width for UVInpaintingNet.")
    parser.add_argument("--topology_channels", type=int, default=128)
    parser.add_argument("--topology_layers", type=int, default=4)
    parser.add_argument("--topology_attention_heads", type=int, default=4)
    parser.add_argument("--topology_dropout", type=float, default=0.05)
    parser.add_argument("--topology_hard_lock_threshold", type=float, default=0.0)
    parser.add_argument("--topology_drop_known_min", type=float, default=0.1)
    parser.add_argument("--topology_drop_known_max", type=float, default=0.5)
    parser.add_argument("--topology_teacher_reveal_unknown", type=float, default=0.1)
    parser.add_argument("--lambda_rgb_token", type=float, default=1.0)
    parser.add_argument("--lambda_rgb_distribution", type=float, default=2.0)
    parser.add_argument("--lambda_alpha_token", type=float, default=0.5)
    parser.add_argument("--preview_generation_steps", type=int, default=4)
    parser.add_argument("--preview_generation_temperature", type=float, default=0.0)
    parser.add_argument(
        "--preview_rgb_decode",
        choices=["mean", "argmax"],
        default="mean",
        help="Deterministic topology RGB decoder used by epoch previews.",
    )
    parser.add_argument(
        "--preview_palette_snap",
        dest="preview_palette_snap",
        action="store_true",
        default=True,
        help="Constrain generated preview colors to observed parser RGB triplets.",
    )
    parser.add_argument(
        "--no_preview_palette_snap",
        dest="preview_palette_snap",
        action="store_false",
    )
    parser.add_argument("--preview_palette_min_confidence", type=float, default=0.75)
    parser.add_argument("--augment", dest="augment", action="store_true", default=False, help="Enable optional render-space geometric augmentation.")
    parser.add_argument("--no_augment", dest="augment", action="store_false")
    parser.add_argument("--augment_validation", dest="augment_validation", action="store_true", default=False)
    parser.add_argument("--no_augment_validation", dest="augment_validation", action="store_false")
    parser.add_argument("--translation_scale", type=float, default=0.0, help="Render-space translation scale for optional augmentation.")
    parser.add_argument("--scale_range", type=float, default=0.0, help="Render-space uniform scale range for optional augmentation.")
    parser.add_argument("--perspective_scale", type=float, default=0.0, help="Render-space perspective warp scale for augmentation.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="no")
    parser.add_argument("--matmul_precision", choices=["highest", "high", "medium"], default="high")
    parser.add_argument("--cudnn_benchmark", dest="cudnn_benchmark", action="store_true", default=True)
    parser.add_argument("--no_cudnn_benchmark", dest="cudnn_benchmark", action="store_false")
    parser.add_argument("--lambda_rgb", type=float, default=2.0)
    parser.add_argument("--lambda_alpha", type=float, default=0.8)
    parser.add_argument("--lambda_alpha_dice", type=float, default=0.5)
    parser.add_argument("--lambda_alpha_edge", type=float, default=0.5)
    parser.add_argument("--lambda_render", type=float, default=0.2)
    parser.add_argument("--lambda_render_alpha", type=float, default=0.4)
    parser.add_argument("--lambda_edge", type=float, default=1.0)
    parser.add_argument("--lambda_gan", type=float, default=0.0, help="PatchGAN adversarial loss weight.")
    parser.add_argument("--render_foreground_weight", type=float, default=1.0)
    parser.add_argument(
        "--supervise_covered_inner",
        action="store_true",
        help="Also supervise inner-layer UV texels hidden behind opaque matching outer-layer texels.",
    )
    parser.add_argument(
        "--covered_inner_alpha_threshold",
        type=float,
        default=0.1,
        help="GT outer-layer alpha threshold used to ignore matching covered inner-layer UV texels.",
    )
    parser.add_argument(
        "--parser_checkpoint",
        default=None,
        help="Dense UV parser checkpoint used to generate inpaint conditioning.",
    )
    parser.add_argument(
        "--parser_splat_fg_threshold",
        type=float,
        default=0.5,
        help="Foreground threshold used when splatting dense parser predictions.",
    )
    parser.add_argument("--parser_semantic_gate", dest="parser_semantic_gate", action="store_true", default=None)
    parser.add_argument("--no_parser_semantic_gate", dest="parser_semantic_gate", action="store_false")
    parser.add_argument("--parser_affine_refine", dest="parser_affine_refine", action="store_true", default=None)
    parser.add_argument("--no_parser_affine_refine", dest="parser_affine_refine", action="store_false")
    parser.add_argument("--parser_affine_refine_translation_px", type=float, default=None)
    parser.add_argument("--parser_affine_refine_scale", type=float, default=None)
    parser.add_argument("--parser_route_confidence_threshold", type=float, default=0.0)
    parser.add_argument("--parser_route_margin_threshold", type=float, default=0.0)
    parser.add_argument("--parser_outer_route_confidence_threshold", type=float, default=0.80)
    parser.add_argument("--parser_outer_route_margin_threshold", type=float, default=0.55)
    parser.add_argument("--parser_outer_uv_min_coverage", type=float, default=None)
    parser.add_argument(
        "--parser_outer_uv_min_source_pixels", type=int, default=15
    )
    parser.add_argument(
        "--parser_outer_geometry_rescue",
        dest="parser_outer_geometry_rescue",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_parser_outer_geometry_rescue",
        dest="parser_outer_geometry_rescue",
        action="store_false",
    )
    parser.add_argument("--parser_outer_rescue_confidence_threshold", type=float, default=0.60)
    parser.add_argument("--parser_outer_rescue_margin_threshold", type=float, default=0.25)
    parser.add_argument("--parser_outer_rescue_min_coverage", type=float, default=0.10)
    parser.add_argument(
        "--parser_geometry_route_texel_consensus",
        dest="parser_geometry_route_texel_consensus",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no_parser_geometry_route_texel_consensus",
        dest="parser_geometry_route_texel_consensus",
        action="store_false",
    )
    parser.add_argument(
        "--parser_splat_color_aggregation",
        choices=SPLAT_COLOR_AGGREGATIONS,
        default="grid_mode",
    )
    parser.add_argument(
        "--parser_allow_semantic_fallback",
        action="store_true",
        help="Keep parser pixels that required a semantic routing fallback.",
    )
    parser.add_argument("--parser_background_augment", dest="parser_background_augment", action="store_true", default=True)
    parser.add_argument("--no_parser_background_augment", dest="parser_background_augment", action="store_false")
    parser.add_argument("--parser_background_augment_prob", type=float, default=0.9)
    parser.add_argument(
        "--best_metric",
        default="loss_recon_total",
        help="Metric key used to select best.pt. Defaults to reconstruction loss without GAN.",
    )
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument(
        "--resume_lr",
        type=float,
        default=None,
        help="Override optimizer learning rates after loading a resumed checkpoint.",
    )
    parser.add_argument("--log_every", type=int, default=50, help="Progress-bar metric sync interval in batches.")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--preview_every", type=int, default=1)
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from.")
    return parser


class Logger(object):
    """Mirror stdout/stderr to a file while retaining terminal stream APIs."""

    def __init__(self, filename, stream, mode="w"):
        self.terminal = stream
        self.log = open(filename, mode, encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        if "\r" in message:
            return
        if message:
            self.log.write(message)
            self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        """Keep compatibility with Transformers, tqdm, and color detection."""
        isatty = getattr(self.terminal, "isatty", None)
        return bool(isatty()) if isatty is not None else False

    def fileno(self):
        """Expose the underlying descriptor to terminal-aware libraries."""
        return self.terminal.fileno()

    @property
    def encoding(self):
        return getattr(self.terminal, "encoding", None) or "utf-8"


def main():
    args = build_arg_parser().parse_args()
    args.conditioning_mode = "dense_parser_inpaint"
    torch.manual_seed(args.seed)
    if not 0.0 <= args.topology_drop_known_min <= args.topology_drop_known_max <= 1.0:
        raise ValueError("Topology known-drop ratios must satisfy 0 <= min <= max <= 1.")
    if not 0.0 <= args.topology_teacher_reveal_unknown <= 1.0:
        raise ValueError("--topology_teacher_reveal_unknown must be in [0, 1].")
    if not 0.0 <= args.topology_hard_lock_threshold <= 1.0:
        raise ValueError("--topology_hard_lock_threshold must be in [0, 1].")
    if args.topology_attention_heads < 1:
        raise ValueError("--topology_attention_heads must be positive.")
    if args.topology_channels % args.topology_attention_heads != 0:
        raise ValueError("Topology channels must be divisible by attention heads.")
    if args.topology_layers < 1:
        raise ValueError("--topology_layers must be positive.")
    if not 0.0 <= args.topology_dropout < 1.0:
        raise ValueError("--topology_dropout must be in [0, 1).")
    if (
        args.lambda_rgb_token < 0.0
        or args.lambda_rgb_distribution < 0.0
        or args.lambda_alpha_token < 0.0
    ):
        raise ValueError("Topology token-loss weights must be non-negative.")
    if args.preview_generation_steps < 1 or args.preview_generation_temperature < 0.0:
        raise ValueError("Preview generation requires positive steps and non-negative temperature.")
    if not 0.0 <= args.preview_palette_min_confidence <= 1.0:
        raise ValueError("--preview_palette_min_confidence must be in [0, 1].")
    if args.parser_outer_uv_min_source_pixels < 1:
        raise ValueError(
            "--parser_outer_uv_min_source_pixels must be positive."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "previews").mkdir(exist_ok=True)

    log_mode = "a" if args.resume else "w"
    sys.stdout = Logger(output_dir / "train.log", sys.stdout, mode=log_mode)
    sys.stderr = Logger(output_dir / "train.log", sys.stderr, mode=log_mode)

    device = get_device(args.device)
    configure_torch(args, device)
    if not args.parser_checkpoint:
        raise ValueError("--parser_checkpoint is required for Semantic UV reconstruction training.")
    dense_parser, parser_checkpoint_args = load_dense_parser(args.parser_checkpoint, device)
    if args.parser_semantic_gate is None:
        args.parser_semantic_gate = parser_checkpoint_args.get("semantic_gate", True)
    if args.parser_affine_refine is None:
        args.parser_affine_refine = parser_checkpoint_args.get("affine_refine", True)
    if args.parser_affine_refine_translation_px is None:
        checkpoint_translation_px = parser_checkpoint_args.get("affine_refine_translation_px")
        args.parser_affine_refine_translation_px = (
            8.0 if checkpoint_translation_px is None else checkpoint_translation_px
        )
    if args.parser_affine_refine_scale is None:
        checkpoint_scale = parser_checkpoint_args.get("affine_refine_scale")
        args.parser_affine_refine_scale = 0.0 if checkpoint_scale is None else checkpoint_scale
    if args.parser_geometry_route_texel_consensus is None:
        args.parser_geometry_route_texel_consensus = parser_checkpoint_args.get(
            "geometry_route_texel_consensus", False
        )
    if args.parser_outer_uv_min_coverage is None:
        args.parser_outer_uv_min_coverage = parser_checkpoint_args.get(
            "outer_uv_min_coverage", 0.0
        )
    parser_views = parse_views(parser_checkpoint_args.get("views", ""))
    if parser_views and parser_views != parse_views(args.views):
        raise ValueError(
            "Parser checkpoint views do not match semantic_uv_reconstruction training views: "
            f"parser={parser_views}, semantic_uv_reconstruction={parse_views(args.views)}"
        )

    dataset = UVInpaintingDataset(
        data_dir=args.data_dir,
        mappings_dir=args.mappings_dir,
        views=args.views,
        image_size=args.render_size,
        include_alpha=args.include_alpha,
        max_samples=args.max_samples,
        augment=args.augment,
        translation_scale=args.translation_scale,
        scale_range=args.scale_range,
        perspective_scale=args.perspective_scale,
    )
    input_channels = (
        12 if args.completion_model == "topology_maskgit" else dataset.input_channels
    )

    val_len = int(len(dataset) * args.val_split) if args.val_split > 0 else 0
    if val_len > 0 and len(dataset) - val_len > 0:
        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(len(dataset), generator=generator).tolist()
        val_indices = indices[:val_len]
        train_indices = indices[val_len:]

        from torch.utils.data import Subset
        train_dataset = Subset(dataset, train_indices)

        val_dataset_base = copy.copy(dataset)
        val_dataset_base.augment = False
        val_dataset = Subset(val_dataset_base, val_indices)
    else:
        train_dataset, val_dataset = dataset, None

    pin_memory = device.type == "cuda"
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            **loader_kwargs,
        )

    if args.completion_model == "topology_maskgit":
        model = TopologyAwareUVCompletionNet(
            input_channels=input_channels,
            hidden_channels=args.topology_channels,
            layers=args.topology_layers,
            attention_heads=args.topology_attention_heads,
            dropout=args.topology_dropout,
            preserve_known=args.preserve_known,
            hard_lock_threshold=args.topology_hard_lock_threshold,
        ).to(device)
    else:
        model = UVInpaintingNet(
            input_channels=input_channels,
            base_channels=args.base_channels,
            preserve_known=args.preserve_known,
        ).to(device)

    discriminator = None
    d_optimizer = None
    if args.lambda_gan > 0:
        discriminator = PatchGANDiscriminator(base_channels=64).to(device)
        d_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=args.lr, weight_decay=0.0)

    criterion = UVInpaintingLoss(
        mappings_dir=args.mappings_dir,
        views=args.views,
        lambda_rgb=args.lambda_rgb,
        lambda_alpha=args.lambda_alpha,
        lambda_alpha_dice=args.lambda_alpha_dice,
        lambda_alpha_edge=args.lambda_alpha_edge,
        lambda_render=args.lambda_render,
        lambda_render_alpha=args.lambda_render_alpha,
        lambda_edge=args.lambda_edge,
        lambda_gan=args.lambda_gan,
        render_foreground_weight=args.render_foreground_weight,
        ignore_covered_inner=not args.supervise_covered_inner,
        covered_inner_alpha_threshold=args.covered_inner_alpha_threshold,
        discriminator=discriminator,
    ).to(device)
    if dense_parser.predict_affine and dense_parser.surface_classes > 0:
        mapping_surface_classes = surface_class_count(criterion.renderer, parse_views(args.views))
        if dense_parser.surface_classes != mapping_surface_classes:
            raise ValueError(
                "Parser/mapping surface-slot mismatch: "
                f"checkpoint={dense_parser.surface_classes}, mappings={mapping_surface_classes}."
            )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = build_grad_scaler(device, args.mixed_precision)

    start_epoch = 1
    best_metric = float("inf")
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        checkpoint_model_type = checkpoint.get("model_config", {}).get("model_type", "unet")
        if checkpoint_model_type != args.completion_model:
            raise ValueError(
                "Cannot resume with a different completion model: "
                f"checkpoint={checkpoint_model_type}, requested={args.completion_model}."
            )
        checkpoint_preserve_known = checkpoint.get("args", {}).get("preserve_known", True)
        if bool(checkpoint_preserve_known) != args.preserve_known:
            raise ValueError(
                "Cannot resume with a different preserve-known mode: "
                f"checkpoint={bool(checkpoint_preserve_known)}, requested={args.preserve_known}."
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if discriminator is not None and "discriminator" in checkpoint:
            discriminator.load_state_dict(checkpoint["discriminator"])
        if d_optimizer is not None and "d_optimizer" in checkpoint:
            d_optimizer.load_state_dict(checkpoint["d_optimizer"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_metric = checkpoint.get("best_metric", best_metric)
        if args.resume_lr is not None:
            set_optimizer_lr(optimizer, args.resume_lr)
            if d_optimizer is not None:
                set_optimizer_lr(d_optimizer, args.resume_lr)

    scheduler = build_scheduler(optimizer, args, start_epoch)
    d_scheduler = build_scheduler(d_optimizer, args, start_epoch) if d_optimizer is not None else None
    if (
        scheduler is not None
        and checkpoint is not None
        and "scheduler" in checkpoint
        and args.resume_lr is None
    ):
        scheduler.load_state_dict(checkpoint["scheduler"])
    if (
        d_scheduler is not None
        and checkpoint is not None
        and "d_scheduler" in checkpoint
        and args.resume_lr is None
    ):
        d_scheduler.load_state_dict(checkpoint["d_scheduler"])

    metadata = {
        "num_samples": len(dataset),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset) if val_dataset is not None else 0,
        "input_channels": input_channels,
        "conditioning_mode": args.conditioning_mode,
        "conditioning_source": "dense_parser",
        "completion_model": args.completion_model,
        "model_config": model.checkpoint_config() if hasattr(model, "checkpoint_config") else None,
        "preserve_known": args.preserve_known,
        "parser_checkpoint": args.parser_checkpoint,
        "parser_splat_fg_threshold": args.parser_splat_fg_threshold,
        "parser_semantic_gate": args.parser_semantic_gate,
        "parser_affine_refine": args.parser_affine_refine,
        "parser_affine_refine_translation_px": args.parser_affine_refine_translation_px,
        "parser_affine_refine_scale": args.parser_affine_refine_scale,
        "parser_checkpoint_views": parse_views(parser_checkpoint_args.get("views", "")) if parser_checkpoint_args else None,
        "best_metric": args.best_metric,
        "scheduler": args.scheduler,
        "min_lr": args.min_lr,
        "log_every": args.log_every,
        "prefetch_factor": args.prefetch_factor,
        "matmul_precision": args.matmul_precision,
        "cudnn_benchmark": args.cudnn_benchmark,
        "augment": args.augment,
        "augment_validation": args.augment_validation,
        "parameters": count_parameters(model),
        "views": dataset.views,
        "device": str(device),
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump({"args": vars(args), "metadata": metadata}, handle, indent=2)
    print(json.dumps(metadata, indent=2))

    views = parse_views(args.views)

    # Create augmenter for training (None if augmentation disabled)
    augmenter = None
    if args.augment:
        augmenter = RenderAugmenter(
            translation_scale=args.translation_scale,
            scale_range=args.scale_range,
            perspective_scale=args.perspective_scale,
            bg_color=dataset.bg_color,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model, criterion, train_loader, optimizer, scaler, device, args.mixed_precision,
            train=True, d_optimizer=d_optimizer, views=views, augmenter=augmenter,
            parser_background_augment=args.parser_background_augment,
            parser_background_augment_prob=args.parser_background_augment_prob,
            dense_parser=dense_parser, parser_splat_fg_threshold=args.parser_splat_fg_threshold,
            parser_semantic_gate=args.parser_semantic_gate,
            parser_affine_refine=args.parser_affine_refine,
            parser_affine_refine_translation_px=args.parser_affine_refine_translation_px,
            parser_affine_refine_scale=args.parser_affine_refine_scale,
            parser_route_confidence_threshold=args.parser_route_confidence_threshold,
            parser_route_margin_threshold=args.parser_route_margin_threshold,
            parser_outer_route_confidence_threshold=args.parser_outer_route_confidence_threshold,
            parser_outer_route_margin_threshold=args.parser_outer_route_margin_threshold,
            parser_outer_uv_min_coverage=args.parser_outer_uv_min_coverage,
            parser_outer_uv_min_source_pixels=args.parser_outer_uv_min_source_pixels,
            parser_outer_geometry_rescue=args.parser_outer_geometry_rescue,
            parser_outer_rescue_confidence_threshold=args.parser_outer_rescue_confidence_threshold,
            parser_outer_rescue_margin_threshold=args.parser_outer_rescue_margin_threshold,
            parser_outer_rescue_min_coverage=args.parser_outer_rescue_min_coverage,
            parser_splat_color_aggregation=args.parser_splat_color_aggregation,
            parser_geometry_route_texel_consensus=args.parser_geometry_route_texel_consensus,
            parser_reject_semantic_fallback=not args.parser_allow_semantic_fallback,
            bg_color=dataset.bg_color, log_every=args.log_every,
            topology_drop_known_min=args.topology_drop_known_min,
            topology_drop_known_max=args.topology_drop_known_max,
            topology_teacher_reveal_unknown=args.topology_teacher_reveal_unknown,
            lambda_rgb_token=args.lambda_rgb_token,
            lambda_rgb_distribution=args.lambda_rgb_distribution,
            lambda_alpha_token=args.lambda_alpha_token,
            ignore_covered_inner=not args.supervise_covered_inner,
            covered_inner_alpha_threshold=args.covered_inner_alpha_threshold,
        )
        metrics = {"train": train_metrics}
        if val_loader is not None:
            with torch.no_grad():
                val_augmenter = None
                if args.augment_validation:
                    val_generator = torch.Generator(device=device)
                    val_generator.manual_seed(args.seed + 1009)
                    val_augmenter = RenderAugmenter(
                        translation_scale=args.translation_scale,
                        scale_range=args.scale_range,
                        perspective_scale=args.perspective_scale,
                        bg_color=dataset.bg_color,
                        generator=val_generator,
                    )
                val_metrics = run_epoch(
                    model, criterion, val_loader, optimizer, scaler, device, args.mixed_precision,
                    train=False, d_optimizer=None, views=views, augmenter=val_augmenter,
                    dense_parser=dense_parser, parser_splat_fg_threshold=args.parser_splat_fg_threshold,
                    parser_semantic_gate=args.parser_semantic_gate,
                    parser_affine_refine=args.parser_affine_refine,
                    parser_affine_refine_translation_px=args.parser_affine_refine_translation_px,
                    parser_affine_refine_scale=args.parser_affine_refine_scale,
                    parser_route_confidence_threshold=args.parser_route_confidence_threshold,
                    parser_route_margin_threshold=args.parser_route_margin_threshold,
                    parser_outer_route_confidence_threshold=args.parser_outer_route_confidence_threshold,
                    parser_outer_route_margin_threshold=args.parser_outer_route_margin_threshold,
                    parser_outer_uv_min_coverage=args.parser_outer_uv_min_coverage,
                    parser_outer_uv_min_source_pixels=args.parser_outer_uv_min_source_pixels,
                    parser_outer_geometry_rescue=args.parser_outer_geometry_rescue,
                    parser_outer_rescue_confidence_threshold=args.parser_outer_rescue_confidence_threshold,
                    parser_outer_rescue_margin_threshold=args.parser_outer_rescue_margin_threshold,
                    parser_outer_rescue_min_coverage=args.parser_outer_rescue_min_coverage,
                    parser_splat_color_aggregation=args.parser_splat_color_aggregation,
                    parser_geometry_route_texel_consensus=args.parser_geometry_route_texel_consensus,
                    parser_reject_semantic_fallback=not args.parser_allow_semantic_fallback,
                    bg_color=dataset.bg_color, log_every=args.log_every,
                    topology_drop_known_min=args.topology_drop_known_min,
                    topology_drop_known_max=args.topology_drop_known_max,
                    topology_teacher_reveal_unknown=args.topology_teacher_reveal_unknown,
                    lambda_rgb_token=args.lambda_rgb_token,
                    lambda_rgb_distribution=args.lambda_rgb_distribution,
                    lambda_alpha_token=args.lambda_alpha_token,
                    ignore_covered_inner=not args.supervise_covered_inner,
                    covered_inner_alpha_threshold=args.covered_inner_alpha_threshold,
                )
            metrics["val"] = val_metrics

        metric_source = metrics.get("val", metrics["train"])
        if args.best_metric not in metric_source:
            raise KeyError(
                f"best_metric={args.best_metric!r} was not logged. "
                f"Available metrics: {', '.join(sorted(metric_source))}"
            )
        metric = metric_source[args.best_metric]
        if scheduler is not None:
            scheduler.step()
        if d_scheduler is not None:
            d_scheduler.step()
        metrics["lr"] = current_lr(optimizer)
        if d_optimizer is not None:
            metrics["d_lr"] = current_lr(d_optimizer)
        print(f"epoch={epoch} metrics={json.dumps(metrics, sort_keys=True)}")

        if epoch % args.preview_every == 0:
            model.eval()
            preview_batch = move_batch(next(iter(val_loader or train_loader)), device)
            with torch.no_grad():
                preview_cond = build_training_conditioning(
                    preview_batch["uv"],
                    criterion.renderer,
                    views,
                    dense_parser=dense_parser,
                    augmenter=None,
                    parser_splat_fg_threshold=args.parser_splat_fg_threshold,
                    parser_semantic_gate=args.parser_semantic_gate,
                    parser_affine_refine=args.parser_affine_refine,
                    parser_affine_refine_translation_px=args.parser_affine_refine_translation_px,
                    parser_affine_refine_scale=args.parser_affine_refine_scale,
                    parser_route_confidence_threshold=args.parser_route_confidence_threshold,
                    parser_route_margin_threshold=args.parser_route_margin_threshold,
                    parser_outer_route_confidence_threshold=args.parser_outer_route_confidence_threshold,
                    parser_outer_route_margin_threshold=args.parser_outer_route_margin_threshold,
                    parser_outer_uv_min_coverage=args.parser_outer_uv_min_coverage,
                    parser_outer_uv_min_source_pixels=args.parser_outer_uv_min_source_pixels,
                    parser_outer_geometry_rescue=args.parser_outer_geometry_rescue,
                    parser_outer_rescue_confidence_threshold=args.parser_outer_rescue_confidence_threshold,
                    parser_outer_rescue_margin_threshold=args.parser_outer_rescue_margin_threshold,
                    parser_outer_rescue_min_coverage=args.parser_outer_rescue_min_coverage,
                    parser_splat_color_aggregation=args.parser_splat_color_aggregation,
                    parser_geometry_route_texel_consensus=args.parser_geometry_route_texel_consensus,
                    parser_reject_semantic_fallback=not args.parser_allow_semantic_fallback,
                    bg_color=dataset.bg_color,
                    confidence_aware_conditioning=(
                        getattr(model, "input_channels", 10) == 12
                    ),
                )
                if hasattr(model, "generate"):
                    pred_uv = model.generate(
                        preview_cond,
                        steps=args.preview_generation_steps,
                        temperature=args.preview_generation_temperature,
                        seed=args.seed,
                        palette_snap=args.preview_palette_snap,
                        palette_min_confidence=args.preview_palette_min_confidence,
                        rgb_decode=args.preview_rgb_decode,
                    )
                else:
                    pred_uv = model(preview_cond)
            save_preview(pred_uv, preview_batch["uv"], output_dir / "previews" / f"epoch_{epoch:04d}.png")

        is_best = metric < best_metric
        if is_best:
            best_metric = metric

        if epoch % args.save_every == 0:
            save_checkpoint(output_dir / "latest.pt", model, optimizer, epoch, args, input_channels, metrics,
                            discriminator=discriminator, d_optimizer=d_optimizer, scheduler=scheduler, d_scheduler=d_scheduler,
                            best_metric=best_metric)
        if is_best:
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, args, input_channels, metrics,
                            discriminator=discriminator, d_optimizer=d_optimizer, scheduler=scheduler, d_scheduler=d_scheduler,
                            best_metric=best_metric)


if __name__ == "__main__":
    main()
