import argparse
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TOOLKIT_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from SkingToolkit.dense_uv_parser.losses import DenseUVParserLoss  # noqa: E402
from SkingToolkit.dense_uv_parser.model import DenseUVParserNet, count_parameters  # noqa: E402
from SkingToolkit.dense_uv_parser.utils import (  # noqa: E402
    FACE_PALETTE,
    IGNORE_INDEX,
    LAYER_FACE_PALETTE,
    LAYER_PALETTE,
    PART_PALETTE,
    UV_SIZE,
    augment_dense_batch,
    build_dense_parser_batch,
    canonicalize_dense_targets,
    colorize_foreground,
    colorize_labels,
    colorize_surface,
    colorize_uv,
    combine_layer_face,
    flat_uv_to_uv01,
    parse_views,
    prediction_uv01,
    randomize_render_background,
    splat_deterministic_targets_to_uv_conditioning,
    splat_parser_predictions_to_uv_conditioning,
    splat_predictions_to_uv_conditioning,
    splat_targets_to_uv_conditioning,
    surface_class_count,
)
from SkingToolkit.inverse_uv.dataset import InverseUVDataset  # noqa: E402
from SkingToolkit.inverse_uv.train import get_device  # noqa: E402
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


def format_metrics(metric_sums, count):
    return {
        name: float((value / max(count, 1)).detach().cpu()) if torch.is_tensor(value) else value / max(count, 1)
        for name, value in metric_sums.items()
    }


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


def run_epoch(model, criterion, renderer, loader, optimizer, scaler, device, precision, args, train=True):
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

        with torch.set_grad_enabled(train):
            with autocast_context(device, precision):
                outputs = model(rendered, view_ids=view_ids)
                losses = criterion(outputs, targets)
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

        sample_count += parser_samples
        for name, value in losses.items():
            detached = value.detach()
            metric_sums[name] = metric_sums.get(name, detached.new_zeros(())) + detached * parser_samples

        if tqdm is not None and args.log_every > 0 and sample_count % (args.log_every * parser_samples) == 0:
            avg = format_metrics(metric_sums, sample_count)
            postfix = {
                "total": f"{avg['loss_total']:.4f}",
                "fg": f"{avg.get('recall_foreground', avg['acc_foreground']):.3f}",
            }
            if args.parser_mode == "global_affine":
                postfix["align"] = f"{avg.get('err_affine_translation_px', 0.0):.2f}px"
                postfix["scale"] = f"{avg.get('err_affine_scale_pct', 0.0):.2f}%"
                postfix["surface"] = f"{avg.get('acc_surface', 0.0):.3f}"
            else:
                postfix["uv"] = f"{avg.get('loss_uv_l1_px', avg['loss_uv']):.2f}px"
                postfix["uv1"] = f"{avg.get('acc_uv_within1', 0.0):.3f}"
            iterator.set_postfix(
                **postfix,
            )

    return format_metrics(metric_sums, sample_count)





def save_preview(model, renderer, loader, device, args, output_path, max_items=2):
    model.eval()
    views = parse_views(args.views)
    batch = move_batch(next(iter(loader)), device)
    rendered, targets, view_count, view_ids = build_parser_inputs(
        batch["uv"], renderer, views, train=False, args=args, apply_augment=False
    )
    with torch.no_grad():
        outputs = model(rendered, view_ids=view_ids)
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
    pred_part = torch.where(
        pred_fg,
        debug_outputs["part"].argmax(dim=1),
        torch.full_like(debug_outputs["part"].argmax(dim=1), IGNORE_INDEX),
    )
    pred_layer = torch.where(
        pred_fg,
        pred_layer_values,
        torch.full_like(debug_outputs["layer"].argmax(dim=1), IGNORE_INDEX),
    )
    pred_face_values = debug_outputs["face"].argmax(dim=1)
    pred_face = torch.where(
        pred_fg,
        pred_face_values,
        torch.full_like(debug_outputs["face"].argmax(dim=1), IGNORE_INDEX),
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
            "layer_face_classes": 12,
        },
    }
    torch.save(checkpoint, path)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train a dense render-pixel to Minecraft UV parser.")
    parser.add_argument("--data_dir", default="../skins")
    parser.add_argument("--output_dir", default="runs/dense_uv_parser")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume; --epochs remains the final epoch number.")
    parser.add_argument("--mappings_dir", default=None)
    parser.add_argument("--views", default="walk_front_both_layer_ortho,walk_back_both_layer_ortho")
    parser.add_argument(
        "--parser_mode",
        choices=["global_affine", "dense"],
        default="global_affine",
        help="global_affine aligns the render then uses fixed renderer UV mappings; dense preserves the legacy UV head.",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--matmul_precision", choices=["highest", "high", "medium"], default="high")
    parser.add_argument("--cudnn_benchmark", dest="cudnn_benchmark", action="store_true", default=True)
    parser.add_argument("--no_cudnn_benchmark", dest="cudnn_benchmark", action="store_false")
    parser.add_argument("--target_alpha_threshold", type=float, default=0.5)
    parser.add_argument("--splat_fg_threshold", type=float, default=0.5)
    parser.add_argument("--affine_refine", dest="affine_refine", action="store_true", default=True)
    parser.add_argument("--no_affine_refine", dest="affine_refine", action="store_false")
    parser.add_argument("--affine_refine_translation_px", type=float, default=2.0)
    parser.add_argument("--affine_refine_scale", type=float, default=0.0)
    parser.add_argument("--route_confidence_threshold", type=float, default=0.05)
    parser.add_argument("--route_margin_threshold", type=float, default=0.10)
    parser.add_argument("--allow_semantic_fallback", action="store_true")
    parser.add_argument("--augment", dest="augment", action="store_true", default=True)
    parser.add_argument("--no_augment", dest="augment", action="store_false")
    parser.add_argument("--augment_validation", dest="augment_validation", action="store_true", default=True)
    parser.add_argument("--no_augment_validation", dest="augment_validation", action="store_false")
    parser.add_argument("--translation_scale", type=float, default=0.03)
    parser.add_argument("--scale_range", type=float, default=0.03)
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
    parser.add_argument("--uv_classification", dest="uv_classification", action="store_true", default=True)
    parser.add_argument("--no_uv_classification", dest="uv_classification", action="store_false")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--preview_every", type=int, default=1)
    parser.add_argument(
        "--best_metric",
        choices=["loss_total", "loss_routing", "loss_surface", "loss_uv_class"],
        default="loss_total",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    args.bg_color = (128, 128, 128)
    if args.scale_range < 0 or args.scale_range >= 1:
        raise ValueError("--scale_range must be in [0, 1).")
    device = get_device(args.device)
    configure_torch(args, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "previews").mkdir(exist_ok=True)

    dataset = InverseUVDataset(
        data_dir=args.data_dir,
        mappings_dir=args.mappings_dir,
        views=args.views,
        max_samples=args.max_samples,
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

    global_affine_mode = args.parser_mode == "global_affine"
    surface_classes = surface_class_count(renderer, parse_views(args.views)) if global_affine_mode else 0
    model = DenseUVParserNet(
        base_channels=args.base_channels,
        uv_size=UV_SIZE,
        uv_classification=args.uv_classification,
        view_classes=len(parse_views(args.views)),
        predict_affine=global_affine_mode,
        affine_translation_scale=args.translation_scale,
        affine_scale_range=args.scale_range,
        surface_classes=surface_classes,
    ).to(device)
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
        uv_size=UV_SIZE,
        use_uv=(not global_affine_mode) or args.uv_classification,
        affine_translation_limit=model.affine_translation_limit if global_affine_mode else 1.0,
        affine_log_scale_limit=model.affine_log_scale_limit if global_affine_mode else 1.0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = build_grad_scaler(device, args.mixed_precision)

    start_epoch = 1
    best_metric = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        if not any(key.startswith("layer_face.") for key in checkpoint["model"]):
            raise ValueError(
                "This checkpoint predates the joint layer-face head. Start a new run without RESUME "
                "so all 12 inner/outer face classes are trained together."
            )
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
            raise ValueError(
                "Cannot resume with a different best metric: "
                f"checkpoint={checkpoint_best_metric!r}, requested={args.best_metric!r}."
            )
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
        "device": str(device),
        "parser_mode": args.parser_mode,
        "uv_classification": model.uv_classification,
        "view_classes": len(parse_views(args.views)),
        "surface_classes": surface_classes,
        "best_metric": args.best_metric,
        "augment": args.augment,
        "augment_validation": args.augment_validation,
        "background_augment": args.background_augment,
        "background_augment_prob": args.background_augment_prob,
        "semantic_gate": args.semantic_gate,
        "affine_refine": args.affine_refine,
        "affine_refine_translation_px": args.affine_refine_translation_px,
        "affine_refine_scale": args.affine_refine_scale,
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump({"args": vars(args), "metadata": metadata}, handle, indent=2)
    print(json.dumps(metadata, indent=2))

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model, criterion, renderer, train_loader, optimizer, scaler, device, args.mixed_precision, args, train=True
        )
        metrics = {"train": train_metrics}
        metric_source = train_metrics
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(
                    model, criterion, renderer, val_loader, optimizer, scaler, device, args.mixed_precision, args, train=False
                )
            metrics["val"] = val_metrics
            metric_source = val_metrics

        if args.best_metric not in metric_source:
            raise ValueError(f"Best metric {args.best_metric!r} is not available in epoch metrics.")
        metric = metric_source[args.best_metric]
        print(f"epoch={epoch} metrics={json.dumps(metrics, sort_keys=True)}")

        if epoch % args.preview_every == 0:
            save_preview(model, renderer, val_loader or train_loader, device, args, output_dir / "previews" / f"epoch_{epoch:04d}.png")

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


if __name__ == "__main__":
    main()
