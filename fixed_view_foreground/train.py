import argparse
import json
import math
import os
import random
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

from SkingToolkit.dense_uv_parser.utils import (  # noqa: E402
    build_dense_parser_batch,
    parse_views,
)
from SkingToolkit.fixed_view_foreground.augmentation import (  # noqa: E402
    composite_random_background,
)
from SkingToolkit.fixed_view_foreground.model import (  # noqa: E402
    FixedViewForegroundNet,
    count_parameters,
)
from SkingToolkit.renderer import DifferentiableRenderer  # noqa: E402
from SkingToolkit.semantic_uv_reconstruction.dataset import (  # noqa: E402
    UVInpaintingDataset,
)
from SkingToolkit.semantic_uv_reconstruction.train import get_device  # noqa: E402

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
    if device.type == "cuda" and precision == "fp16":
        return torch.amp.GradScaler("cuda")
    return None


def learning_rate_for_epoch(base_lr, epoch, epochs, min_lr_ratio):
    progress = min(max((epoch - 1) / max(epochs - 1, 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def build_training_batch(uv, renderer, views, args):
    images = []
    targets = []
    backgrounds = []
    parts = []
    for view in views:
        with torch.no_grad():
            rendered, dense_targets = build_dense_parser_batch(
                uv,
                renderer,
                view,
                alpha_threshold=args.target_alpha_threshold,
            )
            foreground = dense_targets["foreground"].float()
            image, background = composite_random_background(
                rendered,
                foreground,
                source_bg_color=args.bg_color,
            )
        images.append(image)
        targets.append(foreground)
        backgrounds.append(background)
        parts.append(dense_targets["part"])

    def stack_views(values):
        stacked = torch.stack(values, dim=1)
        return stacked.reshape(-1, *stacked.shape[2:])

    batch = uv.shape[0]
    view_ids = torch.arange(len(views), device=uv.device).view(1, -1)
    view_ids = view_ids.expand(batch, -1).reshape(-1)
    return (
        stack_views(images),
        stack_views(targets),
        stack_views(backgrounds),
        stack_views(parts),
        view_ids,
    )


def foreground_loss(logits, target, images, backgrounds, parts, args):
    target = target.float()
    probability = torch.sigmoid(logits.float())
    boundary = (
        F.max_pool2d(target, 3, stride=1, padding=1)
        + F.max_pool2d(-target, 3, stride=1, padding=1)
    ).clamp(0.0, 1.0)
    low_contrast = (
        (images.float() - backgrounds.float()).abs().amax(dim=1, keepdim=True)
        <= args.low_contrast_threshold
    ) & (target > 0.5)
    arms = (((parts == 2) | (parts == 3)).unsqueeze(1)) & (target > 0.5)

    positive = target.sum().clamp_min(1.0)
    negative = (target.numel() - target.sum()).clamp_min(1.0)
    positive_weight = (negative / positive).clamp(1.0, args.positive_weight_max)
    pixel_weight = torch.ones_like(target)
    pixel_weight = torch.where(target > 0.5, pixel_weight * positive_weight, pixel_weight)
    pixel_weight = pixel_weight + args.boundary_weight * boundary
    pixel_weight = pixel_weight + args.low_contrast_weight * low_contrast.float()
    pixel_weight = pixel_weight + args.arm_weight * arms.float()

    bce_map = F.binary_cross_entropy_with_logits(
        logits.float(), target, reduction="none"
    )
    loss_bce = (bce_map * pixel_weight).sum() / pixel_weight.sum().clamp_min(1.0)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    loss_dice = (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    loss_boundary = (bce_map * boundary).sum() / boundary.sum().clamp_min(1.0)
    total = loss_bce + args.lambda_dice * loss_dice + args.lambda_boundary * loss_boundary
    return total, {
        "loss_bce": loss_bce,
        "loss_dice": loss_dice,
        "loss_boundary": loss_boundary,
        "low_contrast_pixels": low_contrast.sum(),
        "arm_pixels": arms.sum(),
    }


def metric_counts(probability, target, parts, threshold):
    predicted = probability >= threshold
    target = target > 0.5
    true_positive = (predicted & target).sum()
    false_positive = (predicted & ~target).sum()
    false_negative = (~predicted & target).sum()
    arm_target = ((parts == 2) | (parts == 3)).unsqueeze(1) & target
    arm_true_positive = (predicted & arm_target).sum()
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "arm_true_positive": arm_true_positive,
        "arm_target": arm_target.sum(),
    }


def save_preview(path, images, probability, target, view_count, threshold, bg_color):
    count = min(images.shape[0], view_count * 2)
    probability_rgb = probability[:count].expand(-1, 3, -1, -1)
    target_rgb = target[:count].expand(-1, 3, -1, -1)
    prediction_rgb = (probability[:count] >= threshold).float().expand(-1, 3, -1, -1)
    bg = images.new_tensor(bg_color).view(1, 3, 1, 1) / 255.0
    cutout = torch.where(prediction_rgb > 0.5, images[:count], bg)
    preview = torch.cat(
        [images[:count], probability_rgb, prediction_rgb, target_rgb, cutout], dim=0
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(preview.detach().cpu().clamp(0.0, 1.0), path, nrow=view_count)


def run_epoch(
    model,
    renderer,
    loader,
    optimizer,
    scaler,
    device,
    views,
    args,
    train,
):
    model.train(train)
    sums = {
        "loss": 0.0,
        "loss_bce": 0.0,
        "loss_dice": 0.0,
        "loss_boundary": 0.0,
        "true_positive": 0.0,
        "false_positive": 0.0,
        "false_negative": 0.0,
        "arm_true_positive": 0.0,
        "arm_target": 0.0,
    }
    sample_count = 0
    preview = None
    iterator = tqdm(loader, leave=False) if tqdm is not None else loader
    for batch in iterator:
        uv = batch["uv"].to(device, non_blocking=True)
        images, target, backgrounds, parts, view_ids = build_training_batch(
            uv, renderer, views, args
        )
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train), autocast_context(device, args.mixed_precision):
            logits = model(images, view_ids=view_ids)
            loss, details = foreground_loss(
                logits, target, images, backgrounds, parts, args
            )
        if train:
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

        probability = torch.sigmoid(logits.detach().float())
        counts = metric_counts(probability, target, parts, args.threshold)
        current = images.shape[0]
        sample_count += current
        sums["loss"] += float(loss.detach().item()) * current
        for name in ("loss_bce", "loss_dice", "loss_boundary"):
            sums[name] += float(details[name].detach().item()) * current
        for name, value in counts.items():
            sums[name] += float(value.detach().item())
        if preview is None:
            preview = (
                images.detach(),
                probability.detach(),
                target.detach(),
            )

    if sample_count == 0:
        raise ValueError("Foreground training loader produced no samples.")
    true_positive = sums["true_positive"]
    false_positive = sums["false_positive"]
    false_negative = sums["false_negative"]
    metrics = {
        "loss": sums["loss"] / sample_count,
        "loss_bce": sums["loss_bce"] / sample_count,
        "loss_dice": sums["loss_dice"] / sample_count,
        "loss_boundary": sums["loss_boundary"] / sample_count,
        "precision": true_positive / max(true_positive + false_positive, 1.0),
        "recall": true_positive / max(true_positive + false_negative, 1.0),
        "iou": true_positive
        / max(true_positive + false_positive + false_negative, 1.0),
        "arm_recall": sums["arm_true_positive"] / max(sums["arm_target"], 1.0),
    }
    metrics["selection"] = (
        1.0 - metrics["iou"]
        + args.arm_selection_weight * (1.0 - metrics["arm_recall"])
    )
    return metrics, preview


def save_checkpoint(path, model, optimizer, scaler, epoch, args, metrics, best_metric):
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "args": vars(args),
        "metrics": metrics,
        "best_metric": best_metric,
        "model_config": {
            "input_channels": model.input_channels,
            "base_channels": model.base_channels,
            "view_classes": model.view_classes,
            "coordinate_channels": model.coordinate_channels,
            "dropout": model.dropout_probability,
        },
    }
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Train a fixed-view Minecraft character foreground segmenter."
    )
    parser.add_argument("--data_dir", default="../skins")
    parser.add_argument("--output_dir", default="runs/fixed_view_foreground")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--mappings_dir", default=None)
    parser.add_argument(
        "--views",
        default="walk_front_both_layer_ortho,walk_back_both_layer_ortho",
    )
    parser.add_argument("--max_samples", type=int, default=30000)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--base_channels", type=int, default=24)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16"
    )
    parser.add_argument("--target_alpha_threshold", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--positive_weight_max", type=float, default=4.0)
    parser.add_argument("--boundary_weight", type=float, default=1.0)
    parser.add_argument("--low_contrast_threshold", type=float, default=0.25)
    parser.add_argument("--low_contrast_weight", type=float, default=2.0)
    parser.add_argument("--arm_weight", type=float, default=2.0)
    parser.add_argument("--lambda_dice", type=float, default=1.0)
    parser.add_argument("--lambda_boundary", type=float, default=0.5)
    parser.add_argument("--arm_selection_weight", type=float, default=0.5)
    parser.add_argument("--bg_color", nargs=3, type=int, default=(128, 128, 128))
    return parser


def main():
    args = build_arg_parser().parse_args()
    if not 0.0 <= args.val_split < 1.0:
        raise ValueError("--val_split must be in [0, 1).")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1].")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch_size must be positive.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "previews").mkdir(exist_ok=True)
    views = parse_views(args.views)
    dataset = UVInpaintingDataset(
        data_dir=args.data_dir,
        mappings_dir=args.mappings_dir,
        views=views,
        max_samples=args.max_samples,
    )
    val_count = int(len(dataset) * args.val_split)
    train_count = len(dataset) - val_count
    split_generator = torch.Generator().manual_seed(args.seed)
    if val_count:
        train_dataset, val_dataset = random_split(
            dataset, [train_count, val_count], generator=split_generator
        )
    else:
        train_dataset, val_dataset = dataset, None
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs.update(
            persistent_workers=True, prefetch_factor=args.prefetch_factor
        )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = (
        DataLoader(val_dataset, shuffle=False, **loader_kwargs)
        if val_dataset is not None
        else None
    )
    renderer = DifferentiableRenderer(
        mappings_dir=args.mappings_dir,
        bg_color=tuple(channel / 255.0 for channel in args.bg_color),
    ).to(device)
    missing_views = [view for view in views if view not in renderer.views]
    if missing_views:
        raise ValueError(
            f"Unknown renderer views {missing_views}. Available: {renderer.views}"
        )
    model = FixedViewForegroundNet(
        base_channels=args.base_channels,
        view_classes=len(views),
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = build_grad_scaler(device, args.mixed_precision)

    start_epoch = 1
    best_metric = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        expected_views = checkpoint.get("args", {}).get("views")
        if expected_views and parse_views(expected_views) != views:
            raise ValueError(
                f"Resume checkpoint views {expected_views!r} do not match {views}."
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_metric = float(checkpoint.get("best_metric", float("inf")))

    metadata = {
        "samples": len(dataset),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset) if val_dataset is not None else 0,
        "views": views,
        "parameters": count_parameters(model),
        "device": str(device),
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump({"args": vars(args), "metadata": metadata}, handle, indent=2)
    print(json.dumps(metadata, indent=2))

    for epoch in range(start_epoch, args.epochs + 1):
        lr = learning_rate_for_epoch(
            args.lr, epoch, args.epochs, args.min_lr_ratio
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        train_metrics, train_preview = run_epoch(
            model,
            renderer,
            train_loader,
            optimizer,
            scaler,
            device,
            views,
            args,
            train=True,
        )
        metrics = {"epoch": epoch, "lr": lr, "train": train_metrics}
        selection_metrics = train_metrics
        preview = train_preview
        if val_loader is not None:
            with torch.no_grad():
                val_metrics, val_preview = run_epoch(
                    model,
                    renderer,
                    val_loader,
                    optimizer,
                    scaler,
                    device,
                    views,
                    args,
                    train=False,
                )
            metrics["val"] = val_metrics
            selection_metrics = val_metrics
            preview = val_preview
        current_metric = selection_metrics["selection"]
        if preview is not None:
            save_preview(
                output_dir / "previews" / f"epoch_{epoch:04d}.png",
                *preview,
                view_count=len(views),
                threshold=args.threshold,
                bg_color=args.bg_color,
            )
        save_checkpoint(
            output_dir / "latest.pt",
            model,
            optimizer,
            scaler,
            epoch,
            args,
            metrics,
            min(best_metric, current_metric),
        )
        if current_metric < best_metric:
            best_metric = current_metric
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scaler,
                epoch,
                args,
                metrics,
                best_metric,
            )
        with open(output_dir / "metrics.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
