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

from SkingToolkit.renderer import DifferentiableRenderer  # noqa: E402
from SkingToolkit.semantic_uv_reconstruction.dataset import finalize_minecraft_alpha, parse_views  # noqa: E402
from SkingToolkit.semantic_uv_reconstruction.semantic_dataset import SemanticUVPairDataset  # noqa: E402
from SkingToolkit.semantic_uv_reconstruction.semantic_losses import SemanticUVReconstructionLoss  # noqa: E402
from SkingToolkit.semantic_uv_reconstruction.semantic_model import SemanticUVReconstructor, count_parameters  # noqa: E402
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
    if precision != "fp16" or device.type != "cuda":
        return None
    return torch.amp.GradScaler("cuda")


def learning_rate_for_epoch(base_lr, epoch, epochs, min_lr_ratio=0.05):
    if epochs <= 1:
        return float(base_lr)
    progress = min(max((epoch - 1) / max(epochs - 1, 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(base_lr) * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def set_optimizer_lr(optimizer, learning_rate):
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def move_batch(batch, device):
    moved = {
        "uv": batch["uv"].to(device, non_blocking=True),
        "path": batch["path"],
    }
    for name in ("semantic_uv",):
        if name in batch:
            moved[name] = batch[name].to(device, non_blocking=True)
    return moved


def render_fixed_views(uv, renderer, views):
    """Render the configured clean views without any random perturbation."""
    with torch.no_grad():
        return torch.stack([renderer.forward_view(uv, view) for view in views], dim=1)


def format_metrics(metric_sums, sample_count):
    result = {}
    for name, value in metric_sums.items():
        denominator = 1 if name.startswith("count_") else max(sample_count, 1)
        averaged = value / denominator
        result[name] = float(averaged.detach().cpu()) if torch.is_tensor(averaged) else averaged
    if all(name in result for name in ("count_outer_tp", "count_outer_fp", "count_outer_fn")):
        true_positive = result["count_outer_tp"]
        false_positive = result["count_outer_fp"]
        false_negative = result["count_outer_fn"]
        result["precision_outer"] = true_positive / max(true_positive + false_positive, 1.0)
        result["recall_outer"] = true_positive / max(true_positive + false_negative, 1.0)
        result["iou_outer"] = true_positive / max(
            true_positive + false_positive + false_negative, 1.0
        )
    return result


def run_epoch(
    model,
    criterion,
    loader,
    renderer,
    views,
    device,
    precision,
    optimizer=None,
    scaler=None,
    grad_clip=1.0,
    description="train",
):
    training = optimizer is not None
    model.train(training)
    metric_sums = {}
    sample_count = 0
    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc=description, leave=False)

    grad_context = torch.enable_grad if training else torch.no_grad
    with grad_context():
        for batch in iterator:
            batch = move_batch(batch, device)
            target_uv = batch["uv"]
            gt_renders = render_fixed_views(target_uv, renderer, views)
            images = gt_renders[:, :, :3]
            if training:
                optimizer.zero_grad(set_to_none=True)

            with autocast_context(device, precision):
                outputs = model(images)
                metrics = criterion(
                    outputs,
                    target_uv,
                    gt_renders=gt_renders,
                    renderer=renderer,
                    views=views,
                    semantic_uv_target=batch.get("semantic_uv"),
                    semantic_encoder=(
                        model.encode_open_semantics if model.has_open_semantics else None
                    ),
                )
                loss = metrics["loss_total"]

            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

            batch_size = target_uv.shape[0]
            sample_count += batch_size
            for name, value in metrics.items():
                detached = value.detach()
                contribution = detached if name.startswith("count_") else detached * batch_size
                metric_sums[name] = metric_sums.get(name, 0.0) + contribution
            if tqdm is not None:
                iterator.set_postfix(
                    loss=f"{float(loss.detach().cpu()):.4f}",
                    outer=f"{float(metrics['loss_outer_alpha'].detach().cpu()):.3f}",
                )
    return format_metrics(metric_sums, sample_count)


@torch.no_grad()
def save_preview(model, loader, renderer, views, device, output_path, max_items=4):
    model.eval()
    batch = move_batch(next(iter(loader)), device)
    target_uv = batch["uv"]
    renders = render_fixed_views(target_uv, renderer, views)
    outputs = model(renders[:, :, :3])
    count = min(max_items, target_uv.shape[0])

    view_rows = []
    for view_index in range(len(views)):
        resized = F.interpolate(
            renders[:count, view_index], size=(64, 64), mode="bilinear", align_corners=False
        )
        view_rows.append(resized)
    pred = finalize_minecraft_alpha(outputs["uv"][:count].detach().cpu()).to(device)
    target = finalize_minecraft_alpha(target_uv[:count].detach().cpu()).to(device)
    pred_outer = outputs["uv"][:count, 3:4] * model.decor_mask
    target_outer = target_uv[:count, 3:4] * model.decor_mask
    pred_outer_rgba = torch.cat([pred_outer.expand(-1, 3, -1, -1), torch.ones_like(pred_outer)], dim=1)
    target_outer_rgba = torch.cat([target_outer.expand(-1, 3, -1, -1), torch.ones_like(target_outer)], dim=1)
    preview = torch.cat([*view_rows, pred, target, pred_outer_rgba, target_outer_rgba], dim=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(preview.clamp(0.0, 1.0).cpu(), output_path, nrow=count)


FROZEN_VISION_PREFIX = "open_semantic_backbone.vision_model."


def checkpoint_model_state(model):
    """Avoid duplicating the frozen Hugging Face tower in every checkpoint."""
    return {
        name: value
        for name, value in model.state_dict().items()
        if not name.startswith(FROZEN_VISION_PREFIX)
    }


def load_checkpoint_model_state(model, state):
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = [
        name for name in incompatible.missing_keys if not name.startswith(FROZEN_VISION_PREFIX)
    ]
    if unexpected or missing:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}"
        )


def save_checkpoint(path, model, optimizer, scaler, epoch, args, metrics):
    checkpoint = {
        "epoch": epoch,
        "model": checkpoint_model_state(model),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "args": vars(args),
        "metrics": metrics,
        "frozen_vision_weights_excluded": model.has_open_semantics,
        "model_config": {
            "view_count": model.view_count,
            "base_channels": model.base_channels,
            "token_channels": model.token_channels,
            "query_size": model.query_size,
            "attention_heads": model.attention_heads,
            "attention_layers": model.attention_layers,
            "attention_dropout": model.attention_dropout,
            "semantic_classes": model.semantic_classes,
            "architecture_version": model.architecture_version,
            "semantic_backbone": model.semantic_backbone_name,
            "siglip_model": model.siglip_model,
            "views": parse_views(args.views),
            "arm_model": "steve",
            "fixed_render_training": True,
        },
    }
    path = Path(path)
    temporary_path = path.with_name(f".{path.name}.tmp")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Train a fixed-view semantic front/back render to UV reconstructor."
    )
    parser.add_argument("--data_dir", default="../skins")
    parser.add_argument("--output_dir", default="runs/semantic_uv_reconstruction")
    parser.add_argument("--mappings_dir", default=None)
    parser.add_argument(
        "--views",
        default="walk_front_both_layer_ortho,walk_back_both_layer_ortho",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--semantic_labels_dir", default=None)
    parser.add_argument("--semantic_classes", type=int, default=13)
    parser.add_argument("--semantic_backbone", choices=["siglip2", "none"], default="siglip2")
    parser.add_argument("--siglip_model", default="google/siglip2-base-patch16-224")
    parser.add_argument("--siglip_local_files_only", action="store_true")
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--token_channels", type=int, default=128)
    parser.add_argument("--query_size", type=int, default=32)
    parser.add_argument("--attention_heads", type=int, default=4)
    parser.add_argument("--attention_layers", type=int, default=2)
    parser.add_argument("--attention_dropout", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--matmul_precision", choices=["highest", "high", "medium"], default="high")
    parser.add_argument("--cudnn_benchmark", dest="cudnn_benchmark", action="store_true", default=True)
    parser.add_argument("--no_cudnn_benchmark", dest="cudnn_benchmark", action="store_false")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--preview_every", type=int, default=1)
    parser.add_argument("--lambda_uv_rgb", type=float, default=2.0)
    parser.add_argument("--lambda_uv_edge", type=float, default=1.0)
    parser.add_argument("--lambda_outer_alpha", type=float, default=1.0)
    parser.add_argument("--lambda_outer_dice", type=float, default=0.5)
    parser.add_argument("--lambda_semantic_uv", type=float, default=0.25)
    parser.add_argument("--lambda_semantic_presence", type=float, default=0.25)
    parser.add_argument("--lambda_semantic_coverage", type=float, default=0.25)
    parser.add_argument("--lambda_semantic_color", type=float, default=0.25)
    parser.add_argument("--lambda_render_rgb", type=float, default=0.5)
    parser.add_argument("--lambda_render_alpha", type=float, default=0.5)
    parser.add_argument("--lambda_siglip_render", type=float, default=0.1)
    return parser


def main():
    args = build_arg_parser().parse_args()
    if not 0.0 <= args.val_split < 1.0:
        raise ValueError("--val_split must be in [0, 1).")
    if args.lr <= 0.0:
        raise ValueError("--lr must be positive.")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("--min_lr_ratio must be in [0, 1].")
    if args.semantic_labels_dir is None and args.semantic_classes != 13:
        raise ValueError("Without --semantic_labels_dir, --semantic_classes must be 13.")
    if args.attention_dropout < 0.0 or args.attention_dropout >= 1.0:
        raise ValueError("--attention_dropout must be in [0, 1).")
    if args.semantic_backbone == "none" and args.lambda_siglip_render > 0.0:
        raise ValueError("--lambda_siglip_render must be 0 when --semantic_backbone=none.")

    device = get_device(args.device)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(args.matmul_precision)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = args.cudnn_benchmark

    views = parse_views(args.views)
    if len(views) < 2:
        raise ValueError("Semantic UV reconstruction expects at least front and back views.")
    if args.mappings_dir is not None and not Path(args.mappings_dir).is_dir():
        raise FileNotFoundError(
            f"Renderer mappings directory does not exist: {args.mappings_dir}. "
            "Set MAPPINGS_DIR to the directory containing <view>_mapping.pt files."
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "previews").mkdir(exist_ok=True)

    dataset = SemanticUVPairDataset(
        args.data_dir,
        max_samples=args.max_samples,
        semantic_labels_dir=args.semantic_labels_dir,
    )
    val_count = int(len(dataset) * args.val_split)
    if args.val_split > 0.0 and val_count == 0:
        val_count = 1
    train_count = len(dataset) - val_count
    if train_count <= 0:
        raise ValueError("Training split is empty.")
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
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs) if val_dataset else None

    renderer = DifferentiableRenderer(mappings_dir=args.mappings_dir).to(device)
    missing_views = [view for view in views if view not in renderer.views]
    if missing_views:
        raise ValueError(
            f"Renderer mappings are missing views {missing_views}. "
            f"Available views in {args.mappings_dir!r}: {renderer.views}. "
            "Regenerate mappings for the configured VIEWS or select a matching MAPPINGS_DIR."
        )
    renderer.eval()
    for parameter in renderer.parameters():
        parameter.requires_grad_(False)

    model = SemanticUVReconstructor(
        view_count=len(views),
        base_channels=args.base_channels,
        token_channels=args.token_channels,
        query_size=args.query_size,
        attention_heads=args.attention_heads,
        attention_layers=args.attention_layers,
        attention_dropout=args.attention_dropout,
        semantic_classes=args.semantic_classes,
        semantic_backbone=args.semantic_backbone,
        siglip_model=args.siglip_model,
        siglip_local_files_only=args.siglip_local_files_only,
    ).to(device)
    criterion = SemanticUVReconstructionLoss(
        lambda_uv_rgb=args.lambda_uv_rgb,
        lambda_uv_edge=args.lambda_uv_edge,
        lambda_outer_alpha=args.lambda_outer_alpha,
        lambda_outer_dice=args.lambda_outer_dice,
        lambda_semantic_uv=args.lambda_semantic_uv,
        lambda_semantic_presence=args.lambda_semantic_presence,
        lambda_semantic_coverage=args.lambda_semantic_coverage,
        lambda_semantic_color=args.lambda_semantic_color,
        lambda_render_rgb=args.lambda_render_rgb,
        lambda_render_alpha=args.lambda_render_alpha,
        lambda_siglip_render=args.lambda_siglip_render,
    ).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = build_grad_scaler(device, args.mixed_precision)

    start_epoch = 1
    best_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        config = checkpoint.get("model_config", {})
        expected = {
            "views": views,
            "semantic_classes": args.semantic_classes,
            "architecture_version": model.architecture_version,
            "query_size": args.query_size,
            "semantic_backbone": args.semantic_backbone,
            "siglip_model": args.siglip_model,
        }
        for name, expected_value in expected.items():
            if config.get(name) != expected_value:
                raise ValueError(
                    f"Resume checkpoint {name}={config.get(name)!r}, expected {expected_value!r}."
                )
        load_checkpoint_model_state(model, checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint.get("metrics", {}).get("best_loss", float("inf")))

    summary = {
        "num_samples": len(dataset),
        "train_samples": train_count,
        "val_samples": val_count,
        "views": views,
        "parameters": count_parameters(model),
        "device": str(device),
        "fixed_render_training": True,
        "render_augmentation": False,
        "architecture_version": model.architecture_version,
        "semantic_classes": args.semantic_classes,
        "semantic_backbone": args.semantic_backbone,
        "siglip_model": args.siglip_model,
        "siglip_frozen": model.has_open_semantics,
        "lambda_siglip_render": args.lambda_siglip_render,
        "lambda_uv_rgb": args.lambda_uv_rgb,
        "lambda_uv_edge": args.lambda_uv_edge,
        "query_size": args.query_size,
        "base_learning_rate": args.lr,
        "min_lr_ratio": args.min_lr_ratio,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    preview_loader = val_loader if val_loader is not None else train_loader
    for epoch in range(start_epoch, args.epochs + 1):
        learning_rate = learning_rate_for_epoch(
            args.lr, epoch, args.epochs, args.min_lr_ratio
        )
        set_optimizer_lr(optimizer, learning_rate)
        train_metrics = run_epoch(
            model,
            criterion,
            train_loader,
            renderer,
            views,
            device,
            args.mixed_precision,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip=args.grad_clip,
            description=f"train {epoch}",
        )
        val_metrics = (
            run_epoch(
                model,
                criterion,
                val_loader,
                renderer,
                views,
                device,
                args.mixed_precision,
                description=f"val {epoch}",
            )
            if val_loader is not None
            else train_metrics
        )
        selected_loss = val_metrics["loss_total"]
        is_best = selected_loss < best_loss
        if is_best:
            best_loss = selected_loss
        epoch_metrics = {
            "train": train_metrics,
            "val": val_metrics,
            "learning_rate": learning_rate,
            "best_loss": best_loss,
        }
        print(f"epoch={epoch} metrics={json.dumps(epoch_metrics, ensure_ascii=False)}")

        save_checkpoint(
            output_dir / "latest.pt",
            model,
            optimizer,
            scaler,
            epoch,
            args,
            epoch_metrics,
        )
        if is_best:
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scaler,
                epoch,
                args,
                epoch_metrics,
            )
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                scaler,
                epoch,
                args,
                epoch_metrics,
            )
        if args.preview_every > 0 and epoch % args.preview_every == 0:
            save_preview(
                model,
                preview_loader,
                renderer,
                views,
                device,
                output_dir / "previews" / f"epoch_{epoch:04d}.png",
            )


if __name__ == "__main__":
    main()
