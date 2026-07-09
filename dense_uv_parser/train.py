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
    augment_dense_batch,
    build_dense_parser_batch,
    parse_views,
    splat_predictions_to_uv_conditioning,
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
        else:
            raise ValueError(f"Unexpected target shape for {key}: {tuple(stacked.shape)}")
    return result


def build_parser_inputs(batch_uv, renderer, views, train, args):
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
            if train and args.augment:
                rendered, targets = augment_dense_batch(
                    rendered,
                    targets,
                    translation_scale=args.translation_scale,
                    scale_range=args.scale_range,
                    bg_color=args.bg_color,
                )
            rendered_by_view.append(rendered)
            targets_by_view.append(targets)

    rendered = torch.stack(rendered_by_view, dim=1)
    B, V, C, H, W = rendered.shape
    rendered = rendered.reshape(B * V, C, H, W)
    targets = stack_view_targets(targets_by_view)
    return rendered, targets, V


def run_epoch(model, criterion, renderer, loader, optimizer, scaler, device, precision, args, train=True):
    model.train(train)
    views = parse_views(args.views)
    metric_sums = {}
    sample_count = 0
    iterator = tqdm(loader, leave=False, file=sys.__stderr__ or sys.stderr) if tqdm is not None else loader

    for batch in iterator:
        batch = move_batch(batch, device)
        rendered, targets, _ = build_parser_inputs(batch["uv"], renderer, views, train=train, args=args)
        parser_samples = rendered.shape[0]

        with torch.set_grad_enabled(train):
            with autocast_context(device, precision):
                outputs = model(rendered)
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
            iterator.set_postfix(
                total=f"{avg['loss_total']:.4f}",
                uv=f"{avg['loss_uv']:.4f}",
                fg=f"{avg['acc_foreground']:.3f}",
            )

    return format_metrics(metric_sums, sample_count)


def save_preview(model, renderer, loader, device, args, output_path, max_items=2):
    model.eval()
    views = parse_views(args.views)
    batch = move_batch(next(iter(loader)), device)
    rendered, _, view_count = build_parser_inputs(batch["uv"], renderer, views, train=False, args=args)
    with torch.no_grad():
        outputs = model(rendered)
        conditioning = splat_predictions_to_uv_conditioning(
            rendered,
            outputs,
            group_size=view_count,
            fg_threshold=args.splat_fg_threshold,
            bg_color=args.bg_color,
        )

    count = min(max_items, conditioning.shape[0])
    conditioning = conditioning[:count].detach().cpu()
    # Show inner/outer RGB layers from the 10-channel conditioning.
    inner_rgb = conditioning[:, 0:3]
    outer_rgb = conditioning[:, 5:8]
    preview = torch.cat([inner_rgb, outer_rgb], dim=0)
    save_image(preview.clamp(0.0, 1.0), output_path, nrow=count)


def save_checkpoint(path, model, optimizer, epoch, args, metrics, best_metric=None):
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "metrics": metrics,
        "best_metric": best_metric,
        "model_config": {
            "input_channels": 4,
            "base_channels": args.base_channels,
        },
    }
    torch.save(checkpoint, path)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train a dense render-pixel to Minecraft UV parser.")
    parser.add_argument("--data_dir", default="../skins")
    parser.add_argument("--output_dir", default="runs/dense_uv_parser")
    parser.add_argument("--mappings_dir", default=None)
    parser.add_argument("--views", default="walk_front_both_layer_ortho,walk_back_both_layer_ortho")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--matmul_precision", choices=["highest", "high", "medium"], default="high")
    parser.add_argument("--cudnn_benchmark", dest="cudnn_benchmark", action="store_true", default=True)
    parser.add_argument("--no_cudnn_benchmark", dest="cudnn_benchmark", action="store_false")
    parser.add_argument("--target_alpha_threshold", type=float, default=0.5)
    parser.add_argument("--splat_fg_threshold", type=float, default=0.5)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--translation_scale", type=float, default=0.035)
    parser.add_argument("--scale_range", type=float, default=0.035)
    parser.add_argument("--lambda_foreground", type=float, default=1.0)
    parser.add_argument("--lambda_layer", type=float, default=1.0)
    parser.add_argument("--lambda_part", type=float, default=0.5)
    parser.add_argument("--lambda_face", type=float, default=0.5)
    parser.add_argument("--lambda_uv", type=float, default=5.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--preview_every", type=int, default=1)
    return parser


def main():
    args = build_arg_parser().parse_args()
    args.bg_color = (128, 128, 128)
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

    model = DenseUVParserNet(base_channels=args.base_channels).to(device)
    criterion = DenseUVParserLoss(
        lambda_foreground=args.lambda_foreground,
        lambda_layer=args.lambda_layer,
        lambda_part=args.lambda_part,
        lambda_face=args.lambda_face,
        lambda_uv=args.lambda_uv,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = build_grad_scaler(device, args.mixed_precision)

    metadata = {
        "num_samples": len(dataset),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset) if val_dataset is not None else 0,
        "views": parse_views(args.views),
        "parameters": count_parameters(model),
        "device": str(device),
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump({"args": vars(args), "metadata": metadata}, handle, indent=2)
    print(json.dumps(metadata, indent=2))

    best_metric = float("inf")
    for epoch in range(1, args.epochs + 1):
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

        metric = metric_source["loss_total"]
        print(f"epoch={epoch} metrics={json.dumps(metrics, sort_keys=True)}")

        if epoch % args.preview_every == 0:
            save_preview(model, renderer, val_loader or train_loader, device, args, output_dir / "previews" / f"epoch_{epoch:04d}.png")

        is_best = metric < best_metric
        if is_best:
            best_metric = metric
        if epoch % args.save_every == 0:
            save_checkpoint(output_dir / "latest.pt", model, optimizer, epoch, args, metrics, best_metric=best_metric)
        if is_best:
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, args, metrics, best_metric=best_metric)


if __name__ == "__main__":
    main()

