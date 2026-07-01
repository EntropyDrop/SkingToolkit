import argparse
import copy
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

from SkingToolkit.inverse_uv.dataset import InverseUVDataset, apply_uv_mask, RenderAugmenter, build_conditioning  # noqa: E402
from SkingToolkit.inverse_uv.losses import InverseUVLoss  # noqa: E402
from SkingToolkit.inverse_uv.model import InverseUVNet, count_parameters  # noqa: E402

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


def move_batch(batch, device):
    res = {
        "uv": batch["uv"].to(device, non_blocking=True),
        "path": batch["path"],
    }
    if "conditioning" in batch:
        res["conditioning"] = batch["conditioning"].to(device, non_blocking=True)
    return res


def save_preview(pred_uv, gt_uv, output_path, max_items=4):
    count = min(max_items, pred_uv.shape[0])
    pred = apply_uv_mask(pred_uv[:count].detach().cpu())
    gt = apply_uv_mask(gt_uv[:count].detach().cpu())
    preview = torch.cat([pred, gt], dim=0)
    save_image(preview.clamp(0.0, 1.0), output_path, nrow=count)


def format_losses(loss_sums, count):
    return {name: value / max(count, 1) for name, value in loss_sums.items()}


def run_epoch(model, criterion, loader, optimizer, scaler, device, precision, train=True, views=None, augmenter=None):
    model.train(train)
    loss_sums = {}
    sample_count = 0
    iterator = tqdm(loader, leave=False) if tqdm is not None else loader

    for batch in iterator:
        batch = move_batch(batch, device)
        batch_size = batch["uv"].shape[0]
        
        # Build conditioning on GPU
        with torch.no_grad():
            batch_augmenter = augmenter if train else None
            batch["conditioning"] = build_conditioning(
                batch["uv"],
                criterion.renderer,
                views,
                augmenter=batch_augmenter,
            )

        with torch.set_grad_enabled(train):
            with autocast_context(device, precision):
                pred_uv = model(batch["conditioning"])
                losses = criterion(pred_uv, batch["uv"])
                loss = losses["loss_total"]

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        sample_count += batch_size
        for name, value in losses.items():
            loss_sums[name] = loss_sums.get(name, 0.0) + float(value.detach().cpu()) * batch_size
        if tqdm is not None:
            avg = format_losses(loss_sums, sample_count)
            iterator.set_postfix(total=f"{avg['loss_total']:.4f}", rgb=f"{avg['loss_rgb']:.4f}")

    return format_losses(loss_sums, sample_count)


def save_checkpoint(path, model, optimizer, epoch, args, input_channels, metrics):
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "input_channels": input_channels,
        "model_config": {
            "input_channels": input_channels,
            "base_channels": args.base_channels,
            "use_coordconv": args.coordconv,
            "use_attention": args.bottleneck_attention,
            "attention_heads": args.attention_heads,
        },
        "metrics": metrics,
    }
    torch.save(checkpoint, path)


def build_grad_scaler(device, precision):
    enabled = device.type == "cuda" and precision == "fp16"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train render-to-UV inverse Minecraft skin model.")
    parser.add_argument("--data_dir", required=True, help="Folder containing GT 64x64 RGBA skin PNGs.")
    parser.add_argument("--output_dir", default="inverse_uv_runs/default", help="Checkpoint/output folder.")
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
    parser.add_argument("--base_channels", type=int, default=64, help="Base channel width for InverseUVNet.")
    parser.add_argument(
        "--coordconv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append normalized x/y coordinate channels inside InverseUVNet.",
    )
    parser.add_argument(
        "--bottleneck_attention",
        "--bottleneck-attention",
        dest="bottleneck_attention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a lightweight spatial self-attention block at the U-Net bottleneck.",
    )
    parser.add_argument(
        "--attention_heads",
        "--attention-heads",
        dest="attention_heads",
        type=int,
        default=4,
        help="Attention heads for bottleneck self-attention.",
    )
    parser.add_argument("--augment", action="store_true", help="Enable online data augmentation during training.")
    parser.add_argument("--distortion_scale", type=float, default=0.08, help="Scale of random local elastic distortion.")
    parser.add_argument("--perspective_scale", type=float, default=0.04, help="Scale of random perspective warp.")
    parser.add_argument("--translation_scale", type=float, default=0.02, help="Scale of random horizontal/vertical translation (shift).")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--resume_lr",
        type=float,
        default=None,
        help="Override optimizer learning rate after loading a resumed checkpoint.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="no")
    parser.add_argument("--lambda_rgb", type=float, default=1.0)
    parser.add_argument("--lambda_alpha", type=float, default=0.5)
    parser.add_argument("--lambda_render", type=float, default=0.1)
    parser.add_argument("--lambda_edge", type=float, default=0.25)
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
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--preview_every", type=int, default=1)
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from.")
    return parser


def main():
    args = build_arg_parser().parse_args()
    args.conditioning_mode = "uv_unproject_inpaint"
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "previews").mkdir(exist_ok=True)

    device = get_device(args.device)
    dataset = InverseUVDataset(
        data_dir=args.data_dir,
        mappings_dir=args.mappings_dir,
        views=args.views,
        image_size=args.render_size,
        include_alpha=args.include_alpha,
        max_samples=args.max_samples,
        augment=args.augment,
        distortion_scale=args.distortion_scale,
        perspective_scale=args.perspective_scale,
        translation_scale=args.translation_scale,
    )
    input_channels = dataset.input_channels

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

    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location=device)
        checkpoint_model_config = resume_checkpoint.get("model_config")
        checkpoint_args = resume_checkpoint.get("args", {})
        if checkpoint_model_config is not None:
            args.base_channels = checkpoint_model_config.get("base_channels", args.base_channels)
            args.coordconv = checkpoint_model_config.get("use_coordconv", args.coordconv)
            args.bottleneck_attention = checkpoint_model_config.get("use_attention", args.bottleneck_attention)
            args.attention_heads = checkpoint_model_config.get("attention_heads", args.attention_heads)
        elif "coordconv" in checkpoint_args or "bottleneck_attention" in checkpoint_args:
            args.base_channels = checkpoint_args.get("base_channels", args.base_channels)
            args.coordconv = checkpoint_args.get("coordconv", args.coordconv)
            args.bottleneck_attention = checkpoint_args.get("bottleneck_attention", args.bottleneck_attention)
            args.attention_heads = checkpoint_args.get("attention_heads", args.attention_heads)
        else:
            args.base_channels = checkpoint_args.get("base_channels", args.base_channels)
            args.coordconv = False
            args.bottleneck_attention = False

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=args.num_workers > 0,
        )

    model = InverseUVNet(
        input_channels=input_channels,
        base_channels=args.base_channels,
        use_coordconv=args.coordconv,
        use_attention=args.bottleneck_attention,
        attention_heads=args.attention_heads,
    ).to(device)
    criterion = InverseUVLoss(
        mappings_dir=args.mappings_dir,
        views=args.views,
        lambda_rgb=args.lambda_rgb,
        lambda_alpha=args.lambda_alpha,
        lambda_render=args.lambda_render,
        lambda_edge=args.lambda_edge,
        render_foreground_weight=args.render_foreground_weight,
        ignore_covered_inner=not args.supervise_covered_inner,
        covered_inner_alpha_threshold=args.covered_inner_alpha_threshold,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = build_grad_scaler(device, args.mixed_precision)

    start_epoch = 1
    if resume_checkpoint is not None:
        checkpoint = resume_checkpoint
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if args.resume_lr is not None:
            for group in optimizer.param_groups:
                group["lr"] = args.resume_lr
        start_epoch = checkpoint.get("epoch", 0) + 1

    metadata = {
        "num_samples": len(dataset),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset) if val_dataset is not None else 0,
        "input_channels": input_channels,
        "conditioning_mode": "uv_unproject_inpaint",
        "parameters": count_parameters(model),
        "views": dataset.views,
        "coordconv": args.coordconv,
        "bottleneck_attention": args.bottleneck_attention,
        "attention_heads": args.attention_heads,
        "lr": optimizer.param_groups[0]["lr"],
        "device": str(device),
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump({"args": vars(args), "metadata": metadata}, handle, indent=2)
    print(json.dumps(metadata, indent=2))

    # Instantiate augmenter if training and args.augment is true
    augmenter = None
    if args.augment:
        augmenter = RenderAugmenter(
            distortion_scale=args.distortion_scale,
            perspective_scale=args.perspective_scale,
            translation_scale=args.translation_scale,
            bg_color=dataset.bg_color,
        )

    best_metric = float("inf")
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            scaler,
            device,
            args.mixed_precision,
            train=True,
            views=dataset.views,
            augmenter=augmenter,
        )
        metrics = {"train": train_metrics}
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(
                    model,
                    criterion,
                    val_loader,
                    optimizer,
                    scaler,
                    device,
                    args.mixed_precision,
                    train=False,
                    views=dataset.views,
                    augmenter=None,
                )
            metrics["val"] = val_metrics

        metric = metrics.get("val", metrics["train"])["loss_total"]
        print(f"epoch={epoch} metrics={json.dumps(metrics, sort_keys=True)}")

        if epoch % args.preview_every == 0:
            model.eval()
            preview_batch = move_batch(next(iter(val_loader or train_loader)), device)
            with torch.no_grad():
                preview_conditioning = build_conditioning(
                    preview_batch["uv"],
                    criterion.renderer,
                    dataset.views,
                    augmenter=None,
                )
                pred_uv = model(preview_conditioning)
            save_preview(pred_uv, preview_batch["uv"], output_dir / "previews" / f"epoch_{epoch:04d}.png")

        if epoch % args.save_every == 0:
            save_checkpoint(output_dir / "latest.pt", model, optimizer, epoch, args, input_channels, metrics)
        if metric < best_metric:
            best_metric = metric
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, args, input_channels, metrics)


if __name__ == "__main__":
    main()
