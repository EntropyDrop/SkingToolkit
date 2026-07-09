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

from SkingToolkit.inverse_uv.dataset import InverseUVDataset, build_conditioning, finalize_minecraft_alpha, RenderAugmenter, parse_views  # noqa: E402
from SkingToolkit.inverse_uv.losses import InverseUVLoss  # noqa: E402
from SkingToolkit.inverse_uv.model import InverseUVNet, PatchGANDiscriminator, count_parameters  # noqa: E402

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
    unproject_mode="mean",
    log_every=50,
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
            batch_augmenter = augmenter if train else None
            result = build_conditioning(
                batch["uv"],
                criterion.renderer,
                views,
                augmenter=batch_augmenter,
                unproject_mode=unproject_mode,
                return_renders=True,
            )
            conditioning, gt_renders = result

        with torch.set_grad_enabled(train):
            with autocast_context(device, precision):
                pred_uv = model(conditioning)
                losses = criterion(pred_uv, batch["uv"], gt_renders=gt_renders)
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
    parser.add_argument("--augment", action="store_true", help="Enable render-space data augmentation for pose robustness.")
    parser.add_argument("--translation_scale", type=float, default=0.03, help="Render-space translation scale for augmentation.")
    parser.add_argument("--scale_range", type=float, default=0.03, help="Render-space uniform scale range for augmentation.")
    parser.add_argument("--perspective_scale", type=float, default=0.008, help="Render-space perspective warp scale for augmentation.")
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
        "--unproject_mode",
        choices=["mode", "mean", "medoid"],
        default="mean",
        help="Method to aggregate render pixels into 64x64 UV texels ('mode'=most frequent color, 'mean'=average, 'medoid'=spatial median).",
    )
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


def main():
    args = build_arg_parser().parse_args()
    args.conditioning_mode = "uv_unproject_inpaint"
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "previews").mkdir(exist_ok=True)

    log_mode = "a" if args.resume else "w"
    sys.stdout = Logger(output_dir / "train.log", sys.stdout, mode=log_mode)
    sys.stderr = Logger(output_dir / "train.log", sys.stderr, mode=log_mode)

    device = get_device(args.device)
    configure_torch(args, device)
    dataset = InverseUVDataset(
        data_dir=args.data_dir,
        mappings_dir=args.mappings_dir,
        views=args.views,
        image_size=args.render_size,
        include_alpha=args.include_alpha,
        max_samples=args.max_samples,
        unproject_mode=args.unproject_mode,
        augment=args.augment,
        translation_scale=args.translation_scale,
        scale_range=args.scale_range,
        perspective_scale=args.perspective_scale,
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

    model = InverseUVNet(input_channels=input_channels, base_channels=args.base_channels).to(device)

    discriminator = None
    d_optimizer = None
    if args.lambda_gan > 0:
        discriminator = PatchGANDiscriminator(base_channels=64).to(device)
        d_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=args.lr, weight_decay=0.0)

    criterion = InverseUVLoss(
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = build_grad_scaler(device, args.mixed_precision)

    start_epoch = 1
    best_metric = float("inf")
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
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
        "conditioning_mode": "uv_unproject_inpaint",
        "unproject_mode": args.unproject_mode,
        "best_metric": args.best_metric,
        "scheduler": args.scheduler,
        "min_lr": args.min_lr,
        "log_every": args.log_every,
        "prefetch_factor": args.prefetch_factor,
        "matmul_precision": args.matmul_precision,
        "cudnn_benchmark": args.cudnn_benchmark,
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
            unproject_mode=args.unproject_mode, log_every=args.log_every,
        )
        metrics = {"train": train_metrics}
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(
                    model, criterion, val_loader, optimizer, scaler, device, args.mixed_precision,
                    train=False, d_optimizer=None, views=views, augmenter=None,
                    unproject_mode=args.unproject_mode, log_every=args.log_every,
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
                preview_cond = build_conditioning(
                    preview_batch["uv"],
                    criterion.renderer,
                    views,
                    augmenter=None,
                    unproject_mode=args.unproject_mode,
                )
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
