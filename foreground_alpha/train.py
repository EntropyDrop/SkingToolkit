import argparse
import json
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

from SkingToolkit.foreground_alpha.dataset import ForegroundAlphaDataset, parse_color  # noqa: E402
from SkingToolkit.foreground_alpha.model import ForegroundAlphaNet, count_parameters  # noqa: E402

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


def build_grad_scaler(device, precision):
    enabled = device.type == "cuda" and precision == "fp16"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def move_batch(batch, device):
    return {
        "image": batch["image"].to(device, non_blocking=True),
        "alpha": batch["alpha"].to(device, non_blocking=True),
        "path": batch["path"],
        "view": batch["view"],
    }


def edge_loss(pred, target):
    dx_pred = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    dx_tgt = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
    dy_pred = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    dy_tgt = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
    return F.l1_loss(dx_pred, dx_tgt) + F.l1_loss(dy_pred, dy_tgt)


def dice_loss(pred, target, eps=1e-6):
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)
    intersect = (pred_flat * target_flat).sum(dim=1)
    denom = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    return (1.0 - (2.0 * intersect + eps) / (denom + eps)).mean()


def hole_loss(pred, target):
    fg_mask = (target > 0.8).float()
    hole_diff = F.relu(target - pred)
    return (fg_mask * (hole_diff ** 2)).sum() / (fg_mask.sum() + 1e-6)


def alpha_losses(pred, target, args):
    pred = pred.clamp(1e-6, 1.0 - 1e-6)
    losses = {
        "loss_bce": F.binary_cross_entropy(pred, target),
        "loss_l1": F.l1_loss(pred, target),
        "loss_dice": dice_loss(pred, target),
        "loss_edge": edge_loss(pred, target),
        "loss_hole": hole_loss(pred, target),
    }
    lambda_hole = getattr(args, "lambda_hole", 1.0)
    losses["loss_total"] = (
        args.lambda_bce * losses["loss_bce"]
        + args.lambda_l1 * losses["loss_l1"]
        + args.lambda_dice * losses["loss_dice"]
        + args.lambda_edge * losses["loss_edge"]
        + lambda_hole * losses["loss_hole"]
    )
    return losses


def format_losses(loss_sums, count):
    return {name: value / max(count, 1) for name, value in loss_sums.items()}


def run_epoch(model, loader, optimizer, scaler, device, precision, args, train=True):
    model.train(train)
    loss_sums = {}
    sample_count = 0
    iterator = tqdm(loader, leave=False) if tqdm is not None else loader

    for batch in iterator:
        batch = move_batch(batch, device)
        batch_size = batch["image"].shape[0]
        with torch.set_grad_enabled(train):
            with autocast_context(device, precision):
                pred = model(batch["image"])
                losses = alpha_losses(pred, batch["alpha"], args)
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
            iterator.set_postfix(total=f"{avg['loss_total']:.4f}", l1=f"{avg['loss_l1']:.4f}")

    return format_losses(loss_sums, sample_count)


def save_preview(batch, pred, output_path, max_items=4):
    count = min(max_items, pred.shape[0])
    images = batch["image"][:count].detach().cpu()
    pred_alpha = pred[:count].detach().cpu().expand(-1, 3, -1, -1)
    target_alpha = batch["alpha"][:count].detach().cpu().expand(-1, 3, -1, -1)
    preview = torch.cat([images, pred_alpha, target_alpha], dim=0)
    save_image(preview.clamp(0.0, 1.0), output_path, nrow=count)


def save_checkpoint(path, model, optimizer, epoch, args, metrics):
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "metrics": metrics,
    }
    torch.save(checkpoint, path)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train RGB render foreground alpha extraction model.")
    parser.add_argument("--data_dir", required=True, help="Folder containing GT 64x64 RGBA skin PNGs.")
    parser.add_argument("--output_dir", default="foreground_alpha_runs/default")
    parser.add_argument("--mappings_dir", default=None)
    parser.add_argument("--views", default="walk_front_both_layer_ortho,walk_back_both_layer_ortho")
    parser.add_argument(
        "--background_mode",
        choices=["random", "black", "white", "gray", "color", "gradient", "pattern", "hard"],
        default="random",
    )
    parser.add_argument("--bg_color", default="0,0,0", help="Used when --background_mode color.")
    parser.add_argument("--hard_bg_prob", type=float, default=0.3, help="Probability of sampling foreground color as background.")
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="no")
    parser.add_argument("--lambda_bce", type=float, default=1.0)
    parser.add_argument("--lambda_l1", type=float, default=1.0)
    parser.add_argument("--lambda_dice", type=float, default=0.5)
    parser.add_argument("--lambda_edge", type=float, default=0.25)
    parser.add_argument("--lambda_hole", type=float, default=1.0, help="Weight for interior foreground hole penalty loss.")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--preview_every", type=int, default=1)
    parser.add_argument("--resume", default=None)
    return parser


def main():
    args = build_arg_parser().parse_args()
    args.bg_color = parse_color(args.bg_color)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "previews").mkdir(exist_ok=True)

    device = get_device(args.device)
    dataset = ForegroundAlphaDataset(
        data_dir=args.data_dir,
        mappings_dir=args.mappings_dir,
        views=args.views,
        background_mode=args.background_mode,
        bg_color=args.bg_color,
        hard_bg_prob=args.hard_bg_prob,
        max_samples=args.max_samples,
    )

    val_len = int(len(dataset) * args.val_split) if args.val_split > 0 else 0
    if val_len > 0 and len(dataset) - val_len > 0:
        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_len, val_len], generator=generator)
    else:
        train_dataset, val_dataset = dataset, None

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

    model = ForegroundAlphaNet(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = build_grad_scaler(device, args.mixed_precision)

    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", 0) + 1

    metadata = {
        "num_samples": len(dataset),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset) if val_dataset is not None else 0,
        "views": dataset.views,
        "background_mode": args.background_mode,
        "parameters": count_parameters(model),
        "device": str(device),
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump({"args": vars(args), "metadata": metadata}, handle, indent=2)
    print(json.dumps(metadata, indent=2))

    best_metric = float("inf")
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, args.mixed_precision, args, train=True)
        metrics = {"train": train_metrics}
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(model, val_loader, optimizer, scaler, device, args.mixed_precision, args, train=False)
            metrics["val"] = val_metrics

        metric = metrics.get("val", metrics["train"])["loss_total"]
        print(f"epoch={epoch} metrics={json.dumps(metrics, sort_keys=True)}")

        if epoch % args.preview_every == 0:
            model.eval()
            preview_batch = move_batch(next(iter(val_loader or train_loader)), device)
            with torch.no_grad():
                pred = model(preview_batch["image"])
            save_preview(preview_batch, pred, output_dir / "previews" / f"epoch_{epoch:04d}.png")

        if epoch % args.save_every == 0:
            save_checkpoint(output_dir / "latest.pt", model, optimizer, epoch, args, metrics)
        if metric < best_metric:
            best_metric = metric
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, args, metrics)


if __name__ == "__main__":
    main()
