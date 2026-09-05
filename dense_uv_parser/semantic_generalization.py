"""Checkpoint-compatible v61 fine tuning with pixel supervision and context regularization.

Run through run_local.py. Real-image generalization is an experimental objective;
synthetic appearance stress tests do not establish real-image accuracy.
"""
import argparse
import hashlib
import inspect
import json
import math
import os
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

from SkingToolkit.dense_uv_parser import train as core
from SkingToolkit.dense_uv_parser.model import DenseUVParserNet
from SkingToolkit.dense_uv_parser.losses import DenseUVParserLoss
from SkingToolkit.dense_uv_parser.semantic import attach_semantic_runtime
from SkingToolkit.dense_uv_parser.skin_dataset import SkinUVDataset
from SkingToolkit.dense_uv_parser.utils import IGNORE_INDEX
from SkingToolkit.dense_uv_parser.semantic_targets import (
    build_part_layer_masks, build_semantic_attribute_targets,
)
from SkingToolkit.renderer import DifferentiableRenderer


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temp, path)


@torch.no_grad()
def appearance_augment(images, foreground, strength=1.0, generator=None):
    """Change appearance without moving pixels, alpha, background, or GT labels."""
    rgb = images[:, :3].float()
    fg = foreground.float()
    n, _, h, w = rgb.shape
    def rand(*shape):
        return torch.rand(shape, device=rgb.device, generator=generator)
    def randn(*shape):
        return torch.randn(shape, device=rgb.device, generator=generator)
    # Normalize the blur by foreground support to prevent background bleeding.
    smooth = F.avg_pool2d(rgb * fg, 3, 1, 1) / F.avg_pool2d(fg, 3, 1, 1).clamp_min(1e-6)
    mix = rand(n, 1, 1, 1) * (0.55 * strength)
    altered = rgb * (1 - mix) + smooth * mix
    illumination = F.interpolate(
        randn(n, 1, 8, 4), size=(h, w), mode="bilinear", align_corners=False
    ) * (0.045 * strength)
    contrast = 1 + (rand(n, 1, 1, 1) - 0.5) * (0.30 * strength)
    tint = (rand(n, 3, 1, 1) - 0.5) * (0.08 * strength)
    altered = (altered - 0.5) * contrast + 0.5 + illumination + tint
    altered = altered + randn(n, 3, h, w) * (0.008 * strength)
    result = images.clone()
    result[:, :3] = torch.where(fg.bool(), altered.clamp(0, 1), rgb)
    return result


def flat_foreground_mask(images, targets):
    rgb = images[:, :3].float()
    variance = (F.avg_pool2d(rgb.square(), 3, 1, 1)
                - F.avg_pool2d(rgb, 3, 1, 1).square()).clamp_min(0)
    interior = F.avg_pool2d(targets["foreground"].float(), 3, 1, 1) > 0.999
    return (variance.amax(1) < (2 / 255) ** 2) & interior[:, 0]


def flat_route_loss(logits, labels, flat):
    per_pixel = F.cross_entropy(logits.float(), labels, ignore_index=IGNORE_INDEX, reduction="none")
    terms = [per_pixel[flat & (labels == role)].mean()
             for role in (0, 1) if (flat & (labels == role)).any()]
    return torch.stack(terms).mean() if terms else logits.sum() * 0


def head_outer_hard_positive_loss(logits, targets):
    """Mine the worst head-outer pixels using actual renderer labels."""
    head_outer = (targets["route_role"] == 1) & (targets["part"] == 0)
    if not head_outer.any():
        return logits.sum() * 0
    errors = -F.log_softmax(logits.float(), dim=1)[:, 1][head_outer]
    return errors.topk(max(1, math.ceil(errors.numel() * 0.20))).values.mean()


def skip_regularizer(model, probability):
    """Drop entire shallow skip inputs per image only during training."""
    def hook(module, inputs):
        if not model.training or probability == 0:
            return inputs
        x, skip = inputs
        keep = (torch.rand(skip.shape[0], 1, 1, 1, device=skip.device) >= probability)
        return x, skip * keep.to(skip.dtype)
    return [block.register_forward_pre_hook(hook) for block in (model.up0, model.up1)]


@contextmanager
def semantics_disabled(model):
    handles = []
    if model.semantic_spatial_fusion is not None:
        handles.append(model.semantic_spatial_fusion.register_forward_hook(
            lambda module, inputs, output: torch.zeros_like(output)))
    if model.semantic_fusion is not None:
        handles.append(model.semantic_fusion.register_forward_hook(
            lambda module, inputs, output: (torch.zeros_like(output[0]), output[1])))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def build_split(dataset, seed, val_split):
    val_count = int(len(dataset) * val_split)
    return random_split(dataset, [len(dataset) - val_count, val_count],
                        generator=torch.Generator().manual_seed(seed))


def count_routes(output, target, flat):
    labels = target["route_role"]
    valid = labels != IGNORE_INDEX
    pred = output["layer"].argmax(1)
    counts = {}
    for role, name in ((0, "inner"), (1, "outer"), (2, "secondary")):
        gt, selected = labels == role, (pred == role) & valid
        counts[f"{name}_tp"] = int((gt & selected).sum())
        counts[f"{name}_fp"] = int((~gt & selected).sum())
        counts[f"{name}_fn"] = int((gt & ~selected).sum())
    inner_flat = flat & (labels == 0)
    counts["flat_inner_count"] = int(inner_flat.sum())
    counts["flat_inner_false_outer"] = int((inner_flat & (pred == 1)).sum())
    head = (target["part"] == 0) & valid
    gt_head_outer, pred_head_outer = head & (labels == 1), head & (pred == 1)
    counts["head_outer_tp"] = int((gt_head_outer & pred_head_outer).sum())
    counts["head_outer_fp"] = int((~gt_head_outer & pred_head_outer).sum())
    counts["head_outer_fn"] = int((gt_head_outer & ~pred_head_outer).sum())
    return counts


def summarize_counts(counts):
    values = dict(counts)
    for name in ("inner", "outer", "secondary", "head_outer"):
        tp, fp, fn = [counts[f"{name}_{k}"] for k in ("tp", "fp", "fn")]
        values[f"{name}_precision"] = tp / max(1, tp + fp)
        values[f"{name}_recall"] = tp / max(1, tp + fn)
        values[f"{name}_iou"] = tp / max(1, tp + fp + fn)
    values["flat_inner_false_outer_rate"] = (
        counts["flat_inner_false_outer"] / max(1, counts["flat_inner_count"]))
    return values


@torch.no_grad()
def evaluate(model, renderer, loader, args, device, ablation=True, hard=True):
    model.eval()
    totals = {name: {} for name in ("clean", "appearance")}
    if ablation:
        totals["no_semantics"] = {}
    hard_totals = {}
    rng = torch.Generator(device=device).manual_seed(90451)
    for batch in loader:
        uv = batch["uv"].to(device)
        images, targets, _, ids = core.build_parser_inputs(uv, renderer, args.views, False, args)
        flat = flat_foreground_mask(images, targets)
        with core.autocast_context(device, "bf16"):
            # Features always correspond to the actual image being classified.
            clean_features = model._runtime_semantic_features(images, targets["foreground"])
            for mode in totals:
                if mode == "appearance":
                    actual = appearance_augment(images, targets["foreground"], 1.25, rng)
                    output = model(actual, view_ids=ids, semantic_foreground=targets["foreground"])
                elif mode == "no_semantics":
                    with semantics_disabled(model):
                        output = model(images, view_ids=ids, semantic_features=clean_features)
                else:
                    output = model(images, view_ids=ids, semantic_features=clean_features)
                counts = count_routes(output, targets, flat)
                for key, value in counts.items():
                    totals[mode][key] = totals[mode].get(key, 0) + value
                if mode == "clean" and hard:
                    hm = core.hard_uv_conditioning_metrics(images, output, targets, renderer, args.views, args)
                    for key, value in hm.items():
                        if key.startswith("count_"):
                            hard_totals[key] = hard_totals.get(key, 0.0) + float(value)
    results = {k: summarize_counts(v) for k, v in totals.items()}
    if hard_totals:
        hard_metrics = {}
        for name in ("inner", "outer"):
            tp, fp, fn = [hard_totals[f"count_hard_{name}_{k}"] for k in ("tp", "fp", "fn")]
            for metric, value in (("precision", tp / max(1, tp + fp)),
                                  ("recall", tp / max(1, tp + fn)),
                                  ("iou", tp / max(1, tp + fp + fn))):
                hard_metrics[f"{name}_{metric}"] = value
            hard_metrics[f"{name}_rgb_mae"] = (
                hard_totals[f"count_hard_{name}_rgb_abs"] /
                max(1, hard_totals[f"count_hard_{name}_rgb_values"]))
        results["hard_uv"] = hard_metrics
    return results


def eligible(metrics, baseline, tolerance):
    # A robustness gain cannot purchase a large regression in ordinary routing.
    for section in ("clean", "hard_uv"):
        for name in ("inner_iou", "outer_iou", "outer_precision", "outer_recall"):
            if metrics[section][name] < baseline[section][name] - tolerance:
                return False
    return True


def robustness_score(metrics):
    appearance = metrics["appearance"]
    return (0.40 * appearance["head_outer_recall"]
            + 0.25 * appearance["outer_recall"]
            + 0.25 * appearance["outer_iou"]
            + 0.10 * appearance["inner_iou"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--checkpoint", type=Path, default=root / "dense_uv_parser/runs/dense_uv_parser_v61/best.pt")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train_samples", type=int, default=0, help="0 uses all original 162000 training skins")
    parser.add_argument("--val_samples", type=int, default=512)
    parser.add_argument("--test_samples", type=int, default=256,
                        help="Additional original validation skins used only for the final report")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--semantic_lr", type=float, default=5e-5)
    parser.add_argument("--skip_dropout", type=float, default=0.10)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--max_clean_regression", type=float, default=0.015)
    parser.add_argument("--evaluate_only", action="store_true")
    opt = parser.parse_args()
    if min(opt.epochs, opt.val_samples, opt.batch_size, opt.eval_every, opt.log_every) < 1:
        parser.error("epochs, val_samples, batch_size, eval_every, log_every must be positive")
    if not 0 <= opt.skip_dropout < 1 or min(opt.train_samples, opt.test_samples, opt.workers) < 0:
        parser.error("invalid dropout or sample/worker count")
    opt.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    core.seed_everything(1234, reproducible=False)
    checkpoint = torch.load(opt.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["model_config"].get("predict_outer_uv_occupancy"):
        raise ValueError("Expected the v61 model without an occupancy head")
    args_dict = vars(core.build_arg_parser().parse_args(["--data_dir", str(root / "skins")]))
    args_dict.update(checkpoint["args"])
    args_dict.update(data_dir=str(root / "skins"),
                     mappings_dir=str(root.parent / "github/differentiable_minecraft_renderer/mappings_256x512"),
                     output_dir=str(opt.output_dir), privileged_views="", resume=str(opt.checkpoint),
                     siglip_local_files_only=True, best_metric="semantic_robustness",
                     outer_uv_min_source_pixels=33,
                     lambda_outer_false_negative=1.25,
                     route_outer_class_weight_cap=1.0)
    args = SimpleNamespace(**args_dict)
    args.views = core.parse_views(args.views)
    args.epochs = checkpoint["epoch"] + opt.epochs
    # The original 180k universe and random split MUST precede any pilot subsampling.
    dataset = SkinUVDataset(args.data_dir, max_samples=args.max_samples)
    original_cache = root / "dense_uv_parser" / checkpoint["args"]["siglip_cache_dir"] / "metadata.json"
    if original_cache.is_file():
        original_names = json.loads(original_cache.read_text())["filenames"]
        if [path.name for path in dataset.skin_paths] != original_names:
            raise ValueError("Dataset ordering differs from the v61 cache; reconstruct the original split first")
    train_set, val_set = build_split(dataset, args.seed, args.val_split)
    if opt.train_samples:
        train_set = Subset(dataset, train_set.indices[:opt.train_samples])
    test_set = Subset(dataset, val_set.indices[opt.val_samples:opt.val_samples + opt.test_samples])
    val_set = Subset(dataset, val_set.indices[:opt.val_samples])
    if not len(train_set) or not len(val_set):
        raise ValueError("Empty training/validation split")
    if set(train_set.indices) & set(val_set.indices):
        raise RuntimeError("Training/validation leakage")
    loader_args = dict(batch_size=opt.batch_size, num_workers=opt.workers, pin_memory=True)
    if opt.workers:
        loader_args.update(persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(train_set, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed + 1), **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args) if len(test_set) else None
    model_kwargs = {key: val for key, val in checkpoint["model_config"].items()
                    if key in inspect.signature(DenseUVParserNet).parameters}
    model = DenseUVParserNet(**model_kwargs).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    # Static coordinate bias is frozen: only image-conditioned behavior is adapted.
    model.route_role_prior.requires_grad_(False)
    attach_semantic_runtime(model, "siglip2", args.siglip_model, device, local_files_only=True)
    renderer = DifferentiableRenderer(args.mappings_dir).to(device)
    loss_kwargs = {key: value for key, value in args_dict.items()
                   if key in inspect.signature(DenseUVParserLoss).parameters}
    loss_kwargs.update(use_uv=False, affine_translation_limit=model.affine_translation_limit,
                       affine_log_scale_limit=model.affine_log_scale_limit)
    criterion = DenseUVParserLoss(**loss_kwargs).to(device)
    masks = tuple(mask.to(device) for mask in build_part_layer_masks())
    semantic, visual = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (semantic if name.startswith(("semantic_", "outer_presence", "outer_coverage"))
             else visual).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": visual, "lr": opt.lr, "initial_lr": opt.lr},
        {"params": semantic, "lr": opt.semantic_lr, "initial_lr": opt.semantic_lr},
    ], weight_decay=args.weight_decay)
    handles = skip_regularizer(model, opt.skip_dropout)
    manifest = {"checkpoint": str(opt.checkpoint.resolve()),
                "checkpoint_sha256": hashlib.sha256(opt.checkpoint.read_bytes()).hexdigest(),
                "source_epoch": checkpoint["epoch"], "checkout": str(root),
                "code_files": {Path(m.__file__).name: str(Path(m.__file__).resolve())
                               for m in (core, inspect.getmodule(DenseUVParserNet), inspect.getmodule(DenseUVParserLoss))},
                "options": {k: str(v) if isinstance(v, Path) else v for k, v in vars(opt).items()},
                "train_indices": list(train_set.indices), "val_indices": list(val_set.indices),
                "test_indices": list(test_set.indices),
                "dataset_filenames_sha256": hashlib.sha256(
                    "\n".join(p.name for p in dataset.skin_paths).encode()).hexdigest(),
                "label_source": "original skin RGBA and v61 renderer surface visibility; no v94 pseudo-labels",
                "semantic_features": "online from actual input, frozen SigLIP2",
                "optimizer": "fresh AdamW for changed fine-tuning objective; v61 model loaded strictly",
                "routing": "v61 thresholds held fixed for hard UV validation, source pixel minimum 33"}
    for path in manifest["code_files"].values():
        if not Path(path).is_relative_to(root):
            raise RuntimeError(f"Imported outside checkout: {path}")
    write_json(opt.output_dir / "config.json", {"args": vars(args), "experiment": manifest})
    source_dir = opt.output_dir / "source"
    source_dir.mkdir()
    for source in (root / "dense_uv_parser").glob("*.py"):
        shutil.copy2(source, source_dir / source.name)
    shutil.copy2(root / "renderer.py", source_dir / "renderer.py")
    started = time.time()
    write_json(opt.output_dir / "status.json", {"state": "baseline", "pid": os.getpid()})
    print("Evaluating untouched v61 baseline", flush=True)
    baseline = evaluate(model, renderer, val_loader, args, device)
    write_json(opt.output_dir / "baseline.json", baseline)
    print("baseline=" + json.dumps(baseline), flush=True)
    if opt.evaluate_only:
        write_json(opt.output_dir / "status.json", {"state": "evaluated", "pid": os.getpid()})
        return
    best_score = robustness_score(baseline)
    step = 0
    stop = False
    rejection_streak = 0
    args.semantic_generalization = manifest["options"]
    for epoch in range(1, opt.epochs + 1):
        running = 0.0
        for batch_index, batch in enumerate(train_loader):
            model.train()
            progress = (step / max(1, opt.epochs * len(train_loader) - 1))
            factor = 0.2 + 0.8 * (1 + math.cos(math.pi * progress)) / 2
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * factor
            uv = batch["uv"].to(device, non_blocking=True)
            images, targets, _, ids = core.build_parser_inputs(uv, renderer, args.views, True, args)
            flat = flat_foreground_mask(images, targets)
            augmented = step % 2 == 1
            actual = appearance_augment(images, targets["foreground"]) if augmented else images
            optimizer.zero_grad(set_to_none=True)
            with core.autocast_context(device, "bf16"):
                output = model(actual, view_ids=ids, semantic_foreground=targets["foreground"])
                losses = criterion(output, targets)
                loss = losses["loss_total"]
                loss = loss + 0.10 * flat_route_loss(output["layer"], targets["route_role"], flat)
                loss = loss + 0.15 * head_outer_hard_positive_loss(output["layer"], targets)
                attributes = build_semantic_attribute_targets(uv, *masks)
                loss = loss + 0.25 * F.binary_cross_entropy_with_logits(
                    output["outer_presence_logits"], attributes["outer_presence"])
                loss = loss + 0.25 * F.smooth_l1_loss(output["outer_coverage"], attributes["outer_coverage"])
                if not augmented:
                    geometry = core.differentiable_geometry_losses(images, uv, output, targets, renderer, args.views)
                    for key in ("soft_uv_rgb", "soft_uv_alpha", "soft_uv_inner_recall",
                                "soft_uv_outer_recall", "render_rgb", "render_alpha"):
                        loss = loss + getattr(args, "lambda_" + key) * geometry["loss_" + key]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            running += float(loss.detach())
            if step % opt.log_every == 0 or step == 1:
                status = {"state": "training", "pid": os.getpid(), "step": step, "epoch": epoch,
                          "total_steps": opt.epochs * len(train_loader), "loss": running / (batch_index + 1),
                          "gradient_norm": float(norm), "semantic_lr": optimizer.param_groups[1]["lr"],
                          "elapsed_seconds": time.time() - started}
                write_json(opt.output_dir / "status.json", status)
                print(json.dumps(status), flush=True)
            if step % opt.eval_every == 0 or batch_index + 1 == len(train_loader):
                metrics = evaluate(model, renderer, val_loader, args, device)
                score = robustness_score(metrics)
                accepted = eligible(metrics, baseline, opt.max_clean_regression)
                promoted = accepted and score > best_score
                record = {"step": step, "epoch": epoch, "score": score, "clean_guard_pass": accepted,
                          "promoted": promoted, "metrics": metrics, "elapsed_seconds": time.time() - started}
                with (opt.output_dir / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print("evaluation=" + json.dumps(record), flush=True)
                if promoted:
                    best_score = score
                args.finetune_step = step
                saved_metrics = {"val": metrics, "semantic_robustness": score}
                core.save_checkpoint(opt.output_dir / "latest.pt", model, optimizer, None,
                                     checkpoint["epoch"] + epoch, args, saved_metrics, best_metric=-best_score)
                if promoted:
                    core.save_checkpoint(opt.output_dir / "best.pt", model, optimizer, None,
                                         checkpoint["epoch"] + epoch, args, saved_metrics, best_metric=-best_score)
                write_json(opt.output_dir / "last_evaluation.json", record)
                rejection_streak = 0 if accepted else rejection_streak + 1
                if rejection_streak >= 2:
                    stop = True
                    break
        if stop:
            break
    for handle in handles:
        handle.remove()
    if test_loader is not None and (opt.output_dir / "best.pt").is_file():
        write_json(opt.output_dir / "status.json", {"state": "final_test", "pid": os.getpid(), "step": step})
        best = torch.load(opt.output_dir / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(best["model"], strict=True)
        candidate_test = evaluate(model, renderer, test_loader, args, device)
        model.load_state_dict(checkpoint["model"], strict=True)
        baseline_test = evaluate(model, renderer, test_loader, args, device)
        write_json(opt.output_dir / "heldout_test.json", {
            "note": "Not used for checkpoint selection; appearance stress is synthetic, not real-image ground truth",
            "samples": len(test_set), "candidate": candidate_test, "baseline": baseline_test})
    write_json(opt.output_dir / "status.json", {"state": "stopped_clean_regression" if stop else "complete",
               "pid": os.getpid(), "step": step, "best_score": best_score,
               "baseline_score": robustness_score(baseline), "best_checkpoint_exists": (opt.output_dir / "best.pt").exists(),
               "elapsed_seconds": time.time() - started})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--output_dir" in sys.argv:
            index = sys.argv.index("--output_dir") + 1
            if index < len(sys.argv):
                failed_dir = Path(sys.argv[index])
                if failed_dir.is_dir() and not isinstance(error, FileExistsError):
                    write_json(failed_dir / "status.json", {
                        "state": "failed", "pid": os.getpid(),
                        "error": f"{type(error).__name__}: {error}"})
        raise
