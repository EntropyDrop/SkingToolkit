"""Build a quality-gated manifest of real stage-one render / UV pairs."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TOOLKIT_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from SkingToolkit.dense_uv_parser.runtime import get_device  # noqa: E402
from SkingToolkit.dense_uv_parser.skin_dataset import (  # noqa: E402
    PairedRenderSkinDataset,
)
from SkingToolkit.dense_uv_parser.utils import parse_views  # noqa: E402
from SkingToolkit.renderer import DifferentiableRenderer  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


MANIFEST_VERSION = 2


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Reject archived *_edited files whose layout or contents do not "
            "match the paired versioned 64x64 result skin."
        )
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument(
        "--result_suffix",
        default="_v94_result",
        help=(
            "Strict UV result suffix paired with each *_edited input. Files "
            "using another suffix are never used as a fallback."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--mappings_dir", required=True)
    parser.add_argument("--views", default="front_left,back_left")
    parser.add_argument("--view_height", type=int, default=512)
    parser.add_argument("--view_width", type=int, default=256)
    parser.add_argument("--min_silhouette_iou", type=float, default=0.90)
    parser.add_argument("--max_rgb_mae", type=float, default=0.12)
    parser.add_argument("--background_seed_tolerance", type=float, default=0.08)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    return parser


def _manifest_is_reusable(path, args, candidate_count):
    if args.force or not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    expected = {
        "version": MANIFEST_VERSION,
        "data_dir": str(Path(args.data_dir).resolve()),
        "result_suffix": args.result_suffix,
        "views": parse_views(args.views),
        "view_size": [args.view_height, args.view_width],
        "min_silhouette_iou": args.min_silhouette_iou,
        "max_rgb_mae": args.max_rgb_mae,
        "background_seed_tolerance": args.background_seed_tolerance,
        "candidate_count": candidate_count,
    }
    return all(manifest.get(key) == value for key, value in expected.items())


@torch.inference_mode()
def main():
    args = build_arg_parser().parse_args()
    views = parse_views(args.views)
    if len(views) != 2:
        raise ValueError("Paired semantic filtering requires exactly two views.")
    for name in (
        "min_silhouette_iou",
        "max_rgb_mae",
        "background_seed_tolerance",
    ):
        if not 0.0 <= getattr(args, name) <= 1.0:
            raise ValueError(f"--{name} must be in [0, 1].")

    dataset = PairedRenderSkinDataset(
        args.data_dir,
        views=views,
        view_size=(args.view_height, args.view_width),
        result_suffix=args.result_suffix,
    )
    output_path = Path(args.output)
    if _manifest_is_reusable(output_path, args, len(dataset)):
        print(f"Reusing paired semantic manifest: {output_path}")
        return

    device = get_device(args.device)
    renderer = DifferentiableRenderer(
        mappings_dir=args.mappings_dir
    ).to(device).eval()
    for view in views:
        mapping_size = tuple(getattr(renderer, f"{view}_inner_mask").shape)
        if mapping_size != (args.view_height, args.view_width):
            raise ValueError(
                f"Mapping {view} has size {mapping_size}, expected "
                f"{(args.view_height, args.view_width)}."
            )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_kwargs)
    iterator = tqdm(loader, desc="filter paired semantics") if tqdm else loader
    accepted = []
    rejected = []
    data_root = Path(args.data_dir).resolve()

    for batch in iterator:
        uv = batch["uv"].to(device, non_blocking=True)
        source = batch["rendered"].to(device, non_blocking=True)
        reference = torch.stack(
            [renderer.forward_view(uv, view) for view in views], dim=1
        )
        batch_size, view_count, _, height, width = source.shape
        source_flat = source.reshape(-1, 4, height, width)
        reference_flat = reference.reshape(-1, 4, height, width)

        seed_rgb = source_flat[:, :3, :1, :1]
        source_foreground = (
            (source_flat[:, :3] - seed_rgb).abs().amax(dim=1)
            > args.background_seed_tolerance
        )
        reference_foreground = reference_flat[:, 3] > 0.5
        intersection = (source_foreground & reference_foreground).flatten(1)
        union = (source_foreground | reference_foreground).flatten(1)
        view_iou = intersection.sum(dim=1).float() / union.sum(
            dim=1
        ).clamp_min(1.0)
        shared = source_foreground & reference_foreground
        view_rgb_mae = (
            (
                (source_flat[:, :3] - reference_flat[:, :3])
                .abs()
                .mean(dim=1)
                * shared
            )
            .flatten(1)
            .sum(dim=1)
            / shared.flatten(1).sum(dim=1).clamp_min(1.0)
        )
        view_iou = view_iou.reshape(batch_size, view_count)
        view_rgb_mae = view_rgb_mae.reshape(batch_size, view_count)
        sample_iou = view_iou.amin(dim=1)
        sample_rgb_mae = view_rgb_mae.amax(dim=1)

        for index, (edited_path, result_path) in enumerate(
            zip(batch["path"], batch["uv_path"])
        ):
            record = {
                "edited": str(Path(edited_path).resolve().relative_to(data_root)),
                "result": str(Path(result_path).resolve().relative_to(data_root)),
                "silhouette_iou": round(float(sample_iou[index].item()), 6),
                "rgb_mae": round(float(sample_rgb_mae[index].item()), 6),
            }
            if (
                sample_iou[index] >= args.min_silhouette_iou
                and sample_rgb_mae[index] <= args.max_rgb_mae
            ):
                accepted.append(record)
            else:
                rejected.append(record)

    manifest = {
        "version": MANIFEST_VERSION,
        "data_dir": str(data_root),
        "result_suffix": args.result_suffix,
        "views": views,
        "view_size": [args.view_height, args.view_width],
        "min_silhouette_iou": args.min_silhouette_iou,
        "max_rgb_mae": args.max_rgb_mae,
        "background_seed_tolerance": args.background_seed_tolerance,
        "candidate_count": len(dataset),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "pairs": accepted,
        "rejected": rejected,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary_path, output_path)
    print(
        "Paired semantic manifest: "
        f"accepted={len(accepted)}, rejected={len(rejected)}, "
        f"output={output_path}"
    )


if __name__ == "__main__":
    main()
