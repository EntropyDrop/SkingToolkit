"""Precompute frozen SigLIP2 features for fixed renderer views.

The direct trainer has no render augmentation, so recomputing the same frozen
vision tower output in every epoch wastes most of the semantic-encoder time.
The optional spatial cache stores full-width FP16 patch features after removing
only deterministic letterbox padding. Trainable projections remain in the
downstream model.
"""

import argparse
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TOOLKIT_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from SkingToolkit.renderer import DifferentiableRenderer  # noqa: E402
from SkingToolkit.semantic_uv_reconstruction.dataset import parse_views  # noqa: E402
from SkingToolkit.semantic_uv_reconstruction.semantic_backbone import (  # noqa: E402
    SigLIP2VisionBackbone,
)
from SkingToolkit.semantic_uv_reconstruction.semantic_dataset import (  # noqa: E402
    SIGLIP_CACHE_VERSION,
    SemanticUVPairDataset,
    SigLIPGlobalCache,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def get_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def autocast_context(device, precision):
    if precision == "no" or device.type == "cpu":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def cache_is_reusable(
    cache_dir,
    dataset,
    views,
    siglip_model,
    require_spatial=False,
):
    try:
        cache = SigLIPGlobalCache(
            cache_dir,
            expected_views=views,
            expected_model=siglip_model,
            expected_data_dir=dataset.data_dir,
            require_spatial=require_spatial,
        )
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return list(cache.filename_to_index) == [path.name for path in dataset.skin_paths]


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Cache fixed-view SigLIP2 global and optional spatial features."
    )
    parser.add_argument("--data_dir", default="../skins")
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--mappings_dir", required=True)
    parser.add_argument(
        "--views",
        default="walk_front_both_layer_ortho,walk_back_both_layer_ortho",
    )
    parser.add_argument("--siglip_model", default="google/siglip2-base-patch16-224")
    parser.add_argument("--siglip_local_files_only", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument(
        "--spatial",
        action="store_true",
        help="Also cache raw cropped patch features as FP16 mmap data.",
    )
    parser.add_argument("--force", action="store_true")
    return parser


@torch.inference_mode()
def main():
    args = build_arg_parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be positive.")
    views = parse_views(args.views)
    if len(views) < 2:
        raise ValueError("At least front and back cache views are required.")
    dataset = SemanticUVPairDataset(args.data_dir, max_samples=args.max_samples)
    cache_dir = Path(args.cache_dir)
    if not args.force and cache_is_reusable(
        cache_dir,
        dataset,
        views,
        args.siglip_model,
        require_spatial=args.spatial,
    ):
        cache_kind = "global + spatial" if args.spatial else "global"
        print(f"Reusing complete SigLIP {cache_kind} cache: {cache_dir}")
        return

    device = get_device(args.device)
    renderer = DifferentiableRenderer(mappings_dir=args.mappings_dir).to(device).eval()
    missing_views = [view for view in views if view not in renderer.views]
    if missing_views:
        raise ValueError(f"Renderer cache mappings are missing views {missing_views}.")
    backbone = SigLIP2VisionBackbone(
        model_name=args.siglip_model,
        token_channels=128,
        local_files_only=args.siglip_local_files_only,
    ).to(device).eval()

    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)

    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary_embeddings = cache_dir / ".embeddings.npy.tmp"
    temporary_spatial_embeddings = cache_dir / ".spatial_embeddings.npy.tmp"
    temporary_metadata = cache_dir / ".metadata.json.tmp"
    embeddings = np.lib.format.open_memmap(
        temporary_embeddings,
        mode="w+",
        dtype=np.float16,
        shape=(len(dataset), len(views), backbone.raw_feature_dim),
    )
    spatial_embeddings = None
    spatial_shape = None
    iterator = tqdm(loader, desc="cache SigLIP2", leave=False) if tqdm else loader
    offset = 0
    for batch in iterator:
        uv = batch["uv"].to(device, non_blocking=True)
        renders = torch.stack(
            [renderer.forward_view(uv, view) for view in views], dim=1
        )
        batch_size, view_count, _, height, width = renders.shape
        images = renders[:, :, :3].reshape(batch_size * view_count, 3, height, width)
        if images.is_cuda:
            images = images.contiguous(memory_format=torch.channels_last)
        with autocast_context(device, args.mixed_precision):
            encoded = (
                backbone.encode_dense(images)
                if args.spatial
                else backbone.encode_global(images)
            )
        raw_global = encoded["raw_global"].reshape(
            batch_size, view_count, backbone.raw_feature_dim
        )
        embeddings[offset : offset + batch_size] = (
            raw_global.detach().to(device="cpu", dtype=torch.float16).numpy()
        )
        if args.spatial:
            raw_spatial = encoded["raw_spatial"]
            current_spatial_shape = tuple(raw_spatial.shape[-2:])
            if spatial_embeddings is None:
                spatial_shape = current_spatial_shape
                spatial_embeddings = np.lib.format.open_memmap(
                    temporary_spatial_embeddings,
                    mode="w+",
                    dtype=np.float16,
                    shape=(
                        len(dataset),
                        len(views),
                        backbone.raw_spatial_feature_dim,
                        *spatial_shape,
                    ),
                )
                spatial_size_gib = (
                    np.prod(spatial_embeddings.shape)
                    * np.dtype(np.float16).itemsize
                    / (1024**3)
                )
                print(
                    "SigLIP2 spatial cache: "
                    f"shape={spatial_embeddings.shape}, "
                    f"estimated_size={spatial_size_gib:.1f} GiB."
                )
            elif current_spatial_shape != spatial_shape:
                raise ValueError(
                    "SigLIP spatial shape changed within a fixed-view cache: "
                    f"expected {spatial_shape}, got {current_spatial_shape}."
                )
            raw_spatial = raw_spatial.reshape(
                batch_size,
                view_count,
                backbone.raw_spatial_feature_dim,
                *spatial_shape,
            )
            spatial_embeddings[offset : offset + batch_size] = (
                raw_spatial.detach().to(device="cpu", dtype=torch.float16).numpy()
            )
        offset += batch_size
    embeddings.flush()
    del embeddings
    if spatial_embeddings is not None:
        spatial_embeddings.flush()
        del spatial_embeddings
    if offset != len(dataset):
        raise RuntimeError(f"Cached {offset} samples, expected {len(dataset)}.")

    metadata = {
        "version": SIGLIP_CACHE_VERSION,
        "data_dir": str(Path(args.data_dir).resolve()),
        "filenames": [path.name for path in dataset.skin_paths],
        "views": views,
        "siglip_model": args.siglip_model,
        "feature_dim": backbone.raw_feature_dim,
        "dtype": "float16",
    }
    if args.spatial:
        if spatial_shape is None:
            raise RuntimeError("Spatial SigLIP cache was requested but no features were written.")
        metadata.update(
            {
                "spatial_feature_dim": backbone.raw_spatial_feature_dim,
                "spatial_height": spatial_shape[0],
                "spatial_width": spatial_shape[1],
                "spatial_dtype": "float16",
                "spatial_layout": "N,V,C,H,W",
                "spatial_letterbox_padding_removed": True,
            }
        )
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary_embeddings, cache_dir / "embeddings.npy")
    if args.spatial:
        os.replace(
            temporary_spatial_embeddings,
            cache_dir / "spatial_embeddings.npy",
        )
    os.replace(temporary_metadata, cache_dir / "metadata.json")
    cache_kind = "global + spatial" if args.spatial else "global"
    print(
        f"Cached {cache_kind} features for {len(dataset)} skins x "
        f"{len(views)} views to {cache_dir} "
        f"({backbone.raw_feature_dim} features/view)."
    )


if __name__ == "__main__":
    main()
