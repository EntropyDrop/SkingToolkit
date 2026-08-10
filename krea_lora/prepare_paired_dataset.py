#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from common import load_config, resolve_device, seed_everything, write_json, write_jsonl
from reference_conditioning import fit_reference_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build strict source->MC preview pairs from source images and valid 64x64 result skins."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def collect_pairs(root: Path) -> list[tuple[str, Path, Path, Path | None]]:
    sources: dict[tuple[str, str], Path] = {}
    results: dict[tuple[str, str], Path] = {}
    edited: dict[tuple[str, str], Path] = {}
    for path in root.rglob("*_source.*"):
        if "_source_err" in path.name:
            continue
        sample_name = path.name.rsplit("_source", 1)[0]
        sources[(path.parent.relative_to(root).as_posix(), sample_name)] = path
    for path in root.rglob("*_result.*"):
        if "_result_err" in path.name:
            continue
        sample_name = path.name.rsplit("_result", 1)[0]
        results[(path.parent.relative_to(root).as_posix(), sample_name)] = path
    for path in root.rglob("*_edited.*"):
        if "_edited_err" in path.name:
            continue
        sample_name = path.name.rsplit("_edited", 1)[0]
        edited[(path.parent.relative_to(root).as_posix(), sample_name)] = path
    pairs = []
    for key in sorted(sources.keys() & results.keys()):
        relative_group, sample_name = key
        stable_key = f"{relative_group}/{sample_name}"
        pairs.append((stable_key, sources[key], results[key], edited.get(key)))
    return pairs


def load_skin(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        if image.size != (64, 64):
            raise ValueError(f"expected a 64x64 result skin, found {image.size}")
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    return torch.from_numpy(rgba).permute(2, 0, 1).float().div_(255.0)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data = config["data"]
    seed = int(config["training"]["seed"])
    seed_everything(seed)
    source_root = Path(data["dataset_source_dir"]).expanduser().resolve()
    toolkit_dir = Path(data["toolkit_dir"]).expanduser().resolve()
    mappings_dir = Path(data["mappings_dir"]).expanduser().resolve()
    dataset_dir = Path(data["dataset_dir"]).expanduser().resolve()
    manifest_path = dataset_dir / "metadata.jsonl"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"{manifest_path} already exists; pass --force to rebuild")
    for required in (source_root, mappings_dir):
        if not required.is_dir():
            raise FileNotFoundError(required)
    if not (toolkit_dir / "renderer.py").is_file():
        raise FileNotFoundError(toolkit_dir / "renderer.py")

    sys.path.insert(0, str(toolkit_dir))
    from renderer import DifferentiableRenderer

    pairs = collect_pairs(source_root)
    random.Random(seed).shuffle(pairs)
    configured_max = int(data.get("max_images", 0))
    max_images = args.max_images if args.max_images is not None else configured_max
    if max_images and max_images > 0:
        pairs = pairs[:max_images]
    if not pairs:
        raise RuntimeError(f"No valid source/result pairs found under {source_root}")

    device = resolve_device(args.device)
    batch_size = int(args.batch_size or data.get("render_batch_size", 8))
    background = tuple(float(value) for value in data["background_rgb"])
    renderer = DifferentiableRenderer(str(mappings_dir), bg_color=background).to(device).eval()
    front_view = str(data["front_view"])
    back_view = str(data["back_view"])
    missing = {front_view, back_view}.difference(renderer.views)
    if missing:
        raise ValueError(f"Mappings are missing views {sorted(missing)}")

    source_dir = dataset_dir / "source"
    target_dir = dataset_dir / "edited"
    source_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    width = int(data.get("width", 512))
    height = int(data.get("height", 512))
    if (width, height) != (512, 512):
        raise ValueError("The selected 256x512 per-view mappings require a 512x512 merged target")
    source_background = tuple(int(value) for value in data.get("source_background_rgb_255", [255, 255, 255]))
    validation_fraction = float(data.get("validation_fraction", 0.05))
    prompt = str(data["training_prompt"])
    prompt_id = str(data.get("prompt_id", "ddj_reference"))
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []

    for start in tqdm(range(0, len(pairs), batch_size), desc="render strict paired targets"):
        batch = pairs[start : start + batch_size]
        valid: list[tuple[str, Path, Path, Path | None]] = []
        skins: list[torch.Tensor] = []
        for item in batch:
            try:
                skins.append(load_skin(item[2]))
                valid.append(item)
            except Exception as exc:
                skipped.append({"key": item[0], "reason": str(exc)})
        if not valid:
            continue
        skin_batch = torch.stack(skins).to(device)
        with torch.inference_mode():
            front = renderer.forward_view(skin_batch, front_view)[:, :3]
            back = renderer.forward_view(skin_batch, back_view)[:, :3]
            targets = torch.cat([front, back], dim=3).clamp_(0, 1)
        target_arrays = targets.mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        for item, target_array in zip(valid, target_arrays, strict=True):
            stable_key, source_path, result_path, original_edited = item
            sample_id = hashlib.sha1(stable_key.encode("utf-8")).hexdigest()[:16]
            normalized_source_path = source_dir / f"{sample_id}.png"
            target_path = target_dir / f"{sample_id}.png"
            fit_reference_image(source_path, width, height, source_background).save(
                normalized_source_path, optimize=True
            )
            Image.fromarray(target_array, mode="RGB").save(target_path, optimize=True)
            split_value = int(hashlib.sha1((stable_key + ":split").encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            rows.append(
                {
                    "id": sample_id,
                    "source_image": str(normalized_source_path),
                    "image": str(target_path),
                    "source_original": str(source_path),
                    "skin": str(result_path),
                    "historical_edited": str(original_edited) if original_edited else None,
                    "caption": prompt,
                    "prompt_id": prompt_id,
                    "split": "validation" if split_value < validation_fraction else "train",
                    "width": width,
                    "height": height,
                }
            )

    write_jsonl(manifest_path, rows)
    summary = {
        "manifest": str(manifest_path),
        "candidate_pairs": len(pairs),
        "written_pairs": len(rows),
        "train": sum(row["split"] == "train" for row in rows),
        "validation": sum(row["split"] == "validation" for row in rows),
        "skipped": len(skipped),
        "target_source": "64x64 RGBA result skin rendered with fixed mappings",
        "views": [front_view, back_view],
        "background_rgb": list(background),
        "resolution": [width, height],
    }
    write_json(dataset_dir / "summary.json", summary)
    if skipped:
        write_json(dataset_dir / "skipped.json", {"items": skipped})
    print(summary)


if __name__ == "__main__":
    main()
