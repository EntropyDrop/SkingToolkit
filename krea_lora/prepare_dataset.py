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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render deterministic front/back Minecraft preview targets.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_skin(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        if image.size != (64, 64):
            raise ValueError(f"expected 64x64, found {image.size}")
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    return torch.from_numpy(rgba).permute(2, 0, 1).float().div_(255.0)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data = config["data"]
    seed = int(config["training"]["seed"])
    seed_everything(seed)

    toolkit_dir = Path(data["toolkit_dir"]).expanduser().resolve()
    skins_dir = Path(data["skins_dir"]).expanduser().resolve()
    mappings_dir = Path(data["mappings_dir"]).expanduser().resolve()
    dataset_dir = Path(data["dataset_dir"]).expanduser().resolve()
    image_dir = dataset_dir / "images"
    manifest_path = dataset_dir / "metadata.jsonl"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"{manifest_path} already exists; pass --force to rebuild")
    if not (toolkit_dir / "renderer.py").is_file():
        raise FileNotFoundError(f"Missing renderer.py under {toolkit_dir}")
    if not skins_dir.is_dir():
        raise FileNotFoundError(skins_dir)
    if not mappings_dir.is_dir():
        raise FileNotFoundError(mappings_dir)

    sys.path.insert(0, str(toolkit_dir))
    from renderer import DifferentiableRenderer

    max_images = int(args.max_images or data["max_images"])
    batch_size = int(args.batch_size or data["render_batch_size"])
    device = resolve_device(args.device)
    candidates = sorted(skins_dir.glob("*.png"))
    random.Random(seed).shuffle(candidates)
    candidates = candidates[:max_images]
    if not candidates:
        raise RuntimeError(f"No PNG skins found in {skins_dir}")

    image_dir.mkdir(parents=True, exist_ok=True)
    renderer = DifferentiableRenderer(
        mappings_dir=str(mappings_dir),
        bg_color=tuple(float(value) for value in data["background_rgb"]),
    ).to(device).eval()
    front_view = data["front_view"]
    back_view = data["back_view"]
    missing = {front_view, back_view}.difference(renderer.views)
    if missing:
        raise ValueError(f"Mappings are missing required views: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    validation_fraction = float(data["validation_fraction"])
    prompt = str(data["training_prompt"])
    prompt_id = str(data.get("prompt_id", "mc_preview"))

    for start in tqdm(range(0, len(candidates), batch_size), desc="render targets"):
        paths = candidates[start : start + batch_size]
        valid_paths: list[Path] = []
        tensors: list[torch.Tensor] = []
        for path in paths:
            try:
                tensors.append(load_skin(path))
                valid_paths.append(path)
            except Exception as exc:
                skipped.append({"path": str(path), "reason": str(exc)})
        if not tensors:
            continue
        skins = torch.stack(tensors).to(device)
        with torch.inference_mode():
            front = renderer.forward_view(skins, front_view)[:, :3]
            back = renderer.forward_view(skins, back_view)[:, :3]
            sheets = torch.cat([front, back], dim=3).clamp_(0, 1)
        arrays = sheets.mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        for source_path, array in zip(valid_paths, arrays, strict=True):
            relative_source = source_path.relative_to(skins_dir).as_posix()
            sample_id = hashlib.sha1(relative_source.encode("utf-8")).hexdigest()[:16]
            image_path = image_dir / f"{sample_id}.png"
            Image.fromarray(array, mode="RGB").save(image_path, optimize=True)
            split_value = int(hashlib.sha1((relative_source + ":split").encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            rows.append(
                {
                    "id": sample_id,
                    "image": str(image_path),
                    "source_skin": str(source_path),
                    "caption": prompt,
                    "prompt_id": prompt_id,
                    "split": "validation" if split_value < validation_fraction else "train",
                    "width": int(array.shape[1]),
                    "height": int(array.shape[0]),
                }
            )

    write_jsonl(manifest_path, rows)
    summary = {
        "manifest": str(manifest_path),
        "images": len(rows),
        "train": sum(row["split"] == "train" for row in rows),
        "validation": sum(row["split"] == "validation" for row in rows),
        "skipped": len(skipped),
        "views": [front_view, back_view],
        "mappings_dir": str(mappings_dir),
        "resolution": [rows[0]["width"], rows[0]["height"]] if rows else None,
    }
    write_json(dataset_dir / "summary.json", summary)
    if skipped:
        write_json(dataset_dir / "skipped.json", {"items": skipped})
    print(summary)


if __name__ == "__main__":
    main()

