#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
from safetensors.torch import load_file

from common import load_config, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate paired manifest and conditional latent caches.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--require-cache", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    dataset_dir = Path(config["data"]["dataset_dir"]).expanduser().resolve()
    rows = read_jsonl(dataset_dir / "metadata.jsonl")
    if not rows:
        raise RuntimeError("Paired manifest is empty")
    required_fields = {"id", "source_image", "image", "skin", "prompt_id", "caption", "split"}
    missing_fields = required_fields.difference(rows[0])
    if missing_fields:
        raise ValueError(f"Manifest is missing fields {sorted(missing_fields)}")
    with Image.open(rows[0]["source_image"]) as source:
        source_size, source_mode = source.size, source.mode
    with Image.open(rows[0]["image"]) as target:
        target_size, target_mode = target.size, target.mode
    with Image.open(rows[0]["skin"]) as skin:
        skin_size, skin_mode = skin.size, skin.mode
    report = {
        "pairs": len(rows),
        "train": sum(row["split"] == "train" for row in rows),
        "validation": sum(row["split"] == "validation" for row in rows),
        "source": {"size": source_size, "mode": source_mode},
        "target": {"size": target_size, "mode": target_mode},
        "skin": {"size": skin_size, "mode": skin_mode},
    }
    if source_size != target_size:
        raise ValueError(f"Source/target image sizes differ: {source_size} vs {target_size}")
    if skin_size != (64, 64) or skin_mode != "RGBA":
        raise ValueError(f"Expected a 64x64 RGBA skin, found {skin_size} {skin_mode}")
    if args.require_cache:
        prompt_cache = dataset_dir / "prompt_cache.safetensors"
        if not prompt_cache.is_file():
            raise FileNotFoundError(prompt_cache)
        shapes = set()
        missing = 0
        for row in rows:
            latent_path = dataset_dir / "latents" / f"{row['id']}.safetensors"
            if not latent_path.is_file():
                missing += 1
                continue
            tensors = load_file(latent_path, device="cpu")
            if set(tensors) != {"source_latents", "target_latents"}:
                raise ValueError(f"Unexpected latent keys in {latent_path}: {sorted(tensors)}")
            if tensors["source_latents"].shape != tensors["target_latents"].shape:
                raise ValueError(f"Source/target latent shape mismatch in {latent_path}")
            shapes.add(tuple(tensors["target_latents"].shape))
        if missing:
            raise RuntimeError(f"Missing {missing} paired latent files")
        report["latent_shapes"] = sorted(shapes)
        report["latent_cache_complete"] = True
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
