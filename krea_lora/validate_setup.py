#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from diffusers import AutoencoderKLQwenImage, Krea2Transformer2DModel
from PIL import Image

from common import load_config, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate model, renderer, dataset, and LoRA target configuration.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--require-dataset", action="store_true")
    parser.add_argument("--require-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_path = Path(config["model"]["path"]).expanduser().resolve()
    data = config["data"]
    checks: dict[str, object] = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_path": str(model_path),
    }
    for required in ["model_index.json", "transformer/config.json", "vae/config.json", "scheduler/scheduler_config.json"]:
        path = model_path / required
        if not path.is_file():
            raise FileNotFoundError(path)
    transformer_config = Krea2Transformer2DModel.load_config(model_path, subfolder="transformer")
    vae_config = AutoencoderKLQwenImage.load_config(model_path, subfolder="vae")
    with init_empty_weights():
        empty_model = Krea2Transformer2DModel.from_config(transformer_config)
    targets = set(config["training"]["target_modules"])
    matched = sorted(
        name for name, _module in empty_model.named_modules() if any(name == target or name.endswith(f".{target}") for target in targets)
    )
    if not matched:
        raise RuntimeError(f"No transformer modules matched LoRA targets {sorted(targets)}")
    checks.update(
        {
            "transformer_layers": transformer_config["num_layers"],
            "transformer_in_channels": transformer_config["in_channels"],
            "vae_z_dim": vae_config["z_dim"],
            "lora_target_matches": len(matched),
            "lora_target_examples": matched[:8],
        }
    )

    mappings_dir = Path(data["mappings_dir"]).expanduser().resolve()
    for view in [data["front_view"], data["back_view"]]:
        path = mappings_dir / f"{view}_mapping.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
    checks["mappings_dir"] = str(mappings_dir)

    dataset_dir = Path(data["dataset_dir"]).expanduser().resolve()
    manifest_path = dataset_dir / "metadata.jsonl"
    if args.require_dataset or manifest_path.exists():
        rows = read_jsonl(manifest_path)
        if not rows:
            raise RuntimeError("Dataset manifest is empty")
        with Image.open(rows[0]["image"]) as sample:
            checks["sample_resolution"] = list(sample.size)
            checks["sample_mode"] = sample.mode
        checks["dataset_rows"] = len(rows)
        checks["dataset_train"] = sum(row.get("split") == "train" for row in rows)
        checks["dataset_validation"] = sum(row.get("split") == "validation" for row in rows)
        if args.require_cache:
            if not (dataset_dir / "prompt_cache.safetensors").is_file():
                raise FileNotFoundError(dataset_dir / "prompt_cache.safetensors")
            missing = [row["id"] for row in rows if not (dataset_dir / "latents" / f"{row['id']}.safetensors").is_file()]
            if missing:
                raise RuntimeError(f"Missing {len(missing)} latent cache files; first id: {missing[0]}")
            checks["latent_cache_complete"] = True
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

