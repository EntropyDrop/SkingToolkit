#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
from diffusers import AutoencoderKLQwenImage
from PIL import Image
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from common import load_config, read_jsonl, resolve_device, resolve_dtype, write_json
from reference_conditioning import encode_qwen_vae_images, image_to_normalized_tensor


class PairDataset(Dataset):
    def __init__(self, rows: list[dict], limit: int | None):
        self.rows = rows[:limit] if limit else rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        with Image.open(row["image"]) as image:
            target = image_to_normalized_tensor(image)
        with Image.open(row["source_image"]) as image:
            source = image_to_normalized_tensor(image)
        return str(row["id"]), target, source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache target and clean-reference Qwen Image VAE latents.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sample-posterior", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_path = Path(config["model"]["path"]).expanduser().resolve()
    dataset_dir = Path(config["data"]["dataset_dir"]).expanduser().resolve()
    rows = read_jsonl(dataset_dir / "metadata.jsonl")
    cache_dir = dataset_dir / "latents"
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    dtype = resolve_dtype(config["model"].get("dtype", "bf16"))
    batch_size = int(args.batch_size or config["cache"].get("vae_batch_size", 1))
    dataset = PairDataset(rows, args.limit)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    vae = AutoencoderKLQwenImage.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device).eval()
    patch_size = int(config["model"].get("patch_size", 2))
    cached = 0
    skipped = 0
    latent_shape: tuple[int, ...] | None = None
    with torch.inference_mode():
        for sample_ids, targets, sources in tqdm(loader, desc="cache paired VAE latents"):
            destinations = [cache_dir / f"{sample_id}.safetensors" for sample_id in sample_ids]
            if not args.force and all(path.exists() for path in destinations):
                skipped += len(destinations)
                continue
            pixels = torch.cat([targets, sources], dim=0).to(device=device, dtype=dtype)
            packed = encode_qwen_vae_images(
                vae,
                pixels,
                patch_size=patch_size,
                sample_posterior=args.sample_posterior,
            ).cpu().contiguous()
            target_latents, source_latents = packed.chunk(2, dim=0)
            latent_shape = tuple(target_latents.shape[1:])
            for destination, target_latent, source_latent in zip(
                destinations, target_latents, source_latents, strict=True
            ):
                if destination.exists() and not args.force:
                    skipped += 1
                    continue
                save_file(
                    {
                        "target_latents": target_latent.contiguous(),
                        "source_latents": source_latent.contiguous(),
                    },
                    destination,
                )
                cached += 1
    write_json(
        dataset_dir / "latent_cache.json",
        {
            "model_path": str(model_path),
            "items_seen": len(dataset),
            "cached": cached,
            "skipped_existing": skipped,
            "packed_latent_shape_each": list(latent_shape) if latent_shape else None,
            "sequence_schema": "target_then_reference",
            "posterior": "sample" if args.sample_posterior else "mode",
            "dtype": str(dtype),
        },
    )
    del vae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print({"cached": cached, "skipped": skipped, "shape": latent_shape})


if __name__ == "__main__":
    main()
