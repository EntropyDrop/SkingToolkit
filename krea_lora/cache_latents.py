#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLQwenImage
from PIL import Image
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from common import load_config, pack_latents, read_jsonl, resolve_device, resolve_dtype, write_json


class ImageDataset(Dataset):
    def __init__(self, rows: list[dict], limit: int | None = None):
        self.rows = rows[:limit] if limit else rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor]:
        row = self.rows[index]
        with Image.open(row["image"]) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(127.5).sub_(1.0)
        return str(row["id"]), tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache normalized, packed Qwen Image VAE latents.")
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
    dataset = ImageDataset(rows, args.limit)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    vae = AutoencoderKLQwenImage.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device).eval()
    latents_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(1, vae.config.z_dim, 1, 1, 1)
    latents_std_inv = torch.tensor(vae.config.latents_std, device=device, dtype=dtype).reciprocal().view(
        1, vae.config.z_dim, 1, 1, 1
    )
    patch_size = int(config["model"].get("patch_size", 2))
    cached = 0
    skipped = 0
    latent_shape: tuple[int, ...] | None = None
    with torch.inference_mode():
        for sample_ids, pixels in tqdm(loader, desc="cache VAE latents"):
            destinations = [cache_dir / f"{sample_id}.safetensors" for sample_id in sample_ids]
            if not args.force and all(path.exists() for path in destinations):
                skipped += len(destinations)
                continue
            pixels = pixels.to(device=device, dtype=dtype)
            # AutoencoderKLQwenImage is a video-shaped VAE even for still images:
            # B,C,H,W -> B,C,T=1,H,W.
            posterior = vae.encode(pixels.unsqueeze(2)).latent_dist
            raw_latents = posterior.sample() if args.sample_posterior else posterior.mode()
            normalized = (raw_latents - latents_mean) * latents_std_inv
            if normalized.ndim != 5 or normalized.shape[2] != 1:
                raise ValueError(f"Unexpected Qwen VAE latent shape: {tuple(normalized.shape)}")
            packed = pack_latents(normalized[:, :, 0], patch_size=patch_size).cpu().contiguous()
            latent_shape = tuple(packed.shape[1:])
            for destination, latent in zip(destinations, packed, strict=True):
                if destination.exists() and not args.force:
                    skipped += 1
                    continue
                save_file({"latents": latent}, destination)
                cached += 1
    write_json(
        dataset_dir / "latent_cache.json",
        {
            "model_path": str(model_path),
            "items_seen": len(dataset),
            "cached": cached,
            "skipped_existing": skipped,
            "packed_latent_shape": list(latent_shape) if latent_shape else None,
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
