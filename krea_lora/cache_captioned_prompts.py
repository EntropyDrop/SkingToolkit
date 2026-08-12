#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
from diffusers import Krea2Pipeline
from safetensors.torch import save_file
from tqdm.auto import tqdm

from common import load_config, read_jsonl, resolve_device, resolve_dtype, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache sharded per-image Krea prompt embeddings.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def cache_group(
    pipe: Krea2Pipeline,
    rows: list[dict],
    output_dir: Path,
    max_sequence_length: int,
    force: bool,
    label: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cached = 0
    skipped = 0
    total_bytes = 0
    lengths: list[int] = []
    with torch.inference_mode():
        for row in tqdm(rows, desc=label):
            destination = output_dir / f"{row['prompt_id']}.safetensors"
            if destination.is_file() and not force:
                skipped += 1
                total_bytes += destination.stat().st_size
                continue
            embeds, mask = pipe.encode_prompt(
                str(row["caption"]),
                device=pipe._execution_device,
                max_sequence_length=max_sequence_length,
            )
            active = mask[0].bool()
            active_length = int(active.sum())
            if active_length >= max_sequence_length:
                raise ValueError(
                    f"Prompt {row['prompt_id']} fills all {max_sequence_length} tokens and may be truncated"
                )
            # Text position IDs are all zero in Krea2. Removing masked padding
            # preserves the active tokens while saving substantial disk and I/O.
            trimmed_embeds = embeds[0, active].cpu().contiguous()
            trimmed_mask = torch.ones(active_length, dtype=torch.bool)
            save_file(
                {
                    "embeds": trimmed_embeds,
                    "mask": trimmed_mask,
                },
                destination,
                metadata={
                    "prompt_id": str(row["prompt_id"]),
                    "active_tokens": str(active_length),
                },
            )
            cached += 1
            lengths.append(active_length)
            total_bytes += destination.stat().st_size
    return {
        "rows": len(rows),
        "cached": cached,
        "skipped": skipped,
        "bytes": total_bytes,
        "new_token_length_min": min(lengths) if lengths else None,
        "new_token_length_max": max(lengths) if lengths else None,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_path = Path(config["model"]["path"]).expanduser().resolve()
    dataset_dir = Path(config["data"]["dataset_dir"]).expanduser().resolve()
    train_rows = read_jsonl(dataset_dir / "metadata.jsonl")
    if args.limit:
        train_rows = train_rows[: args.limit]
    checkpoint_rows = read_jsonl(dataset_dir / "checkpoint_metadata.jsonl")
    train_dir = dataset_dir / "prompt_cache"
    checkpoint_dir = dataset_dir / "checkpoint_prompt_cache"
    missing = [
        row
        for row, directory in [(row, checkpoint_dir) for row in checkpoint_rows]
        + [(row, train_dir) for row in train_rows]
        if args.force or not (directory / f"{row['prompt_id']}.safetensors").is_file()
    ]
    if not missing:
        print("All sharded prompt embeddings already exist")
        return

    device = resolve_device(args.device)
    dtype = resolve_dtype(config["model"].get("dtype", "bf16"))
    pipe = Krea2Pipeline.from_pretrained(
        model_path,
        transformer=None,
        vae=None,
        torch_dtype=dtype,
        local_files_only=True,
    )
    pipe.to(device)
    max_sequence_length = int(config["model"].get("max_sequence_length", 512))
    checkpoint_summary = cache_group(
        pipe,
        checkpoint_rows,
        checkpoint_dir,
        max_sequence_length,
        args.force,
        "cache checkpoint prompts",
    )
    train_summary = cache_group(
        pipe,
        train_rows,
        train_dir,
        max_sequence_length,
        args.force,
        "cache per-image prompts",
    )
    write_json(
        dataset_dir / "prompt_cache.json",
        {
            "model_path": str(model_path),
            "max_sequence_length": max_sequence_length,
            "dtype": str(dtype),
            "storage": "one safetensors file per prompt, masked padding removed",
            "train": train_summary,
            "checkpoint": checkpoint_summary,
        },
    )
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print({"train": train_summary, "checkpoint": checkpoint_summary})


if __name__ == "__main__":
    main()
