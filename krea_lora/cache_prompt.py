#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
from diffusers import Krea2Pipeline
from safetensors.torch import save_file

from common import load_config, prompt_cache_key, read_jsonl, resolve_device, resolve_dtype, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen Krea2/Qwen3-VL prompt embeddings.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_path = Path(config["model"]["path"]).expanduser().resolve()
    dataset_dir = Path(config["data"]["dataset_dir"]).expanduser().resolve()
    manifest = read_jsonl(dataset_dir / "metadata.jsonl")
    output_path = dataset_dir / "prompt_cache.safetensors"
    if output_path.exists() and not args.force:
        print(f"Prompt cache already exists: {output_path}")
        return

    prompts: dict[str, str] = {}
    for row in manifest:
        prompt_id = str(row["prompt_id"])
        caption = str(row["caption"])
        if prompt_id in prompts and prompts[prompt_id] != caption:
            raise ValueError(f"prompt_id {prompt_id!r} maps to more than one caption")
        prompts[prompt_id] = caption
    if not prompts:
        raise RuntimeError("Manifest contains no prompts")

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
    tensors: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for prompt_id, caption in prompts.items():
            embeds, mask = pipe.encode_prompt(
                caption,
                device=device,
                max_sequence_length=max_sequence_length,
            )
            tensors[prompt_cache_key("embeds", prompt_id)] = embeds.cpu().contiguous()
            tensors[prompt_cache_key("mask", prompt_id)] = mask.cpu().contiguous()
            print(f"cached {prompt_id}: embeds={tuple(embeds.shape)}, tokens={int(mask.sum())}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, output_path)
    write_json(
        dataset_dir / "prompt_cache.json",
        {
            "model_path": str(model_path),
            "max_sequence_length": max_sequence_length,
            "prompt_ids": sorted(prompts),
            "dtype": str(dtype),
        },
    )
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(output_path)


if __name__ == "__main__":
    main()

