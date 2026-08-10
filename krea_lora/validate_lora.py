#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import Krea2Pipeline

from common import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed-seed Krea2 LoRA validation images.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--lora", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_path = Path(config["model"]["path"]).expanduser().resolve()
    train_config = config["training"]
    validation = config["validation"]
    lora_path = Path(args.lora or (Path(train_config["output_dir"]) / "final")).expanduser().resolve()
    output_dir = Path(args.output_dir or validation["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = args.prompt or validation["prompts"]
    dtype = torch.bfloat16 if config["model"].get("dtype", "bf16") == "bf16" else torch.float16
    pipe = Krea2Pipeline.from_pretrained(model_path, torch_dtype=dtype, local_files_only=True)
    pipe.load_lora_weights(lora_path)
    pipe.to("cuda")
    seed = int(args.seed if args.seed is not None else validation["seed"])
    steps = int(args.steps or validation.get("steps", 28))
    for index, prompt in enumerate(prompts):
        generator = torch.Generator(device="cuda").manual_seed(seed + index)
        image = pipe(
            prompt=prompt,
            negative_prompt=str(validation.get("negative_prompt", "")),
            height=int(validation["height"]),
            width=int(validation["width"]),
            num_inference_steps=steps,
            guidance_scale=float(validation.get("guidance_scale", 4.5)),
            max_sequence_length=int(config["model"].get("max_sequence_length", 512)),
            generator=generator,
        ).images[0]
        destination = output_dir / f"validation_{index:02d}_seed_{seed + index}.png"
        image.save(destination)
        print(destination)


if __name__ == "__main__":
    main()

