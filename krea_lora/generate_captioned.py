#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from diffusers import Krea2Pipeline
from diffusers.pipelines.krea2.pipeline_krea2 import calculate_shift
from PIL import Image, ImageOps

from caption_prompt import build_captioned_prompt
from common import load_config, resolve_dtype, write_json
from image_postprocess import minecraft_crisp_postprocess
from qwen_captioner import QwenCaptioner
from reference_conditioning import normalize_qwen_vae_latents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen-captioned Krea Raw LoRA generation/img2img.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--source", required=True)
    parser.add_argument("--description", default=None, help="Skip Qwen and use this precomputed identity caption.")
    parser.add_argument("--description-suffix", default=None, help="Append user-supplied identity details to Qwen.")
    parser.add_argument("--format-prompt", default=None, help="Override the editable MC format prompt.")
    parser.add_argument("--lora", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--mode", choices=["txt2img", "img2img"], default=None)
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument(
        "--strengths",
        default=None,
        help="Comma-separated Img2Img strengths to compare while reusing one loaded pipeline.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.inference_mode()
def prepare_img2img_latents(
    pipe: Krea2Pipeline,
    image: Image.Image,
    width: int,
    height: int,
    steps: int,
    strength: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, list[float]]:
    if not 0.05 <= strength <= 1.0:
        raise ValueError("Img2Img strength must be between 0.05 and 1.0")
    device = pipe._execution_device
    fitted = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
    pixels = pipe.image_processor.preprocess(fitted, height=height, width=width)
    pixels = pixels.to(device=device, dtype=pipe.vae.dtype).unsqueeze(2)
    raw_latents = pipe.vae.encode(pixels).latent_dist.sample(generator=generator)
    normalized = normalize_qwen_vae_latents(pipe.vae, raw_latents)[:, :, 0]
    packed = pipe._pack_latents(
        normalized,
        normalized.shape[0],
        normalized.shape[1],
        normalized.shape[2],
        normalized.shape[3],
    )

    total_steps = steps if strength >= 1.0 else max(steps, int(steps / strength))
    start_index = total_steps - steps
    raw_sigmas = np.linspace(1.0, 1.0 / total_steps, total_steps, dtype=np.float32)
    mu = calculate_shift(
        packed.shape[1],
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 6400),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    pipe.scheduler.set_timesteps(sigmas=raw_sigmas, device=device, mu=mu)
    timesteps = pipe.scheduler.timesteps[start_index:]
    noise = torch.randn(packed.shape, generator=generator, device=device, dtype=packed.dtype)
    packed = pipe.scheduler.scale_noise(packed, timesteps[:1].repeat(packed.shape[0]), noise)
    return packed, raw_sigmas[start_index:].tolist()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    inference = config["inference"]
    source_path = Path(args.source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    with Image.open(source_path) as opened:
        source = ImageOps.exif_transpose(opened).convert("RGB")

    if args.description:
        description = " ".join(args.description.strip().split())
    else:
        captioner = QwenCaptioner(config["captioning"], device=args.device)
        try:
            description = captioner.describe(source)
        finally:
            captioner.close()
    if args.description_suffix:
        description = f"{description} {' '.join(args.description_suffix.strip().split())}".strip()
    if args.format_prompt:
        format_prompt = " ".join(args.format_prompt.strip().split())
        if not format_prompt:
            raise ValueError("--format-prompt is empty")
        prompt = f"{format_prompt} {str(config['prompt']['identity_prefix']).strip()} {description}"
    else:
        prompt = build_captioned_prompt(config, description)
    print(f"Qwen description: {description}")

    model_path = Path(config["model"]["path"]).expanduser().resolve()
    lora_path = Path(args.lora or (Path(config["training"]["output_dir"]) / "final")).expanduser().resolve()
    output_path = Path(args.output or inference["output_path"]).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dtype = resolve_dtype(config["model"].get("dtype", "bf16"))
    pipe = Krea2Pipeline.from_pretrained(model_path, torch_dtype=dtype, local_files_only=True)
    pipe.load_lora_weights(lora_path)
    if bool(inference.get("layerwise_casting", False)):
        pipe.transformer.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn,
            compute_dtype=dtype,
        )
    pipe.to(args.device)
    pipe.set_progress_bar_config(disable=False)

    mode = str(args.mode or inference.get("mode", "img2img"))
    width = int(inference.get("width", 512))
    height = int(inference.get("height", 512))
    steps = int(args.steps or inference.get("steps", 28))
    strength = float(args.strength if args.strength is not None else inference.get("strength", 0.9))
    if args.strengths:
        if mode != "img2img":
            raise ValueError("--strengths is only supported with --mode img2img")
        strengths = [float(value.strip()) for value in args.strengths.split(",") if value.strip()]
        if not strengths:
            raise ValueError("--strengths did not contain any values")
    else:
        strengths = [strength]
    guidance_scale = float(
        args.guidance_scale if args.guidance_scale is not None else inference.get("guidance_scale", 0.0)
    )
    seed = int(args.seed if args.seed is not None else inference.get("seed", 20260812))
    for current_strength in strengths:
        generator = torch.Generator(device=args.device).manual_seed(seed)
        call_kwargs: dict = {}
        if mode == "img2img":
            latents, sigmas = prepare_img2img_latents(
                pipe,
                source,
                width,
                height,
                steps,
                current_strength,
                generator,
            )
            call_kwargs.update(latents=latents, sigmas=sigmas)
        elif mode != "txt2img":
            raise ValueError(f"Unsupported inference mode: {mode}")

        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                negative_prompt=str(inference.get("negative_prompt", "")),
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                max_sequence_length=int(config["model"].get("max_sequence_length", 512)),
                **call_kwargs,
            ).images[0]
        white_background_threshold = int(inference.get("white_background_threshold", 250))
        crisp_enabled = bool(inference.get("crisp_postprocess", True))
        sharpen_radius = float(inference.get("sharpen_radius", 0.6))
        sharpen_percent = int(inference.get("sharpen_percent", 80))
        sharpen_threshold = int(inference.get("sharpen_threshold", 3))
        contrast = float(inference.get("contrast", 1.16))
        saturation = float(inference.get("saturation", 1.10))
        posterize_bits = int(inference.get("posterize_bits", 4))
        image = minecraft_crisp_postprocess(
            image,
            enabled=crisp_enabled,
            white_threshold=white_background_threshold,
            sharpen_radius=sharpen_radius,
            sharpen_percent=sharpen_percent,
            sharpen_threshold=sharpen_threshold,
            contrast=contrast,
            saturation=saturation,
            posterize_bits=posterize_bits,
        )
        if len(strengths) > 1:
            suffix = f"_strength_{current_strength:.2f}".replace(".", "p")
            current_output = output_path.with_name(f"{output_path.stem}{suffix}{output_path.suffix}")
        else:
            current_output = output_path
        image.save(current_output)
        write_json(
            current_output.with_suffix(".json"),
            {
                "source_image": str(source_path),
                "output_image": str(current_output),
                "lora": str(lora_path),
                "mode": mode,
                "strength": current_strength if mode == "img2img" else None,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "qwen_description": description,
                "prompt": prompt,
                "white_background_threshold": white_background_threshold,
                "crisp_postprocess": crisp_enabled,
                "sharpen_radius": sharpen_radius,
                "sharpen_percent": sharpen_percent,
                "sharpen_threshold": sharpen_threshold,
                "contrast": contrast,
                "saturation": saturation,
                "posterize_bits": posterize_bits,
            },
        )
        print(current_output)
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
