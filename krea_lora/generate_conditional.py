#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from diffusers import Krea2Pipeline
from diffusers.pipelines.krea2.pipeline_krea2 import calculate_shift, retrieve_timesteps
from tqdm.auto import tqdm

from common import load_config, resolve_dtype
from reference_conditioning import (
    denormalize_qwen_vae_latents,
    encode_qwen_vae_images,
    fit_reference_image,
    image_to_normalized_tensor,
    prepare_paired_position_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a strict MC preview conditioned on an arbitrary image.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--source", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--lora", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.inference_mode()
def generate(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    model_config = config["model"]
    training = config["training"]
    inference = config["inference"]
    conditioning_mode = str(training.get("conditioning_mode", "source_bridge"))
    if conditioning_mode not in {"source_bridge", "target_reference_concat"}:
        raise ValueError(f"Unsupported conditioning_mode: {conditioning_mode}")
    model_path = Path(model_config["path"]).expanduser().resolve()
    lora_path = Path(args.lora or (Path(training["output_dir"]) / "final")).expanduser().resolve()
    output_path = Path(args.output or inference["output_path"]).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dtype = resolve_dtype(model_config.get("dtype", "bf16"))
    device = torch.device(args.device)
    pipe = Krea2Pipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        local_files_only=True,
    )
    pipe.load_lora_weights(lora_path)
    if bool(inference.get("layerwise_casting", False)):
        pipe.transformer.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn,
            compute_dtype=dtype,
        )
    pipe.to(device)

    width = int(inference.get("width", 512))
    height = int(inference.get("height", 512))
    reference = fit_reference_image(args.source, width, height, (255, 255, 255))
    reference.save(output_path.with_name(output_path.stem + "_reference.png"), optimize=True)
    reference_pixels = image_to_normalized_tensor(reference).unsqueeze(0).to(device=device, dtype=pipe.vae.dtype)
    source_latents = encode_qwen_vae_images(
        pipe.vae,
        reference_pixels,
        patch_size=int(model_config.get("patch_size", 2)),
        sample_posterior=False,
    ).to(dtype=dtype)

    base_prompt = str(inference["base_prompt"])
    prompt = base_prompt if not args.prompt.strip() else f"{base_prompt} Character details: {args.prompt.strip()}"
    max_sequence_length = int(model_config.get("max_sequence_length", 512))
    prompt_embeds, prompt_mask = pipe.encode_prompt(
        prompt,
        device=device,
        max_sequence_length=max_sequence_length,
    )
    negative_embeds, negative_mask = pipe.encode_prompt(
        str(inference.get("negative_prompt", "")),
        device=device,
        max_sequence_length=max_sequence_length,
    )
    seed = int(args.seed if args.seed is not None else inference.get("seed", 20260811))
    generator = torch.Generator(device=device).manual_seed(seed)
    if conditioning_mode == "source_bridge":
        source_noise_strength = float(inference.get("source_noise_strength", 0.0))
        if not 0.0 <= source_noise_strength <= 1.0:
            raise ValueError("inference.source_noise_strength must be between 0 and 1")
        target_latents = source_latents.clone()
        if source_noise_strength > 0:
            source_noise = torch.randn(
                source_latents.shape,
                generator=generator,
                device=device,
                dtype=source_latents.dtype,
            )
            target_latents = (
                (1.0 - source_noise_strength) * source_latents
                + source_noise_strength * source_noise
            )
    else:
        num_channels = pipe.transformer.config.in_channels // (pipe.patch_size**2)
        target_latents = pipe.prepare_latents(
            1,
            num_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )
        if source_latents.shape != target_latents.shape:
            raise ValueError(
                f"Source/target packed latent shapes differ: {source_latents.shape} vs {target_latents.shape}"
            )
    grid_height = height // (pipe.vae_scale_factor * pipe.patch_size)
    grid_width = width // (pipe.vae_scale_factor * pipe.patch_size)
    if conditioning_mode == "source_bridge":
        position_ids = Krea2Pipeline.prepare_position_ids(
            prompt_embeds.shape[1],
            grid_height,
            grid_width,
            device,
        )
    else:
        position_ids = prepare_paired_position_ids(prompt_embeds.shape[1], grid_height, grid_width, device)

    steps = int(args.steps or inference.get("steps", 28))
    guidance_scale = float(
        args.guidance_scale if args.guidance_scale is not None else inference.get("guidance_scale", 4.5)
    )
    sigmas = np.linspace(1.0, 1.0 / steps, steps)
    target_seq_len = target_latents.shape[1]
    mu = calculate_shift(
        target_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 6400),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, _ = retrieve_timesteps(pipe.scheduler, steps, device, sigmas=sigmas, mu=mu)
    pipe.scheduler.set_begin_index(0)
    for timestep_value in tqdm(timesteps, desc="conditional denoise"):
        timestep = (timestep_value / pipe.scheduler.config.num_train_timesteps).expand(1).to(dtype)
        model_input = (
            target_latents
            if conditioning_mode == "source_bridge"
            else torch.cat([target_latents, source_latents], dim=1)
        )
        conditional_full = pipe.transformer(
            hidden_states=model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            position_ids=position_ids,
            encoder_attention_mask=prompt_mask,
            attention_kwargs={"scale": float(args.lora_scale)},
            return_dict=False,
        )[0]
        prediction = (
            conditional_full
            if conditioning_mode == "source_bridge"
            else conditional_full[:, :target_seq_len]
        )
        if guidance_scale > 0:
            negative_full = pipe.transformer(
                hidden_states=model_input,
                encoder_hidden_states=negative_embeds,
                timestep=timestep,
                position_ids=position_ids,
                encoder_attention_mask=negative_mask,
                attention_kwargs={"scale": float(args.lora_scale)},
                return_dict=False,
            )[0]
            negative_prediction = (
                negative_full
                if conditioning_mode == "source_bridge"
                else negative_full[:, :target_seq_len]
            )
            prediction = prediction + guidance_scale * (prediction - negative_prediction)
        target_latents = pipe.scheduler.step(
            prediction,
            timestep_value,
            target_latents,
            return_dict=False,
        )[0]

    unpacked = pipe._unpack_latents(target_latents, height, width).to(pipe.vae.dtype)
    raw_latents = denormalize_qwen_vae_latents(pipe.vae, unpacked)
    decoded = pipe.vae.decode(raw_latents, return_dict=False)[0][:, :, 0]
    image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
    image.save(output_path)
    print(output_path)
    return output_path


if __name__ == "__main__":
    generate(parse_args())
