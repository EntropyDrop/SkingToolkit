from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, Krea2Pipeline
from diffusers.pipelines.krea2.pipeline_krea2 import calculate_shift, retrieve_timesteps
from PIL import Image

from reference_conditioning import (
    denormalize_qwen_vae_latents,
    encode_qwen_vae_images,
    fit_reference_image,
    image_to_normalized_tensor,
)


CHECKPOINT_TEST_IMAGES_ENV = "KREA_CHECKPOINT_TEST_IMAGES"


def checkpoint_test_image_paths(value: str | None = None) -> list[Path]:
    """Resolve colon-separated checkpoint test images from one environment variable."""
    raw_value = os.environ.get(CHECKPOINT_TEST_IMAGES_ENV, "") if value is None else value
    paths: list[Path] = []
    seen: set[Path] = set()
    for item in raw_value.split(os.pathsep):
        if not item.strip():
            continue
        path = Path(item.strip()).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{CHECKPOINT_TEST_IMAGES_ENV} image does not exist: {path}")
        if path not in seen:
            paths.append(path)
            seen.add(path)
    return paths


def unpack_latents(
    latents: torch.Tensor,
    height: int,
    width: int,
    vae_scale_factor: int,
    patch_size: int,
) -> torch.Tensor:
    """Inverse of Krea2/Qwen Image latent packing without constructing a full pipeline."""
    batch_size, _, channels = latents.shape
    latent_height = patch_size * (height // (vae_scale_factor * patch_size))
    latent_width = patch_size * (width // (vae_scale_factor * patch_size))
    if channels % (patch_size * patch_size):
        raise ValueError(f"Packed latent channels {channels} are incompatible with patch size {patch_size}")
    unpacked = latents.view(
        batch_size,
        latent_height // patch_size,
        latent_width // patch_size,
        channels // (patch_size * patch_size),
        patch_size,
        patch_size,
    )
    unpacked = unpacked.permute(0, 3, 1, 4, 2, 5)
    return unpacked.reshape(
        batch_size,
        channels // (patch_size * patch_size),
        1,
        latent_height,
        latent_width,
    )


def decoded_tensor_to_pil(decoded: torch.Tensor) -> Image.Image:
    pixels = decoded[0].float().div(2).add(0.5).clamp(0, 1)
    array = pixels.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.rint(array * 255).astype(np.uint8), mode="RGB")


def safe_output_stem(path: Path, index: int) -> str:
    stem = re.sub(r"[^0-9A-Za-z._-]+", "_", path.stem).strip("._-") or "image"
    return f"{index:02d}_{stem[:80]}"


@dataclass
class EncodedTestImage:
    source_path: Path
    output_stem: str
    reference: Image.Image
    source_latents: torch.Tensor


class CheckpointPreviewer:
    """Generate deterministic source-bridge samples from the in-memory training LoRA."""

    def __init__(
        self,
        *,
        image_paths: list[Path],
        model_path: Path,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        width: int,
        height: int,
        vae_scale_factor: int,
        patch_size: int,
        steps: int,
        seed: int,
        source_noise_strength: float,
    ) -> None:
        if steps < 1:
            raise ValueError("Checkpoint preview steps must be at least 1")
        if not 0.0 <= source_noise_strength <= 1.0:
            raise ValueError("Checkpoint preview source_noise_strength must be between 0 and 1")
        self.device = device
        self.dtype = dtype
        self.width = width
        self.height = height
        self.vae_scale_factor = vae_scale_factor
        self.patch_size = patch_size
        self.steps = steps
        self.seed = seed
        self.source_noise_strength = source_noise_strength
        self.prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        self.prompt_mask = prompt_mask.to(device=device)
        self.position_ids = Krea2Pipeline.prepare_position_ids(
            self.prompt_embeds.shape[1],
            height // (vae_scale_factor * patch_size),
            width // (vae_scale_factor * patch_size),
            device,
        )
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path,
            subfolder="scheduler",
            local_files_only=True,
        )
        self.vae = AutoencoderKLQwenImage.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=dtype,
            local_files_only=True,
        ).to(device).eval()
        self.vae.requires_grad_(False)
        self.images: list[EncodedTestImage] = []
        with torch.inference_mode():
            for index, source_path in enumerate(image_paths):
                reference = fit_reference_image(source_path, width, height, (255, 255, 255))
                pixels = image_to_normalized_tensor(reference).unsqueeze(0).to(
                    device=device,
                    dtype=self.vae.dtype,
                )
                source_latents = encode_qwen_vae_images(
                    self.vae,
                    pixels,
                    patch_size=patch_size,
                    sample_posterior=False,
                ).to(dtype=dtype)
                self.images.append(
                    EncodedTestImage(
                        source_path=source_path,
                        output_stem=safe_output_stem(source_path, index),
                        reference=reference,
                        source_latents=source_latents,
                    )
                )

    @torch.inference_mode()
    def save(self, transformer: torch.nn.Module, output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        was_training = transformer.training
        transformer.eval()
        generated_paths: list[Path] = []
        try:
            for index, item in enumerate(self.images):
                item.reference.save(output_dir / f"{item.output_stem}_reference.png", optimize=True)
                latents = item.source_latents.clone()
                if self.source_noise_strength > 0:
                    generator = torch.Generator(device=self.device).manual_seed(self.seed + index)
                    source_noise = torch.randn(
                        latents.shape,
                        generator=generator,
                        device=self.device,
                        dtype=latents.dtype,
                    )
                    latents = (
                        (1.0 - self.source_noise_strength) * latents
                        + self.source_noise_strength * source_noise
                    )
                sigmas = np.linspace(1.0, 1.0 / self.steps, self.steps)
                mu = calculate_shift(
                    latents.shape[1],
                    self.scheduler.config.get("base_image_seq_len", 256),
                    self.scheduler.config.get("max_image_seq_len", 6400),
                    self.scheduler.config.get("base_shift", 0.5),
                    self.scheduler.config.get("max_shift", 1.15),
                )
                timesteps, _ = retrieve_timesteps(
                    self.scheduler,
                    self.steps,
                    self.device,
                    sigmas=sigmas,
                    mu=mu,
                )
                self.scheduler.set_begin_index(0)
                for timestep_value in timesteps:
                    timestep = (
                        timestep_value / self.scheduler.config.num_train_timesteps
                    ).expand(1).to(self.dtype)
                    prediction = transformer(
                        hidden_states=latents,
                        encoder_hidden_states=self.prompt_embeds,
                        timestep=timestep,
                        position_ids=self.position_ids,
                        encoder_attention_mask=self.prompt_mask,
                        return_dict=False,
                    )[0]
                    latents = self.scheduler.step(
                        prediction,
                        timestep_value,
                        latents,
                        return_dict=False,
                    )[0]
                unpacked = unpack_latents(
                    latents,
                    self.height,
                    self.width,
                    self.vae_scale_factor,
                    self.patch_size,
                ).to(self.vae.dtype)
                raw_latents = denormalize_qwen_vae_latents(self.vae, unpacked)
                decoded = self.vae.decode(raw_latents, return_dict=False)[0][:, :, 0]
                generated_path = output_dir / f"{item.output_stem}_generated.png"
                decoded_tensor_to_pil(decoded).save(generated_path, optimize=True)
                generated_paths.append(generated_path)
        finally:
            transformer.train(was_training)
        return generated_paths

