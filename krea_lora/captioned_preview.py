from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, Krea2Pipeline
from diffusers.pipelines.krea2.pipeline_krea2 import calculate_shift, retrieve_timesteps
from PIL import Image, ImageOps
from safetensors.torch import load_file

from checkpoint_preview import decoded_tensor_to_pil, safe_output_stem, unpack_latents
from common import pack_latents
from image_postprocess import snap_near_white_to_white
from reference_conditioning import denormalize_qwen_vae_latents, image_to_normalized_tensor, normalize_qwen_vae_latents


@dataclass
class CaptionedPreviewItem:
    source_path: Path
    output_stem: str
    reference: Image.Image
    description: str
    prompt_id: str
    embeds: torch.Tensor
    mask: torch.Tensor
    source_latents: torch.Tensor
    seed: int


class CaptionedCheckpointPreviewer:
    """Qwen-caption-conditioned Krea text-to-image/img2img checkpoint evaluator."""

    def __init__(
        self,
        *,
        rows: list[dict],
        prompt_cache_dir: Path,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype,
        width: int,
        height: int,
        vae_scale_factor: int,
        patch_size: int,
        steps: int,
        seed: int,
        mode: str,
        strength: float,
        white_background_threshold: int,
    ) -> None:
        if mode not in {"txt2img", "img2img"}:
            raise ValueError(f"Unsupported checkpoint preview mode: {mode}")
        if steps < 1:
            raise ValueError("Checkpoint preview steps must be at least 1")
        if not 0.05 <= strength <= 1.0:
            raise ValueError("Checkpoint preview strength must be between 0.05 and 1.0")
        self.device = device
        self.dtype = dtype
        self.width = width
        self.height = height
        self.vae_scale_factor = vae_scale_factor
        self.patch_size = patch_size
        self.steps = steps
        self.mode = mode
        self.strength = strength
        self.white_background_threshold = white_background_threshold
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
        self.items: list[CaptionedPreviewItem] = []
        with torch.inference_mode():
            for index, row in enumerate(rows):
                source_path = Path(row["source_image"]).expanduser().resolve()
                with Image.open(source_path) as opened:
                    source = ImageOps.exif_transpose(opened).convert("RGB")
                reference = ImageOps.fit(source, (width, height), method=Image.Resampling.LANCZOS)
                prompt_tensors = load_file(
                    prompt_cache_dir / f"{row['prompt_id']}.safetensors",
                    device="cpu",
                )
                generator = torch.Generator(device=device).manual_seed(seed + index)
                pixels = image_to_normalized_tensor(reference).unsqueeze(0).to(device=device, dtype=self.vae.dtype)
                posterior = self.vae.encode(pixels.unsqueeze(2)).latent_dist
                raw_latents = posterior.sample(generator=generator)
                normalized = normalize_qwen_vae_latents(self.vae, raw_latents)[:, :, 0]
                source_latents = pack_latents(normalized, patch_size=patch_size).to(dtype=dtype)
                self.items.append(
                    CaptionedPreviewItem(
                        source_path=source_path,
                        output_stem=safe_output_stem(source_path, index),
                        reference=reference,
                        description=str(row["description"]),
                        prompt_id=str(row["prompt_id"]),
                        embeds=prompt_tensors["embeds"].unsqueeze(0).to(device=device, dtype=dtype),
                        mask=prompt_tensors["mask"].unsqueeze(0).to(device=device),
                        source_latents=source_latents,
                        seed=seed + index,
                    )
                )

    def _initial_latents(self, item: CaptionedPreviewItem) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator(device=self.device).manual_seed(item.seed + 10_000)
        noise = torch.randn(
            item.source_latents.shape,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        image_seq_len = item.source_latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 6400),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        if self.mode == "txt2img":
            sigmas = np.linspace(1.0, 1.0 / self.steps, self.steps)
            timesteps, _ = retrieve_timesteps(
                self.scheduler,
                self.steps,
                self.device,
                sigmas=sigmas,
                mu=mu,
            )
            self.scheduler.set_begin_index(0)
            return noise, timesteps

        total_steps = self.steps if self.strength >= 1.0 else max(self.steps, int(self.steps / self.strength))
        start_index = total_steps - self.steps
        sigmas = np.linspace(1.0, 1.0 / total_steps, total_steps, dtype=np.float32)
        self.scheduler.set_timesteps(sigmas=sigmas, device=self.device, mu=mu)
        timesteps = self.scheduler.timesteps[start_index:]
        self.scheduler.set_begin_index(start_index)
        latent_timestep = timesteps[:1].repeat(item.source_latents.shape[0])
        latents = self.scheduler.scale_noise(item.source_latents, latent_timestep, noise)
        return latents, timesteps

    @torch.inference_mode()
    def save(self, transformer: torch.nn.Module, output_dir: Path) -> list[dict]:
        output_dir.mkdir(parents=True, exist_ok=True)
        was_training = transformer.training
        transformer.eval()
        results: list[dict] = []
        try:
            for item in self.items:
                item.reference.save(output_dir / f"{item.output_stem}_reference.png", optimize=True)
                latents, timesteps = self._initial_latents(item)
                position_ids = Krea2Pipeline.prepare_position_ids(
                    item.embeds.shape[1],
                    self.height // (self.vae_scale_factor * self.patch_size),
                    self.width // (self.vae_scale_factor * self.patch_size),
                    self.device,
                )
                for timestep_value in timesteps:
                    timestep = (
                        timestep_value / self.scheduler.config.num_train_timesteps
                    ).expand(1).to(self.dtype)
                    prediction = transformer(
                        hidden_states=latents,
                        encoder_hidden_states=item.embeds,
                        timestep=timestep,
                        position_ids=position_ids,
                        encoder_attention_mask=item.mask,
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
                generated = snap_near_white_to_white(
                    decoded_tensor_to_pil(decoded),
                    self.white_background_threshold,
                )
                generated.save(generated_path, optimize=True)
                results.append(
                    {
                        "source_image": str(item.source_path),
                        "generated_image": str(generated_path),
                        "prompt_id": item.prompt_id,
                        "description": item.description,
                        "seed": item.seed,
                    }
                )
        finally:
            transformer.train(was_training)
        return results
