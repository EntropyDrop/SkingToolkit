from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from common import pack_latents


def prepare_paired_position_ids(
    text_seq_len: int,
    grid_height: int,
    grid_width: int,
    device: torch.device,
) -> torch.Tensor:
    """RoPE coordinates for [text | noisy target | clean reference] tokens.

    Target coordinates remain identical to the base Krea2 pipeline. Reference
    tokens use temporal/image-region coordinate 1 so LoRA attention can tell
    the two equally sized image grids apart without changing model channels.
    """
    text_ids = torch.zeros(text_seq_len, 3, device=device)
    target_ids = torch.zeros(grid_height, grid_width, 3, device=device)
    target_ids[..., 1] = torch.arange(grid_height, device=device)[:, None]
    target_ids[..., 2] = torch.arange(grid_width, device=device)[None, :]
    reference_ids = target_ids.clone()
    reference_ids[..., 0] = 1
    return torch.cat(
        [text_ids, target_ids.reshape(-1, 3), reference_ids.reshape(-1, 3)],
        dim=0,
    )


def normalize_qwen_vae_latents(vae, raw_latents: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(
        vae.config.latents_mean,
        device=raw_latents.device,
        dtype=raw_latents.dtype,
    ).view(1, vae.config.z_dim, 1, 1, 1)
    std_inverse = torch.tensor(
        vae.config.latents_std,
        device=raw_latents.device,
        dtype=raw_latents.dtype,
    ).reciprocal().view(1, vae.config.z_dim, 1, 1, 1)
    return (raw_latents - mean) * std_inverse


def denormalize_qwen_vae_latents(vae, normalized: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(
        vae.config.latents_mean,
        device=normalized.device,
        dtype=normalized.dtype,
    ).view(1, vae.config.z_dim, 1, 1, 1)
    std = torch.tensor(
        vae.config.latents_std,
        device=normalized.device,
        dtype=normalized.dtype,
    ).view(1, vae.config.z_dim, 1, 1, 1)
    return normalized * std + mean


def encode_qwen_vae_images(
    vae,
    pixels: torch.Tensor,
    patch_size: int = 2,
    sample_posterior: bool = False,
) -> torch.Tensor:
    """Encode BCHW [-1, 1] images to packed Krea2 latent tokens."""
    posterior = vae.encode(pixels.unsqueeze(2)).latent_dist
    raw_latents = posterior.sample() if sample_posterior else posterior.mode()
    normalized = normalize_qwen_vae_latents(vae, raw_latents)
    if normalized.ndim != 5 or normalized.shape[2] != 1:
        raise ValueError(f"Unexpected Qwen Image latent shape: {tuple(normalized.shape)}")
    return pack_latents(normalized[:, :, 0], patch_size=patch_size)


def image_to_normalized_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(127.5).sub_(1.0)


def fit_reference_image(
    image_or_path: Image.Image | str | Path,
    width: int,
    height: int,
    background_rgb: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Contain an arbitrary reference without cropping character details."""
    if isinstance(image_or_path, Image.Image):
        source = image_or_path.convert("RGB")
    else:
        with Image.open(image_or_path) as opened:
            source = opened.convert("RGB")
    fitted = ImageOps.contain(source, (width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), background_rgb)
    canvas.paste(fitted, ((width - fitted.width) // 2, (height - fitted.height) // 2))
    return canvas
