from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def snap_near_white_to_white(image: Image.Image, threshold: int = 250) -> Image.Image:
    """Remove Qwen VAE near-white drift from an intentionally pure-white canvas."""
    if not 0 <= threshold <= 255:
        raise ValueError("white threshold must be between 0 and 255")
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    pixels[np.all(pixels >= threshold, axis=-1)] = 255
    return Image.fromarray(pixels, mode="RGB")


def minecraft_crisp_postprocess(
    image: Image.Image,
    *,
    enabled: bool = True,
    white_threshold: int = 250,
    sharpen_radius: float = 0.6,
    sharpen_percent: int = 80,
    sharpen_threshold: int = 3,
    contrast: float = 1.16,
    saturation: float = 1.10,
    posterize_bits: int = 4,
) -> Image.Image:
    """Recover hard Minecraft-like color blocks after diffusion/VAE decoding.

    The target renders are already sharp, but diffusion decoding introduces many
    near-identical intermediate colors. A small unsharp mask restores edge
    contrast and posterization removes those intermediate colors without
    dithering. The original near-white mask is restored after posterization so
    the canvas remains exact RGB(255, 255, 255).
    """
    if not 0 <= white_threshold <= 255:
        raise ValueError("white threshold must be between 0 and 255")
    rgb = image.convert("RGB")
    original = np.asarray(rgb, dtype=np.uint8)
    white_mask = np.all(original >= white_threshold, axis=-1)
    if not enabled:
        pixels = original.copy()
        pixels[white_mask] = 255
        return Image.fromarray(pixels, mode="RGB")
    if sharpen_radius < 0:
        raise ValueError("sharpen radius must be non-negative")
    if sharpen_percent < 0:
        raise ValueError("sharpen percent must be non-negative")
    if sharpen_threshold < 0:
        raise ValueError("sharpen threshold must be non-negative")
    if contrast < 0:
        raise ValueError("contrast must be non-negative")
    if saturation < 0:
        raise ValueError("saturation must be non-negative")
    if not 1 <= posterize_bits <= 8:
        raise ValueError("posterize bits must be between 1 and 8")

    crisp = rgb.filter(
        ImageFilter.UnsharpMask(
            radius=sharpen_radius,
            percent=sharpen_percent,
            threshold=sharpen_threshold,
        )
    )
    crisp = ImageEnhance.Contrast(crisp).enhance(contrast)
    crisp = ImageEnhance.Color(crisp).enhance(saturation)
    if posterize_bits < 8:
        crisp = ImageOps.posterize(crisp, posterize_bits)
    pixels = np.asarray(crisp, dtype=np.uint8).copy()
    pixels[white_mask] = 255
    return Image.fromarray(pixels, mode="RGB")
