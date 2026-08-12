from __future__ import annotations

import numpy as np
from PIL import Image


def snap_near_white_to_white(image: Image.Image, threshold: int = 250) -> Image.Image:
    """Remove Qwen VAE near-white drift from an intentionally pure-white canvas."""
    if not 0 <= threshold <= 255:
        raise ValueError("white threshold must be between 0 and 255")
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    pixels[np.all(pixels >= threshold, axis=-1)] = 255
    return Image.fromarray(pixels, mode="RGB")
