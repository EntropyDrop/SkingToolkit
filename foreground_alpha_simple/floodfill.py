import cv2
import numpy as np
from PIL import Image


def get_seed_color(image_np, seed=(0, 0)):
    """
    Extract color at seed position (x, y).
    image_np: uint8 numpy array [H, W, 3] (RGB or BGR).
    seed: (x, y) tuple, default (0, 0).
    """
    x, y = seed
    h, w = image_np.shape[:2]
    x = max(0, min(w - 1, int(x)))
    y = max(0, min(h - 1, int(y)))
    return image_np[y, x].tolist()


def flood_fill_alpha_simple(
    image,
    seed=(0, 0),
    tolerance=15,
    color_space="RGB",
    fixed_range=True,
    extra_seeds=None,
    fill_holes=False,
    opening_size=0,
):
    """
    Removes contiguous similar background color starting from seed coordinate (x, y), default (0, 0).

    Args:
        image: PIL.Image or numpy.ndarray (RGB [H, W, 3]).
        seed: Tuple (x, y) for initial seed point. Default is (0, 0).
        tolerance: Color difference tolerance (0 to 255). Default 15.
        color_space: "RGB" or "LAB" color space for matching. Default "RGB".
        fixed_range: If True, compares pixel color against original seed color (fixed range).
                     If False, compares pixel color against adjacent pixel color (floating range).
        extra_seeds: Optional list of additional seed (x, y) tuples (e.g. for merged images or 4 corners).
        fill_holes: If True, fills enclosed holes inside the foreground mask using morphological operations.
        opening_size: Kernel size for morphological opening to remove thin edge artifacts (0 = disabled).

    Returns:
        alpha_mask: uint8 numpy array [H, W] where 255 is foreground and 0 is background.
    """
    if isinstance(image, Image.Image):
        img_rgb = np.array(image.convert("RGB"))
    else:
        img_rgb = np.array(image)

    h, w, c = img_rgb.shape

    if color_space.upper() == "LAB":
        work_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    else:
        work_img = img_rgb

    # Mask for cv2.floodFill needs to be (H+2, W+2), uint8
    combined_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    seeds = [seed]
    if extra_seeds:
        seeds.extend(extra_seeds)

    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    if fixed_range:
        flags |= cv2.FLOODFILL_FIXED_RANGE

    if isinstance(tolerance, (tuple, list)):
        tol_val = tuple(tolerance)
    else:
        tol_val = (float(tolerance), float(tolerance), float(tolerance))

    for s in seeds:
        sx, sy = int(s[0]), int(s[1])
        if 0 <= sx < w and 0 <= sy < h:
            # Note OpenCV floodFill accepts seedPoint as (x, y)
            cv2.floodFill(
                work_img.copy(),
                combined_mask,
                (sx, sy),
                0,  # newVal is unused when FLOODFILL_MASK_ONLY is set
                tol_val,
                tol_val,
                flags=flags,
            )

    # combined_mask has 255 where background flood fill touched
    bg_mask = combined_mask[1:-1, 1:-1] == 255
    fg_mask = (~bg_mask).astype(np.uint8) * 255

    # Optional morphological cleanup
    if opening_size and opening_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (opening_size, opening_size))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    if fill_holes:
        # Fill holes inside the foreground mask
        contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
        fg_mask = filled_mask

    return fg_mask


def uncompose_background_np(img_rgb, alpha_mask, bg_color, min_alpha=10):
    """
    Recover foreground RGB color by removing solid background color blending at semi-transparent edges.

    img_rgb: uint8 array [H, W, 3]
    alpha_mask: uint8 array [H, W] (0 to 255)
    bg_color: tuple (R, G, B) or list
    """
    img_float = img_rgb.astype(np.float32) / 255.0
    alpha_float = (alpha_mask.astype(np.float32) / 255.0)[:, :, None]
    bg_float = np.array(bg_color, dtype=np.float32).reshape(1, 1, 3) / 255.0

    fg_float = (img_float - bg_float * (1.0 - alpha_float)) / np.maximum(alpha_float, min_alpha / 255.0)
    fg_float = np.clip(fg_float, 0.0, 1.0)

    res = np.where(alpha_float > (min_alpha / 255.0), fg_float, img_float)
    return (res * 255.0).astype(np.uint8)


def remove_background_simple(
    image,
    seed=(0, 0),
    tolerance=15,
    color_space="RGB",
    fixed_range=True,
    extra_seeds=None,
    fill_holes=False,
    opening_size=0,
    uncompose=False,
):
    """
    Full workflow function to remove background from an image using seed point (0,0) flood fill.

    Returns:
        rgba_image: PIL.Image in RGBA format.
    """
    if isinstance(image, Image.Image):
        img_pil = image.convert("RGB")
    else:
        img_pil = Image.fromarray(image).convert("RGB")

    img_np = np.array(img_pil)
    seed_color = get_seed_color(img_np, seed=seed)

    alpha_mask = flood_fill_alpha_simple(
        img_np,
        seed=seed,
        tolerance=tolerance,
        color_space=color_space,
        fixed_range=fixed_range,
        extra_seeds=extra_seeds,
        fill_holes=fill_holes,
        opening_size=opening_size,
    )

    if uncompose:
        clean_rgb = uncompose_background_np(img_np, alpha_mask, bg_color=seed_color)
    else:
        clean_rgb = img_np

    rgba_np = np.dstack([clean_rgb, alpha_mask])
    return Image.fromarray(rgba_np, mode="RGBA")
