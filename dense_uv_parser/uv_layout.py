"""Minecraft skin-atlas layout and RGBA helpers used by Dense UV Parser."""

import torch
import torchvision.transforms.functional as TF


UV_SIZE = 64
PART_COUNT = 6
FACE_COUNT = 6
LAYER_COUNT = 2


def minecraft_layer_rects(is_slim=False):
    """Return inner face rectangles and their matching outer-layer offsets."""
    arm_width = 3 if is_slim else 4
    slim_shift = 1 if is_slim else 0
    parts = [
        (
            [
                ((8, 8), (8, 8)),
                ((8, 8), (24, 8)),
                ((8, 8), (16, 8)),
                ((8, 8), (0, 8)),
                ((8, 8), (8, 0)),
                ((8, 8), (16, 0)),
            ],
            (32, 0),
        ),
        (
            [
                ((8, 12), (20, 20)),
                ((8, 12), (32, 20)),
                ((4, 12), (28, 20)),
                ((4, 12), (16, 20)),
                ((8, 4), (20, 16)),
                ((8, 4), (28, 16)),
            ],
            (0, 16),
        ),
        (
            [
                ((arm_width, 12), (36, 52)),
                ((arm_width, 12), (44 - slim_shift, 52)),
                ((4, 12), (40 - slim_shift, 52)),
                ((4, 12), (32, 52)),
                ((arm_width, 4), (36, 48)),
                ((arm_width, 4), (40 - slim_shift, 48)),
            ],
            (16, 0),
        ),
        (
            [
                ((arm_width, 12), (44, 20)),
                ((arm_width, 12), (52 - slim_shift, 20)),
                ((4, 12), (48 - slim_shift, 20)),
                ((4, 12), (40, 20)),
                ((arm_width, 4), (44, 16)),
                ((arm_width, 4), (48 - slim_shift, 16)),
            ],
            (0, 16),
        ),
        (
            [
                ((4, 12), (20, 52)),
                ((4, 12), (28, 52)),
                ((4, 12), (24, 52)),
                ((4, 12), (16, 52)),
                ((4, 4), (20, 48)),
                ((4, 4), (24, 48)),
            ],
            (-16, 0),
        ),
        (
            [
                ((4, 12), (4, 20)),
                ((4, 12), (12, 20)),
                ((4, 12), (8, 20)),
                ((4, 12), (0, 20)),
                ((4, 4), (4, 16)),
                ((4, 4), (8, 16)),
            ],
            (0, 16),
        ),
    ]

    rects = []
    for faces, decor_offset in parts:
        for (width, height), (inner_x, inner_y) in faces:
            rects.append(
                (
                    inner_x,
                    inner_y,
                    width,
                    height,
                    decor_offset[0],
                    decor_offset[1],
                )
            )
    return rects


def build_part_layer_masks(is_slim=False):
    """Return exact inner/outer atlas masks grouped into six body parts."""
    inner = torch.zeros(PART_COUNT, 1, UV_SIZE, UV_SIZE, dtype=torch.float32)
    outer = torch.zeros_like(inner)
    for rect_index, (inner_x, inner_y, width, height, decor_dx, decor_dy) in enumerate(
        minecraft_layer_rects(is_slim=is_slim)
    ):
        part = rect_index // FACE_COUNT
        inner[part, :, inner_y : inner_y + height, inner_x : inner_x + width] = 1.0
        outer_x = inner_x + decor_dx
        outer_y = inner_y + decor_dy
        outer[part, :, outer_y : outer_y + height, outer_x : outer_x + width] = 1.0
    return inner, outer


def build_uv_masks(is_slim=False):
    inner, outer = build_part_layer_masks(is_slim=is_slim)
    return (
        inner.amax(dim=0),
        outer.amax(dim=0),
    )


def finalize_minecraft_alpha(
    tensor,
    alpha_threshold=0.5,
    enforce_base_alpha=True,
    is_slim=False,
):
    if tensor.shape[-3] != 4:
        raise ValueError(
            f"Expected RGBA tensor with 4 channels, got shape {tuple(tensor.shape)}."
        )
    if not 0.0 <= alpha_threshold <= 1.0:
        raise ValueError("alpha_threshold must be in [0, 1].")

    base_mask, decor_mask = build_uv_masks(is_slim=is_slim)
    valid_mask = torch.maximum(base_mask, decor_mask)
    base_mask = base_mask.to(device=tensor.device, dtype=tensor.dtype)
    valid_mask = valid_mask.to(device=tensor.device, dtype=tensor.dtype)

    out = tensor.clone()
    if tensor.dim() == 4:
        base_mask = base_mask.unsqueeze(0)
        valid_mask = valid_mask.unsqueeze(0)
        alpha = (out[:, 3:4] > alpha_threshold).to(dtype=out.dtype)
        if enforce_base_alpha:
            alpha = torch.where(base_mask > 0, torch.ones_like(alpha), alpha)
        out[:, :3] = out[:, :3] * valid_mask
        out[:, 3:4] = alpha * valid_mask
        return out
    if tensor.dim() == 3:
        alpha = (out[3:4] > alpha_threshold).to(dtype=out.dtype)
        if enforce_base_alpha:
            alpha = torch.where(base_mask > 0, torch.ones_like(alpha), alpha)
        out[:3] = out[:3] * valid_mask
        out[3:4] = alpha * valid_mask
        return out
    raise ValueError(f"Expected CHW or NCHW tensor, got shape {tuple(tensor.shape)}.")


def tensor_to_rgba_image(tensor):
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    if tensor.shape[0] == 3:
        tensor = torch.cat([tensor, torch.ones_like(tensor[:1])], dim=0)
    return TF.to_pil_image(tensor)


def view_native_size(renderer, view):
    return tuple(getattr(renderer, f"{view}_inner_mask").shape)
