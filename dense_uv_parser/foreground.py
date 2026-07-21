from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image


ADAPTIVE_BACKGROUND_CANDIDATES = (
    (10, 10, 10),
    (245, 245, 245),
    (13, 204, 242),
    (242, 26, 191),
    (26, 230, 64),
    (242, 191, 13),
    (38, 64, 242),
    (242, 51, 26),
)


def select_adaptive_background(
    rendered,
    foreground_mask,
    boundary_width=2,
    distance_quantile=0.10,
):
    """Choose a deterministic parser background far from foreground edges."""
    if foreground_mask.dim() == 3:
        foreground_mask = foreground_mask.unsqueeze(1)
    expected = rendered.shape[:1] + (1,) + rendered.shape[-2:]
    if foreground_mask.shape != expected:
        raise ValueError(
            f"Expected foreground mask shape {expected}, got "
            f"{tuple(foreground_mask.shape)}."
        )
    if boundary_width < 1:
        raise ValueError("boundary_width must be positive.")
    if not 0.0 <= distance_quantile <= 1.0:
        raise ValueError("distance_quantile must be in [0, 1].")

    mask = foreground_mask.to(device=rendered.device, dtype=torch.bool)
    kernel = boundary_width * 2 + 1
    eroded = -F.max_pool2d(
        -mask.float(), kernel, stride=1, padding=boundary_width
    )
    boundary = mask & (eroded < 0.5)
    candidates = rendered.new_tensor(ADAPTIVE_BACKGROUND_CANDIDATES) / 255.0
    selected = []
    selected_indices = []
    for item in range(rendered.shape[0]):
        sample_mask = boundary[item, 0]
        if not sample_mask.any():
            sample_mask = mask[item, 0]
        if not sample_mask.any():
            selected_indices.append(0)
            selected.append(candidates[0])
            continue
        colors = rendered[item, :3, sample_mask].transpose(0, 1).float()
        distances = (
            colors.unsqueeze(0) - candidates.float().unsqueeze(1)
        ).square().mean(dim=2).sqrt()
        scores = torch.quantile(distances, distance_quantile, dim=1)
        index = int(scores.argmax().item())
        selected_indices.append(index)
        selected.append(candidates[index])
    return torch.stack(selected, dim=0).view(-1, 3, 1, 1), selected_indices


def build_parser_input(
    rendered,
    foreground_mask,
    bg_color=(128, 128, 128),
    background_mode="adaptive",
    return_background=False,
):
    """Composite a foreground mask into opaque RGB for the dense parser."""
    if foreground_mask.dim() == 3:
        foreground_mask = foreground_mask.unsqueeze(1)
    expected = rendered.shape[:1] + (1,) + rendered.shape[-2:]
    if foreground_mask.shape != expected:
        raise ValueError(
            f"Expected foreground mask shape {expected}, got "
            f"{tuple(foreground_mask.shape)}."
        )
    mask = foreground_mask.to(device=rendered.device, dtype=torch.bool)
    if background_mode == "adaptive":
        background, selected_indices = select_adaptive_background(
            rendered, mask
        )
    elif background_mode == "neutral":
        background = rendered.new_tensor(bg_color).view(1, 3, 1, 1) / 255.0
        background = background.expand(rendered.shape[0], -1, -1, -1)
        selected_indices = None
    else:
        raise ValueError(
            f"Unknown foreground parser background mode {background_mode!r}."
        )

    rgb = torch.where(
        mask.expand(-1, 3, -1, -1), rendered[:, :3], background
    )
    parser_input = torch.cat([rgb, torch.ones_like(rendered[:, 3:4])], dim=1)
    if return_background:
        return parser_input, background[:, :, 0, 0], selected_indices
    return parser_input


def save_flood_outputs(
    rendered,
    foreground_mask,
    view_count,
    probability_output=None,
    raw_mask_output=None,
    mask_output=None,
    cutout_output=None,
):
    """Save the deterministic flood mask and transparent RGBA cutout."""
    if foreground_mask.dim() == 3:
        foreground_mask = foreground_mask.unsqueeze(1)
    expected = rendered.shape[:1] + (1,) + rendered.shape[-2:]
    if foreground_mask.shape != expected:
        raise ValueError(
            f"Expected foreground mask shape {expected}, got "
            f"{tuple(foreground_mask.shape)}."
        )
    mask = foreground_mask.to(device=rendered.device, dtype=torch.bool)
    score = mask.to(dtype=rendered.dtype)
    cutout = torch.cat([rendered[:, :3], score], dim=1)
    for tensor, output in (
        (score, probability_output),
        (score, raw_mask_output),
        (score, mask_output),
        (cutout, cutout_output),
    ):
        if output is None:
            continue
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_image(tensor.detach().cpu(), path, nrow=view_count)
    return mask[:, 0]
