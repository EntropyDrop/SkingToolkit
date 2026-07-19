import re
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from SkingToolkit.fixed_view_foreground.model import FixedViewForegroundNet


RUN_PATTERN = re.compile(r"^fixed_view_foreground_v(\d+)$")


def find_latest_checkpoint(runs_dir=None, checkpoint_name="best.pt"):
    if runs_dir is None:
        runs_dir = Path(__file__).resolve().parent / "runs"
    runs_dir = Path(runs_dir)
    candidates = []
    if runs_dir.is_dir():
        for run_dir in runs_dir.iterdir():
            match = RUN_PATTERN.match(run_dir.name)
            checkpoint = run_dir / checkpoint_name
            if match and checkpoint.is_file():
                candidates.append((int(match.group(1)), checkpoint))
    return max(candidates, default=(-1, None), key=lambda item: item[0])[1]


def load_foreground_model(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("model_config", {})
    model = FixedViewForegroundNet(
        input_channels=config.get("input_channels", 3),
        base_channels=config.get("base_channels", 24),
        view_classes=config.get("view_classes", 2),
        coordinate_channels=config.get("coordinate_channels", True),
        # Checkpoints created before geometry conditioning remain loadable.
        geometry_channels=config.get("geometry_channels", 0),
        dropout=config.get("dropout", 0.05),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint.get("args", {}), checkpoint


def build_geometry_prior(renderer, views, batch_size, device, dtype):
    """Return fixed direct inner/outer silhouettes in grouped BxV order."""
    priors = []
    for view in views:
        inner = getattr(renderer, f"{view}_inner_mask").to(
            device=device, dtype=dtype
        )
        outer = getattr(renderer, f"{view}_outer_mask").to(
            device=device, dtype=dtype
        )
        priors.append(torch.stack([inner, outer], dim=0))
    prior = torch.stack(priors, dim=0)
    return (
        prior.unsqueeze(0)
        .expand(batch_size, -1, -1, -1, -1)
        .reshape(batch_size * len(views), 2, *prior.shape[-2:])
    )


def fill_enclosed_holes(mask):
    """Fill background components that cannot reach an image border."""
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.bool()
    background = ~mask
    connected = torch.zeros_like(background)
    connected[:, :, 0, :] = background[:, :, 0, :]
    connected[:, :, -1, :] = background[:, :, -1, :]
    connected[:, :, :, 0] |= background[:, :, :, 0]
    connected[:, :, :, -1] |= background[:, :, :, -1]
    for _ in range(max(mask.shape[-2:])):
        expanded = F.max_pool2d(
            connected.float(), kernel_size=3, stride=1, padding=1
        ).bool()
        updated = connected | (background & expanded)
        if torch.equal(updated, connected):
            break
        connected = updated
    holes = background & ~connected
    return mask | holes


def refine_foreground_mask(raw_mask, geometry_prior=None, core_radius=2):
    """Protect guaranteed base-layer interior and remove enclosed mask holes."""
    if raw_mask.dim() == 3:
        raw_mask = raw_mask.unsqueeze(1)
    refined = raw_mask.bool()
    if geometry_prior is not None:
        if geometry_prior.dim() != 4 or geometry_prior.shape[1] < 1:
            raise ValueError("geometry_prior must be NCHW with an inner channel.")
        if (
            geometry_prior.shape[0] != refined.shape[0]
            or geometry_prior.shape[-2:] != refined.shape[-2:]
        ):
            raise ValueError(
                "geometry_prior must match the foreground mask batch and size."
            )
        inner = (geometry_prior[:, 0:1] > 0.5).float()
        if core_radius > 0:
            kernel = core_radius * 2 + 1
            padded = F.pad(
                inner,
                (core_radius, core_radius, core_radius, core_radius),
                value=0.0,
            )
            inner = -F.max_pool2d(
                -padded, kernel_size=kernel, stride=1, padding=0
            )
        refined = refined | (inner > 0.5)
    return fill_enclosed_holes(refined)


@torch.no_grad()
def predict_foreground(model, rendered, view_ids, geometry_prior=None):
    probability = torch.sigmoid(
        model(
            rendered[:, :3],
            view_ids=view_ids,
            geometry_prior=geometry_prior,
        ).float()
    )
    return probability.clamp(0.0, 1.0)


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
    foreground_probability=None,
    boundary_width=2,
    distance_quantile=0.10,
):
    """Choose a deterministic solid color far from foreground boundary colors."""
    if foreground_mask.dim() == 3:
        foreground_mask = foreground_mask.unsqueeze(1)
    expected = rendered.shape[:1] + (1,) + rendered.shape[-2:]
    if foreground_mask.shape != expected:
        raise ValueError(
            f"Expected foreground mask shape {expected}, got {tuple(foreground_mask.shape)}."
        )
    if boundary_width < 1:
        raise ValueError("boundary_width must be positive.")
    if not 0.0 <= distance_quantile <= 1.0:
        raise ValueError("distance_quantile must be in [0, 1].")
    mask = foreground_mask.to(device=rendered.device, dtype=torch.bool)
    mask_float = mask.float()
    kernel = boundary_width * 2 + 1
    eroded = -F.max_pool2d(-mask_float, kernel, stride=1, padding=boundary_width)
    boundary = mask & (eroded < 0.5)
    if foreground_probability is not None:
        if foreground_probability.dim() == 3:
            foreground_probability = foreground_probability.unsqueeze(1)
        if foreground_probability.shape != expected:
            raise ValueError(
                "foreground_probability must match the foreground mask shape."
            )
        # Prefer reliable edge colors, but fall back to every boundary pixel
        # for pale or antialiased subjects whose edge confidence is lower.
        reliable_boundary = boundary & (
            foreground_probability.to(device=rendered.device) >= 0.75
        )
    else:
        reliable_boundary = boundary

    candidates = rendered.new_tensor(ADAPTIVE_BACKGROUND_CANDIDATES) / 255.0
    selected = []
    selected_indices = []
    for item in range(rendered.shape[0]):
        sample_mask = reliable_boundary[item, 0]
        if not sample_mask.any():
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
        # Maximize a low distance quantile instead of the mean: a candidate
        # that matches even a substantial minority of the silhouette is bad.
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
    foreground_probability=None,
    return_background=False,
):
    """Replace rejected RGB with the parser's neutral background.

    The dense parser is trained on opaque RGB inputs, so its fourth channel
    remains one. Original foreground RGB is preserved exactly; UV splatting can
    still use the untouched source render separately.
    """
    if foreground_mask.dim() == 3:
        foreground_mask = foreground_mask.unsqueeze(1)
    expected = rendered.shape[:1] + (1,) + rendered.shape[-2:]
    if foreground_mask.shape != expected:
        raise ValueError(
            f"Expected foreground mask shape {expected}, got {tuple(foreground_mask.shape)}."
        )
    mask = foreground_mask.to(device=rendered.device, dtype=torch.bool)
    if background_mode == "adaptive":
        bg, selected_indices = select_adaptive_background(
            rendered,
            mask,
            foreground_probability=foreground_probability,
        )
    elif background_mode == "neutral":
        bg = rendered.new_tensor(bg_color).view(1, 3, 1, 1) / 255.0
        bg = bg.expand(rendered.shape[0], -1, -1, -1)
        selected_indices = None
    else:
        raise ValueError(
            f"Unknown foreground parser background mode {background_mode!r}."
        )
    rgb = torch.where(mask.expand(-1, 3, -1, -1), rendered[:, :3], bg)
    parser_input = torch.cat([rgb, torch.ones_like(rendered[:, 3:4])], dim=1)
    if return_background:
        return parser_input, bg[:, :, 0, 0], selected_indices
    return parser_input


def save_foreground_outputs(
    rendered,
    probability,
    threshold,
    view_count,
    probability_output=None,
    raw_mask_output=None,
    mask_output=None,
    cutout_output=None,
    bg_color=(128, 128, 128),
    geometry_prior=None,
):
    raw_mask = probability >= threshold
    mask = refine_foreground_mask(raw_mask, geometry_prior=geometry_prior)
    # Keep source RGB under alpha so the saved PNG is a reusable transparent
    # cutout rather than a gray-background preview.
    cutout = torch.cat(
        [rendered[:, :3], mask.to(dtype=rendered.dtype)], dim=1
    )
    for tensor, path in (
        (probability, probability_output),
        (raw_mask.to(dtype=rendered.dtype), raw_mask_output),
        (mask.to(dtype=rendered.dtype), mask_output),
        (cutout, cutout_output),
    ):
        if path is None:
            continue
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_image(tensor.detach().cpu(), path, nrow=view_count)
    return mask[:, 0]
