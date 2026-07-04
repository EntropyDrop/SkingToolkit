import os
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TOOLKIT_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from SkingToolkit.inverse_uv.dataset import IMAGE_EXTENSIONS, load_skin, parse_views  # noqa: E402
from SkingToolkit.renderer import DifferentiableRenderer  # noqa: E402


def parse_color(value):
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) != 3:
            raise ValueError(f"Expected RGB color as 'r,g,b', got {value!r}.")
        return tuple(int(part) for part in parts)
    return tuple(value)


def _solid_background(color, height, width, dtype):
    color = torch.tensor(color, dtype=dtype).view(3, 1, 1) / 255.0
    return color.expand(3, height, width)


def _gradient_background(height, width, dtype):
    c1 = torch.rand(3, 1, 1, dtype=dtype)
    c2 = torch.rand(3, 1, 1, dtype=dtype)
    if torch.rand(1).item() < 0.5:
        grid = torch.linspace(0, 1, steps=width, dtype=dtype).view(1, 1, width)
    else:
        grid = torch.linspace(0, 1, steps=height, dtype=dtype).view(1, height, 1)
    return c1 * (1.0 - grid) + c2 * grid


def _pattern_background(height, width, dtype):
    c1 = torch.rand(3, 1, 1, dtype=dtype)
    c2 = torch.rand(3, 1, 1, dtype=dtype)
    grid_size = torch.randint(4, 16, (1,)).item()
    y_idx = torch.arange(height, dtype=torch.int64).view(1, height, 1) // grid_size
    x_idx = torch.arange(width, dtype=torch.int64).view(1, 1, width) // grid_size
    checker = ((y_idx + x_idx) % 2).to(dtype=dtype)
    return c1 * checker + c2 * (1.0 - checker)


def _hard_color_background(rendered_rgba, height, width, dtype):
    if rendered_rgba is not None and rendered_rgba.shape[0] >= 4:
        mask = rendered_rgba[3] > 0.5
        if mask.any():
            fg_pixels = rendered_rgba[:3][:, mask]  # [3, N]
            idx = torch.randint(0, fg_pixels.shape[1], (1,)).item()
            color = fg_pixels[:, idx].view(3, 1, 1)
            # Add slight random noise to simulate subtle color variations
            color = (color + (torch.rand_like(color) - 0.5) * 0.1).clamp(0.0, 1.0)
            return color.expand(3, height, width)
    random_color = torch.randint(0, 256, (3, 1, 1), dtype=torch.int64).to(dtype=dtype) / 255.0
    return random_color.expand(3, height, width)


def make_background(mode, color, height, width, dtype, rendered_rgba=None, hard_bg_prob=0.3):
    if mode == "black":
        return _solid_background((0, 0, 0), height, width, dtype)
    if mode == "white":
        return _solid_background((255, 255, 255), height, width, dtype)
    if mode == "gray":
        return _solid_background((128, 128, 128), height, width, dtype)
    if mode == "color":
        return _solid_background(color, height, width, dtype)
    if mode == "gradient":
        return _gradient_background(height, width, dtype)
    if mode == "pattern":
        return _pattern_background(height, width, dtype)
    if mode == "hard":
        return _hard_color_background(rendered_rgba, height, width, dtype)

    if mode != "random":
        raise ValueError(f"Unknown background_mode={mode!r}.")

    # "random" mode: dynamically pick among solid, hard color (color matching foreground), gradient, and pattern
    rand_val = torch.rand(1).item()
    if rand_val < hard_bg_prob and rendered_rgba is not None:
        return _hard_color_background(rendered_rgba, height, width, dtype)
    elif rand_val < hard_bg_prob + 0.2:
        return _gradient_background(height, width, dtype)
    elif rand_val < hard_bg_prob + 0.3:
        return _pattern_background(height, width, dtype)
    else:
        random_color = torch.randint(0, 256, (3, 1, 1), dtype=torch.int64).to(dtype=dtype) / 255.0
        return random_color.expand(3, height, width)


def composite_over_background(rendered_rgba, background):
    alpha = rendered_rgba[3:4].clamp(0.0, 1.0)
    fg_rgb = rendered_rgba[:3] / alpha.clamp_min(1e-4)
    fg_rgb = torch.where(alpha > 1e-4, fg_rgb, torch.zeros_like(fg_rgb)).clamp(0.0, 1.0)
    rgb = fg_rgb * alpha + background * (1.0 - alpha)
    return rgb.clamp(0.0, 1.0), alpha


class ForegroundAlphaDataset(Dataset):
    def __init__(
        self,
        data_dir,
        mappings_dir=None,
        views="walk_front_both_layer_ortho,walk_back_both_layer_ortho",
        background_mode="random",
        bg_color=(0, 0, 0),
        hard_bg_prob=0.3,
        max_samples=None,
        normalize_model=True,
    ):
        self.data_dir = data_dir
        self.views = parse_views(views)
        self.background_mode = background_mode
        self.bg_color = parse_color(bg_color)
        self.hard_bg_prob = hard_bg_prob
        self.normalize_model = normalize_model
        self.renderer = DifferentiableRenderer(mappings_dir=mappings_dir, bg_color=(0.0, 0.0, 0.0))

        missing_views = [view for view in self.views if view not in self.renderer.views]
        if missing_views:
            raise ValueError(
                f"Unknown renderer views {missing_views}. "
                f"Available views: {', '.join(self.renderer.views)}"
            )

        self.skin_paths = sorted(
            os.path.join(data_dir, filename)
            for filename in os.listdir(data_dir)
            if filename.lower().endswith(IMAGE_EXTENSIONS) and not filename.startswith("half_")
        )
        if max_samples is not None:
            self.skin_paths = self.skin_paths[:max_samples]
        if not self.skin_paths:
            raise ValueError(f"No skin images found in {data_dir}")

    def __len__(self):
        return len(self.skin_paths) * len(self.views)

    def __getitem__(self, index):
        skin_index = index // len(self.views)
        view_index = index % len(self.views)
        skin_path = self.skin_paths[skin_index]
        view = self.views[view_index]

        skin = load_skin(skin_path, bg_color=(0, 0, 0), normalize_model=self.normalize_model)
        with torch.no_grad():
            rendered = self.renderer.forward_view(skin.unsqueeze(0), view).squeeze(0).clamp(0.0, 1.0)

        height, width = rendered.shape[-2:]
        background = make_background(
            self.background_mode,
            self.bg_color,
            height,
            width,
            rendered.dtype,
            rendered_rgba=rendered,
            hard_bg_prob=self.hard_bg_prob,
        )
        rgb, alpha = composite_over_background(rendered, background)
        return {
            "image": rgb,
            "alpha": alpha,
            "path": skin_path,
            "view": view,
        }

