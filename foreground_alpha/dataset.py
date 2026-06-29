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


def make_background(mode, color, height, width, dtype):
    if mode == "black":
        return _solid_background((0, 0, 0), height, width, dtype)
    if mode == "white":
        return _solid_background((255, 255, 255), height, width, dtype)
    if mode == "gray":
        return _solid_background((128, 128, 128), height, width, dtype)
    if mode == "color":
        return _solid_background(color, height, width, dtype)
    if mode != "random":
        raise ValueError(f"Unknown background_mode={mode!r}.")
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
        views="walk_perspective_ortho,walk_perspective_back_ortho",
        background_mode="random",
        bg_color=(0, 0, 0),
        max_samples=None,
        normalize_model=True,
    ):
        self.data_dir = data_dir
        self.views = parse_views(views)
        self.background_mode = background_mode
        self.bg_color = parse_color(bg_color)
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
        background = make_background(self.background_mode, self.bg_color, height, width, rendered.dtype)
        rgb, alpha = composite_over_background(rendered, background)
        return {
            "image": rgb,
            "alpha": alpha,
            "path": skin_path,
            "view": view,
        }
