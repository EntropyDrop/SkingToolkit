import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TOOLKIT_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from SkingToolkit.dataset import alice_to_steve, resolve_voxel_consistency  # noqa: E402
from SkingToolkit.renderer import DifferentiableRenderer  # noqa: E402


IMAGE_EXTENSIONS = (".png", ".webp", ".jpg", ".jpeg")
UV_SIZE = 64
CONDITIONING_LAYERS = ("inner", "outer")
CONDITIONING_CHANNELS = len(CONDITIONING_LAYERS) * 5


def parse_views(views):
    if isinstance(views, str):
        return [view.strip() for view in views.split(",") if view.strip()]
    return list(views)


def load_skin(path, bg_color=(128, 128, 128), normalize_model=True):
    skin = Image.open(path).convert("RGBA")
    if normalize_model and skin.getpixel((47, 52))[3] == 0:
        skin = alice_to_steve(skin)
    if normalize_model:
        skin = resolve_voxel_consistency(skin)

    skin_np = np.array(skin, dtype=np.uint8)
    alpha_np = skin_np[..., 3]
    skin_np[(alpha_np > 0) & (alpha_np < 255), 3] = 255
    skin = Image.fromarray(skin_np)

    rgba = torch.from_numpy(np.array(skin, dtype="float32") / 255.0).permute(2, 0, 1)
    alpha = rgba[3:4]
    bg = torch.tensor(bg_color, dtype=rgba.dtype).view(3, 1, 1) / 255.0
    rgba[:3] = torch.where(alpha > 0, rgba[:3], bg)
    return rgba.clamp(0.0, 1.0)


def view_native_size(renderer, view):
    mask = getattr(renderer, f"{view}_inner_mask")
    return tuple(mask.shape)


def _uv_indices_from_grid(grid, mask):
    coords = ((grid[mask] + 1.0) * 0.5 * (UV_SIZE - 1)).round().long()
    coords[:, 0].clamp_(0, UV_SIZE - 1)
    coords[:, 1].clamp_(0, UV_SIZE - 1)
    return coords[:, 1] * UV_SIZE + coords[:, 0]


def _ensure_rgba(rendered, geometry_mask=None):
    if rendered.shape[0] == 4:
        return rendered
    if rendered.shape[0] != 3:
        raise ValueError(f"Expected RGB or RGBA render tensor, got {rendered.shape[0]} channels.")
    if geometry_mask is None:
        alpha = torch.ones_like(rendered[:1])
    else:
        alpha = geometry_mask.to(dtype=rendered.dtype, device=rendered.device).unsqueeze(0)
    return torch.cat([rendered, alpha], dim=0)


def unproject_renders_to_uv(rendered_views, renderer, views, bg_color=(128, 128, 128)):
    views = parse_views(views)
    if len(rendered_views) != len(views):
        raise ValueError(f"Expected {len(views)} rendered views, got {len(rendered_views)}.")

    sample = rendered_views[0]
    device = sample.device
    dtype = sample.dtype
    accum = sample.new_zeros(len(CONDITIONING_LAYERS), 4, UV_SIZE, UV_SIZE)
    counts = sample.new_zeros(len(CONDITIONING_LAYERS), 1, UV_SIZE, UV_SIZE)

    for view, rendered in zip(views, rendered_views):
        inner_mask = getattr(renderer, f"{view}_inner_mask").to(device=device).bool()
        outer_mask = getattr(renderer, f"{view}_outer_mask").to(device=device).bool()
        geometry_mask = inner_mask | outer_mask
        rendered = _ensure_rgba(rendered.to(device=device, dtype=dtype), geometry_mask=geometry_mask)

        for layer_index, layer in enumerate(CONDITIONING_LAYERS):
            mask = getattr(renderer, f"{view}_{layer}_mask").to(device=device).bool()
            if not bool(mask.any()):
                continue
            grid = getattr(renderer, f"{view}_{layer}_grid").to(device=device, dtype=dtype)
            flat_uv = _uv_indices_from_grid(grid, mask)
            values = rendered[:, mask]

            accum[layer_index].reshape(4, -1).index_add_(1, flat_uv, values)
            ones = torch.ones((1, values.shape[1]), dtype=dtype, device=device)
            counts[layer_index].reshape(1, -1).index_add_(1, flat_uv, ones)

    known = (counts > 0).to(dtype=dtype)
    averaged = accum / counts.clamp_min(1.0)
    bg = sample.new_tensor(bg_color, dtype=dtype).view(1, 3, 1, 1) / 255.0
    rgb = torch.where(known.expand(-1, 3, -1, -1) > 0, averaged[:, :3], bg.expand_as(averaged[:, :3]))
    alpha = torch.where(known > 0, averaged[:, 3:4], torch.zeros_like(averaged[:, 3:4]))
    layers = torch.cat([rgb, alpha, known], dim=1)
    return layers.reshape(-1, UV_SIZE, UV_SIZE).clamp(0.0, 1.0)


def build_conditioning(
    skin,
    renderer,
    views,
    image_size=256,
    include_alpha=False,
):
    _ = image_size, include_alpha
    rendered_views = []
    with torch.no_grad():
        skin_batch = skin.unsqueeze(0)
        for view in views:
            rendered = renderer.forward_view(skin_batch, view).squeeze(0)
            rendered_views.append(rendered)
    return unproject_renders_to_uv(rendered_views, renderer, views)


class InverseUVDataset(Dataset):
    def __init__(
        self,
        data_dir,
        mappings_dir=None,
        views="static_front,static_back",
        image_size=256,
        include_alpha=False,
        bg_color=(128, 128, 128),
        max_samples=None,
        normalize_model=True,
    ):
        self.data_dir = data_dir
        self.views = parse_views(views)
        self.image_size = image_size
        self.include_alpha = include_alpha
        self.bg_color = bg_color
        self.normalize_model = normalize_model
        self.renderer = DifferentiableRenderer(
            mappings_dir=mappings_dir,
            bg_color=tuple(channel / 255.0 for channel in bg_color),
        )

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

        self.input_channels = CONDITIONING_CHANNELS

    def __len__(self):
        return len(self.skin_paths)

    def __getitem__(self, index):
        skin_path = self.skin_paths[index]
        uv = load_skin(
            skin_path,
            bg_color=self.bg_color,
            normalize_model=self.normalize_model,
        )
        conditioning = build_conditioning(
            uv,
            self.renderer,
            self.views,
            image_size=self.image_size,
            include_alpha=self.include_alpha,
        )
        return {
            "conditioning": conditioning,
            "uv": uv,
            "path": skin_path,
        }


def tensor_to_rgba_image(tensor):
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    if tensor.shape[0] == 3:
        tensor = torch.cat([tensor, torch.ones_like(tensor[:1])], dim=0)
    return TF.to_pil_image(tensor)
