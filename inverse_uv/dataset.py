import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TOOLKIT_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from mc_skin_utils.alice_to_steve import alice_to_steve  # noqa: E402
from SkingToolkit.renderer import DifferentiableRenderer  # noqa: E402


IMAGE_EXTENSIONS = (".png", ".webp", ".jpg", ".jpeg")
UV_SIZE = 64
CONDITIONING_LAYERS = ("inner", "outer")
CONDITIONING_CHANNELS = len(CONDITIONING_LAYERS) * 5


class RenderAugmenter:
    """Apply pose-robustness augmentation in render space before UV unprojection.

    Translation + mild perspective warp simulate slight viewpoint shifts.
    Uniform scaling simulates distance/camera zoom variation.
    All transforms are applied to rendered views (not UV), so perturbed pixels
    get mapped to slightly wrong UV texels during unprojection — teaching the
    model to be robust to imperfect conditioning.
    """

    def __init__(
        self,
        translation_scale=0.0,
        scale_range=0.0,
        perspective_scale=0.0,
        bg_color=(128, 128, 128),
        generator=None,
    ):
        self.translation_scale = translation_scale
        self.scale_range = scale_range
        self.perspective_scale = perspective_scale
        self.bg_color = bg_color
        self.generator = generator

    def __call__(self, rendered_tensor):
        if rendered_tensor.dim() == 4:
            return self._call_batch(rendered_tensor)
        return self._call_single(rendered_tensor)

    def _fill_color(self, device, dtype):
        return torch.tensor(
            [self.bg_color[0] / 255.0, self.bg_color[1] / 255.0, self.bg_color[2] / 255.0, 0.0],
            device=device,
            dtype=dtype,
        )

    def _call_batch(self, rendered_tensor):
        # rendered_tensor: (B, C, H, W)
        B, C, H, W = rendered_tensor.shape
        device = rendered_tensor.device
        dtype = rendered_tensor.dtype
        fill = self._fill_color(device, dtype)[:C].view(1, C, 1, 1)
        img = rendered_tensor

        if self.translation_scale > 0 or self.scale_range > 0:
            dx = (torch.rand(B, device=device, dtype=dtype, generator=self.generator) - 0.5) * 2 * self.translation_scale * W
            dy = (torch.rand(B, device=device, dtype=dtype, generator=self.generator) - 0.5) * 2 * self.translation_scale * H
            scale = 1.0 + (torch.rand(B, device=device, dtype=dtype, generator=self.generator) - 0.5) * 2 * self.scale_range
            inv_scale = scale.reciprocal()

            theta = torch.zeros(B, 2, 3, device=device, dtype=dtype)
            theta[:, 0, 0] = inv_scale
            theta[:, 1, 1] = inv_scale
            theta[:, 0, 2] = -2.0 * dx / max(W, 1)
            theta[:, 1, 2] = -2.0 * dy / max(H, 1)

            grid = F.affine_grid(theta, img.shape, align_corners=False)
            img = F.grid_sample(
                img - fill,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            ) + fill

        if self.perspective_scale > 0:
            img = torch.stack([self._call_perspective_single(img[i]) for i in range(B)], dim=0)
        return img

    def _call_perspective_single(self, rendered_tensor):
        # rendered_tensor: (C, H, W) — single render view
        C, H, W = rendered_tensor.shape
        device = rendered_tensor.device
        dtype = rendered_tensor.dtype
        fill_color = self._fill_color(device, dtype)[:C].tolist()
        img = rendered_tensor.unsqueeze(0)  # (1, C, H, W)

        if self.perspective_scale > 0:
            startpoints = [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]]
            endpoints = []
            for x, y in startpoints:
                ex = (torch.rand(1).item() - 0.5) * 2 * self.perspective_scale * W
                ey = (torch.rand(1).item() - 0.5) * 2 * self.perspective_scale * H
                endpoints.append([x + ex, y + ey])
            img = TF.perspective(
                img, startpoints, endpoints,
                interpolation=TF.InterpolationMode.BILINEAR, fill=fill_color,
            )

        return img.squeeze(0)

    def _call_single(self, rendered_tensor):
        # rendered_tensor: (C, H, W) — single render view
        C, H, W = rendered_tensor.shape
        dtype = rendered_tensor.dtype
        fill_color = self._fill_color(rendered_tensor.device, dtype)[:C].tolist()
        img = rendered_tensor.unsqueeze(0)  # (1, C, H, W)

        # 1. Random translation + uniform scale (single affine call)
        dx = int((torch.rand(1).item() - 0.5) * 2 * self.translation_scale * W)
        dy = int((torch.rand(1).item() - 0.5) * 2 * self.translation_scale * H)
        s = 1.0 + (torch.rand(1).item() - 0.5) * 2 * self.scale_range
        img = TF.affine(
            img, angle=0.0, translate=[dx, dy], scale=s, shear=[0.0, 0.0],
            interpolation=TF.InterpolationMode.BILINEAR, fill=fill_color,
        )

        return self._call_perspective_single(img.squeeze(0))


def parse_views(views):
    if isinstance(views, str):
        return [view.strip() for view in views.split(",") if view.strip()]
    return list(views)


def load_skin(path, bg_color=(128, 128, 128), normalize_model=True):
    skin = Image.open(path).convert("RGBA")
    if normalize_model and skin.getpixel((47, 52))[3] == 0:
        skin = alice_to_steve(skin)

    skin_np = np.array(skin, dtype=np.uint8)
    alpha_np = skin_np[..., 3]
    skin_np[(alpha_np > 0) & (alpha_np < 255), 3] = 255
    skin = Image.fromarray(skin_np)

    rgba = torch.from_numpy(np.array(skin, dtype="float32") / 255.0).permute(2, 0, 1)
    alpha = rgba[3:4]
    bg = torch.tensor(bg_color, dtype=rgba.dtype).view(3, 1, 1) / 255.0
    rgba[:3] = torch.where(alpha > 0, rgba[:3], bg)
    return rgba.clamp(0.0, 1.0)


def load_uv_mask():
    masks = load_uv_masks()
    if masks is None:
        return None
    base_mask, decor_mask = masks
    return torch.maximum(base_mask, decor_mask)


def load_uv_masks():
    mask_path = Path(__file__).resolve().parent / "skin-mask.png"
    decor_mask_path = Path(__file__).resolve().parent / "skin-decor-mask.png"
    if not mask_path.exists() or not decor_mask_path.exists():
        return None

    skin_mask = np.array(Image.open(mask_path).convert("RGBA"))
    skin_decor_mask = np.array(Image.open(decor_mask_path).convert("RGBA"))
    base_mask = torch.from_numpy(skin_mask[:, :, 3] > 0).float().unsqueeze(0)
    decor_mask = torch.from_numpy(skin_decor_mask[:, :, 3] > 0).float().unsqueeze(0)
    return base_mask, decor_mask


def apply_uv_mask(tensor):
    if tensor.shape[-3] != 4:
        raise ValueError(f"Expected RGBA tensor with 4 channels, got shape {tuple(tensor.shape)}.")

    uv_mask = load_uv_mask()
    if uv_mask is None:
        return tensor

    uv_mask = uv_mask.to(device=tensor.device, dtype=tensor.dtype)
    if tensor.dim() == 4:
        uv_mask = uv_mask.unsqueeze(0)
        out = tensor.clone()
        out[:, :3] = out[:, :3] * uv_mask
        out[:, 3:4] = out[:, 3:4] * uv_mask
        return out
    if tensor.dim() == 3:
        out = tensor.clone()
        out[:3] = out[:3] * uv_mask
        out[3:4] = out[3:4] * uv_mask
        return out
    raise ValueError(f"Expected CHW or NCHW tensor, got shape {tuple(tensor.shape)}.")


def finalize_minecraft_alpha(tensor, alpha_threshold=0.5, enforce_base_alpha=True):
    if tensor.shape[-3] != 4:
        raise ValueError(f"Expected RGBA tensor with 4 channels, got shape {tuple(tensor.shape)}.")

    masks = load_uv_masks()
    if masks is None:
        out = tensor.clone()
        out[..., 3:4, :, :] = (out[..., 3:4, :, :] > alpha_threshold).to(dtype=out.dtype)
        return out

    base_mask, decor_mask = masks
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


def view_native_size(renderer, view):
    mask = getattr(renderer, f"{view}_inner_mask")
    return tuple(mask.shape)


def _uv_indices_from_grid(grid, mask):
    coords = ((grid[mask] + 1.0) * 0.5 * (UV_SIZE - 1)).round().long()
    coords[:, 0].clamp_(0, UV_SIZE - 1)
    coords[:, 1].clamp_(0, UV_SIZE - 1)
    return coords[:, 1] * UV_SIZE + coords[:, 0]


def _get_unprojection_view_cache(renderer, view, device):
    cache = getattr(renderer, "_inverse_uv_unprojection_cache", None)
    if cache is None:
        cache = {}
        setattr(renderer, "_inverse_uv_unprojection_cache", cache)

    key = (view, device.type, device.index)
    if key in cache:
        return cache[key]

    inner_mask = getattr(renderer, f"{view}_inner_mask").to(device=device).bool()
    outer_mask = getattr(renderer, f"{view}_outer_mask").to(device=device).bool()
    geometry_mask = inner_mask | outer_mask

    layers = []
    for layer in CONDITIONING_LAYERS:
        mask = getattr(renderer, f"{view}_{layer}_mask").to(device=device).bool()
        grid = getattr(renderer, f"{view}_{layer}_grid").to(device=device, dtype=torch.float32)
        flat_uv = _uv_indices_from_grid(grid, mask)
        counts = torch.bincount(flat_uv, minlength=UV_SIZE * UV_SIZE).reshape(1, UV_SIZE, UV_SIZE)
        layers.append({
            "mask": mask,
            "flat_uv": flat_uv,
            "counts": counts,
        })

    cache[key] = {
        "geometry_mask": geometry_mask,
        "layers": layers,
    }
    return cache[key]


def _ensure_rgba(rendered, geometry_mask=None):
    is_batched = rendered.dim() == 4
    if is_batched:
        if rendered.shape[1] == 4:
            return rendered
        if rendered.shape[1] != 3:
            raise ValueError(f"Expected RGB or RGBA render tensor, got {rendered.shape[1]} channels.")
        if geometry_mask is None:
            alpha = torch.ones_like(rendered[:, :1])
        else:
            alpha = geometry_mask.to(dtype=rendered.dtype, device=rendered.device).unsqueeze(0).unsqueeze(1).expand(rendered.shape[0], 1, -1, -1)
        return torch.cat([rendered, alpha], dim=1)
    else:
        if rendered.shape[0] == 4:
            return rendered
        if rendered.shape[0] != 3:
            raise ValueError(f"Expected RGB or RGBA render tensor, got {rendered.shape[0]} channels.")
        if geometry_mask is None:
            alpha = torch.ones_like(rendered[:1])
        else:
            alpha = geometry_mask.to(dtype=rendered.dtype, device=rendered.device).unsqueeze(0)
        return torch.cat([rendered, alpha], dim=0)


def unproject_renders_to_uv(rendered_views, renderer, views, bg_color=(128, 128, 128), unproject_mode="mean"):
    views = parse_views(views)
    if len(rendered_views) != len(views):
        raise ValueError(f"Expected {len(views)} rendered views, got {len(rendered_views)}.")
    if unproject_mode not in ("mode", "mean", "medoid"):
        raise ValueError(f"Unsupported unproject_mode={unproject_mode!r}. Options: 'mode', 'mean', 'medoid'.")

    sample = rendered_views[0]
    is_batched = sample.dim() == 4
    device = sample.device
    dtype = sample.dtype

    if is_batched:
        if unproject_mode != "mean":
            raise ValueError(
                "Batched UV unprojection currently supports only unproject_mode='mean'. "
                "Use UNPROJECT_MODE=mean for training, or run unbatched inference for mode/medoid aggregation."
            )
        B = sample.shape[0]
        accum = sample.new_zeros(B, len(CONDITIONING_LAYERS), 4, UV_SIZE, UV_SIZE)
        counts_template = sample.new_zeros(len(CONDITIONING_LAYERS), 1, UV_SIZE, UV_SIZE)
        view_caches = [_get_unprojection_view_cache(renderer, view, device) for view in views]

        for view_cache in view_caches:
            for layer_index, layer_cache in enumerate(view_cache["layers"]):
                counts_template[layer_index] += layer_cache["counts"].to(device=device, dtype=dtype)

        for view_cache, rendered in zip(view_caches, rendered_views):
            rendered = _ensure_rgba(
                rendered.to(device=device, dtype=dtype),
                geometry_mask=view_cache["geometry_mask"],
            )

            for layer_index, layer_cache in enumerate(view_cache["layers"]):
                flat_uv = layer_cache["flat_uv"]
                if flat_uv.numel() == 0:
                    continue
                mask = layer_cache["mask"]
                values = rendered[:, :, mask]  # (B, 4, N)

                accum[:, layer_index].reshape(B, 4, -1).index_add_(2, flat_uv, values)

        counts = counts_template.unsqueeze(0)
        known = (counts > 0).to(dtype=dtype).expand(B, -1, -1, -1, -1)
        aggregated = accum / counts.clamp_min(1.0)
        bg = sample.new_tensor(bg_color, dtype=dtype).view(1, 1, 3, 1, 1) / 255.0
        rgb = torch.where(known.expand(-1, -1, 3, -1, -1) > 0, aggregated[:, :, :3], bg.expand_as(aggregated[:, :, :3]))
        alpha = torch.where(known > 0, aggregated[:, :, 3:4], torch.zeros_like(aggregated[:, :, 3:4]))
        layers = torch.cat([rgb, alpha, known], dim=2)
        return layers.reshape(B, -1, UV_SIZE, UV_SIZE).clamp(0.0, 1.0)

    # Unbatched path: supports mode/medoid/mean with per-texel aggregation
    accum = sample.new_zeros(len(CONDITIONING_LAYERS), 4, UV_SIZE, UV_SIZE)
    counts = sample.new_zeros(len(CONDITIONING_LAYERS), 1, UV_SIZE, UV_SIZE)

    layer_values_list = [[] for _ in CONDITIONING_LAYERS]
    layer_uv_list = [[] for _ in CONDITIONING_LAYERS]
    view_caches = [_get_unprojection_view_cache(renderer, view, device) for view in views]

    for view_cache, rendered in zip(view_caches, rendered_views):
        rendered = _ensure_rgba(
            rendered.to(device=device, dtype=dtype),
            geometry_mask=view_cache["geometry_mask"],
        )

        for layer_index, layer_cache in enumerate(view_cache["layers"]):
            flat_uv = layer_cache["flat_uv"]
            if flat_uv.numel() == 0:
                continue
            mask = layer_cache["mask"]
            values = rendered[:, mask]

            if unproject_mode == "mean":
                accum[layer_index].reshape(4, -1).index_add_(1, flat_uv, values)
                counts[layer_index] += layer_cache["counts"].to(device=device, dtype=dtype)
            else:
                layer_values_list[layer_index].append(values)
                layer_uv_list[layer_index].append(flat_uv)

    if unproject_mode == "mean":
        known = (counts > 0).to(dtype=dtype)
        aggregated = accum / counts.clamp_min(1.0)
    else:
        aggregated = sample.new_zeros(len(CONDITIONING_LAYERS), 4, UV_SIZE, UV_SIZE)
        counts = sample.new_zeros(len(CONDITIONING_LAYERS), 1, UV_SIZE, UV_SIZE)

        for layer_index in range(len(CONDITIONING_LAYERS)):
            if not layer_values_list[layer_index]:
                continue
            all_values = torch.cat(layer_values_list[layer_index], dim=1)
            all_flat_uv = torch.cat(layer_uv_list[layer_index], dim=0)

            counts[layer_index] = torch.bincount(
                all_flat_uv,
                minlength=UV_SIZE * UV_SIZE,
            ).reshape(1, UV_SIZE, UV_SIZE).to(device=device, dtype=dtype)

            unique_uvs = torch.unique(all_flat_uv)
            flat_aggregated = aggregated[layer_index].reshape(4, -1)

            if unproject_mode == "mode":
                quant = (all_values.clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.int64)
                keys = quant[0] | (quant[1] << 8) | (quant[2] << 16) | (quant[3] << 24)
                for u_idx in unique_uvs:
                    mask_u = (all_flat_uv == u_idx)
                    texel_keys = keys[mask_u]
                    if texel_keys.numel() == 1:
                        mode_key = texel_keys[0].item()
                    else:
                        uniques, cnts = torch.unique(texel_keys, return_counts=True)
                        mode_key = uniques[cnts.argmax()].item()
                    r = (mode_key & 255) / 255.0
                    g = ((mode_key >> 8) & 255) / 255.0
                    b = ((mode_key >> 16) & 255) / 255.0
                    a = ((mode_key >> 24) & 255) / 255.0
                    flat_aggregated[:, u_idx] = torch.tensor([r, g, b, a], dtype=dtype, device=device)
            elif unproject_mode == "medoid":
                for u_idx in unique_uvs:
                    mask_u = (all_flat_uv == u_idx)
                    texel_vals = all_values[:, mask_u]
                    if texel_vals.shape[1] == 1:
                        medoid_val = texel_vals[:, 0]
                    else:
                        dists = torch.cdist(texel_vals.t(), texel_vals.t(), p=2)
                        medoid_val = texel_vals[:, dists.sum(dim=1).argmin()]
                    flat_aggregated[:, u_idx] = medoid_val

        known = (counts > 0).to(dtype=dtype)

    bg = sample.new_tensor(bg_color, dtype=dtype).view(1, 3, 1, 1) / 255.0
    rgb = torch.where(known.expand(-1, 3, -1, -1) > 0, aggregated[:, :3], bg.expand_as(aggregated[:, :3]))
    alpha = torch.where(known > 0, aggregated[:, 3:4], torch.zeros_like(aggregated[:, 3:4]))
    layers = torch.cat([rgb, alpha, known], dim=1)
    return layers.reshape(-1, UV_SIZE, UV_SIZE).clamp(0.0, 1.0)


def build_conditioning(
    skin,
    renderer,
    views,
    image_size=256,
    include_alpha=False,
    unproject_mode="mean",
    augmenter=None,
    return_renders=False,
):
    _ = image_size, include_alpha
    is_batched = skin.dim() == 4
    if not is_batched:
        skin_batch = skin.unsqueeze(0)
    else:
        skin_batch = skin

    rendered_views = []
    gt_renders = {} if return_renders else None
    with torch.no_grad():
        for view in views:
            rendered = renderer.forward_view(skin_batch, view)
            if not is_batched:
                rendered = rendered.squeeze(0)
            if return_renders:
                gt_renders[view] = rendered  # save clean renders for loss reuse
            if augmenter is not None:
                rendered = augmenter(rendered)
            rendered_views.append(rendered)
    conditioning = unproject_renders_to_uv(rendered_views, renderer, views, unproject_mode=unproject_mode)
    if return_renders:
        return conditioning, gt_renders
    return conditioning


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
        unproject_mode="mean",
        augment=False,
        translation_scale=0.0,
        scale_range=0.0,
        perspective_scale=0.0,
    ):
        self.data_dir = data_dir
        self.views = parse_views(views)
        self.image_size = image_size
        self.include_alpha = include_alpha
        self.bg_color = bg_color
        self.normalize_model = normalize_model
        self.unproject_mode = unproject_mode
        self.augment = augment
        self.translation_scale = translation_scale
        self.scale_range = scale_range
        self.perspective_scale = perspective_scale
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
        return {
            "uv": uv,
            "path": skin_path,
        }


def tensor_to_rgba_image(tensor):
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    if tensor.shape[0] == 3:
        tensor = torch.cat([tensor, torch.ones_like(tensor[:1])], dim=0)
    return TF.to_pil_image(tensor)
