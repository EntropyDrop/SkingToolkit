import sys
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TOOLKIT_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from SkingToolkit.renderer import DifferentiableRenderer  # noqa: E402


def parse_views(views):
    if isinstance(views, str):
        return [view.strip() for view in views.split(",") if view.strip()]
    return list(views)


def gaussian(window_size, sigma):
    gauss = torch.exp(torch.tensor([-(x - window_size//2)**2 / (2 * sigma**2) for x in range(window_size)], dtype=torch.float32))
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def masked_ssim_loss(pred_rgb, gt_rgb, mask=None, window_size=11):
    device = pred_rgb.device
    dtype = pred_rgb.dtype
    channel = 3
    window = create_window(window_size, channel).to(device=device, dtype=dtype)
    
    mu1 = F.conv2d(pred_rgb, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(gt_rgb, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(pred_rgb * pred_rgb, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(gt_rgb * gt_rgb, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred_rgb * gt_rgb, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    loss_map = 1.0 - ssim_map
    
    if mask is not None:
        mask = mask.to(device=device, dtype=dtype)
        mask_expanded = mask.expand(-1, channel, -1, -1)
        denom = mask_expanded.sum(dim=(1, 2, 3)).clamp_min(1.0)
        per_sample = (loss_map * mask_expanded).sum(dim=(1, 2, 3)) / denom
        return per_sample.mean()
    else:
        return loss_map.mean()


def alpha_masked_rgb_l1(pred_uv, gt_uv, uv_mask=None):
    alpha_gt = gt_uv[:, 3:4].detach()
    if uv_mask is not None:
        alpha_gt = alpha_gt * uv_mask
    denom = (alpha_gt.sum(dim=(1, 2, 3)) * 3.0).clamp_min(1.0)
    per_sample = ((pred_uv[:, :3] - gt_uv[:, :3]).abs() * alpha_gt).sum(dim=(1, 2, 3)) / denom
    return per_sample.mean()


def alpha_bce(pred_uv, gt_uv, uv_mask=None):
    pred_alpha = pred_uv[:, 3:4].clamp(1e-4, 1.0 - 1e-4)
    loss = F.binary_cross_entropy(pred_alpha, gt_uv[:, 3:4], reduction="none")
    if uv_mask is None:
        return loss.mean()

    uv_mask = uv_mask.to(device=loss.device, dtype=loss.dtype)
    if uv_mask.shape != loss.shape:
        uv_mask = uv_mask.expand_as(loss)
    denom = uv_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return ((loss * uv_mask).sum(dim=(1, 2, 3)) / denom).mean()


def minecraft_layer_rects(is_slim=False):
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
            rects.append((inner_x, inner_y, width, height, decor_offset[0], decor_offset[1]))
    return rects


def covered_inner_mask(gt_uv, rects, alpha_threshold=0.5):
    outer_alpha = gt_uv[:, 3:4].detach()
    covered = gt_uv.new_zeros((gt_uv.shape[0], 1, gt_uv.shape[2], gt_uv.shape[3]))

    for inner_x, inner_y, width, height, decor_dx, decor_dy in rects:
        outer_x = inner_x + decor_dx
        outer_y = inner_y + decor_dy
        outer_visible = outer_alpha[:, :, outer_y : outer_y + height, outer_x : outer_x + width] > alpha_threshold
        inner_slice = covered[:, :, inner_y : inner_y + height, inner_x : inner_x + width]
        covered[:, :, inner_y : inner_y + height, inner_x : inner_x + width] = torch.maximum(
            inner_slice,
            outer_visible.to(dtype=covered.dtype),
        )
    return covered


def edge_l1(pred_uv, gt_uv, uv_mask=None):
    alpha_gt = gt_uv[:, 3:4].detach()
    if uv_mask is not None:
        alpha_gt = alpha_gt * uv_mask

    pred_rgb = pred_uv[:, :3]
    gt_rgb = gt_uv[:, :3]
    dx_mask = alpha_gt[:, :, :, 1:] * alpha_gt[:, :, :, :-1]
    dy_mask = alpha_gt[:, :, 1:, :] * alpha_gt[:, :, :-1, :]

    dx = (pred_rgb[:, :, :, 1:] - pred_rgb[:, :, :, :-1]) - (gt_rgb[:, :, :, 1:] - gt_rgb[:, :, :, :-1])
    dy = (pred_rgb[:, :, 1:, :] - pred_rgb[:, :, :-1, :]) - (gt_rgb[:, :, 1:, :] - gt_rgb[:, :, :-1, :])
    dx_loss = dx.abs() * dx_mask
    dy_loss = dy.abs() * dy_mask
    denom = ((dx_mask.sum() + dy_mask.sum()) * 3.0).clamp_min(1.0)
    return (dx_loss.sum() + dy_loss.sum()) / denom


class InverseUVLoss(nn.Module):
    def __init__(
        self,
        mappings_dir=None,
        views="static_front,static_back",
        bg_color=(128 / 255, 128 / 255, 128 / 255),
        lambda_rgb=1.0,
        lambda_alpha=0.5,
        lambda_render=0.1,
        lambda_edge=0.25,
        lambda_ssim=0.2,
        ssim_window_size=11,
        render_foreground_weight=1.0,
        ignore_covered_inner=True,
        covered_inner_alpha_threshold=0.5,
    ):
        super().__init__()
        self.lambda_rgb = lambda_rgb
        self.lambda_alpha = lambda_alpha
        self.lambda_render = lambda_render
        self.lambda_edge = lambda_edge
        self.lambda_ssim = lambda_ssim
        self.ssim_window_size = ssim_window_size
        self.render_foreground_weight = render_foreground_weight
        self.ignore_covered_inner = ignore_covered_inner
        self.covered_inner_alpha_threshold = covered_inner_alpha_threshold
        self.covered_inner_rects = minecraft_layer_rects(is_slim=False)

        self.renderer = DifferentiableRenderer(mappings_dir=mappings_dir, bg_color=bg_color)
        self.views = parse_views(views)
        missing_views = [view for view in self.views if view not in self.renderer.views]
        if missing_views:
            raise ValueError(
                f"Unknown renderer views {missing_views}. "
                f"Available views: {', '.join(self.renderer.views)}"
            )

        # Load UV mask
        mask_path = Path(__file__).resolve().parent / "skin-mask.png"
        decor_mask_path = Path(__file__).resolve().parent / "skin-decor-mask.png"
        
        if mask_path.exists() and decor_mask_path.exists():
            skin_mask = np.array(Image.open(mask_path).convert("RGBA"))
            skin_decor_mask = np.array(Image.open(decor_mask_path).convert("RGBA"))
            valid_mask = (skin_mask[:, :, 3] > 0) | (skin_decor_mask[:, :, 3] > 0)
            uv_mask = torch.from_numpy(valid_mask).float().unsqueeze(0).unsqueeze(0)
            self.register_buffer("uv_mask", uv_mask)
        else:
            print("WARNING: UV masks not found, falling back to full UV loss.")
            self.uv_mask = None

    def render_loss(self, pred_uv, gt_uv, gt_renders=None):
        if self.lambda_render <= 0:
            return pred_uv.new_tensor(0.0)

        total = pred_uv.new_tensor(0.0)
        for view in self.views:
            pred_render = self.renderer.forward_view(pred_uv, view)
            if gt_renders is not None and view in gt_renders:
                gt_render = gt_renders[view]
            else:
                with torch.no_grad():
                    gt_render = self.renderer.forward_view(gt_uv, view)
            fg_mask = torch.maximum(pred_render[:, 3:4], gt_render[:, 3:4]).detach()
            if self.render_foreground_weight > 0:
                denom = (fg_mask.sum(dim=(1, 2, 3)) * 3.0).clamp_min(1.0)
                per_sample = ((pred_render[:, :3] - gt_render[:, :3]).abs() * fg_mask).sum(dim=(1, 2, 3)) / denom
                total = total + per_sample.mean() * self.render_foreground_weight
            else:
                total = total + F.l1_loss(pred_render[:, :3], gt_render[:, :3])
        return total / max(len(self.views), 1)

    def forward(self, pred_uv, gt_uv, gt_renders=None):
        uv_mask = getattr(self, "uv_mask", None)
        if self.ignore_covered_inner:
            supervised_inner_mask = 1.0 - covered_inner_mask(
                gt_uv,
                self.covered_inner_rects,
                alpha_threshold=self.covered_inner_alpha_threshold,
            )
            uv_mask = supervised_inner_mask if uv_mask is None else uv_mask * supervised_inner_mask
        loss_rgb = alpha_masked_rgb_l1(pred_uv, gt_uv, uv_mask)
        loss_alpha = alpha_bce(pred_uv, gt_uv, uv_mask)
        loss_render = self.render_loss(pred_uv, gt_uv, gt_renders=gt_renders)
        loss_edge = edge_l1(pred_uv, gt_uv, uv_mask)
        
        if self.lambda_ssim > 0:
            alpha_gt = gt_uv[:, 3:4].detach()
            if uv_mask is not None:
                alpha_gt = alpha_gt * uv_mask
            loss_ssim = masked_ssim_loss(pred_uv[:, :3], gt_uv[:, :3], alpha_gt, window_size=self.ssim_window_size)
        else:
            loss_ssim = pred_uv.new_tensor(0.0)

        loss_total = (
            self.lambda_rgb * loss_rgb
            + self.lambda_alpha * loss_alpha
            + self.lambda_render * loss_render
            + self.lambda_edge * loss_edge
            + self.lambda_ssim * loss_ssim
        )
        return {
            "loss_total": loss_total,
            "loss_rgb": loss_rgb,
            "loss_alpha": loss_alpha,
            "loss_render": loss_render,
            "loss_edge": loss_edge,
            "loss_ssim": loss_ssim,
        }
