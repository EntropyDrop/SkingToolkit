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


def alpha_masked_rgb_l1(pred_uv, gt_uv, uv_mask=None):
    alpha_gt = gt_uv[:, 3:4].detach()
    if uv_mask is not None:
        alpha_gt = alpha_gt * uv_mask
    denom = (alpha_gt.sum(dim=(1, 2, 3)) * 3.0).clamp_min(1.0)
    per_sample = ((pred_uv[:, :3] - gt_uv[:, :3]).abs() * alpha_gt).sum(dim=(1, 2, 3)) / denom
    return per_sample.mean()


def alpha_bce(pred_uv, gt_uv, uv_mask=None):
    _ = uv_mask
    pred_alpha = pred_uv[:, 3:4].clamp(1e-4, 1.0 - 1e-4)
    return F.binary_cross_entropy(pred_alpha, gt_uv[:, 3:4])


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
        render_foreground_weight=1.0,
    ):
        super().__init__()
        self.lambda_rgb = lambda_rgb
        self.lambda_alpha = lambda_alpha
        self.lambda_render = lambda_render
        self.lambda_edge = lambda_edge
        self.render_foreground_weight = render_foreground_weight

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

    def render_loss(self, pred_uv, gt_uv):
        if self.lambda_render <= 0:
            return pred_uv.new_tensor(0.0)

        total = pred_uv.new_tensor(0.0)
        for view in self.views:
            pred_render = self.renderer.forward_view(pred_uv, view)
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

    def forward(self, pred_uv, gt_uv):
        uv_mask = getattr(self, "uv_mask", None)
        loss_rgb = alpha_masked_rgb_l1(pred_uv, gt_uv, uv_mask)
        loss_alpha = alpha_bce(pred_uv, gt_uv, uv_mask)
        loss_render = self.render_loss(pred_uv, gt_uv)
        loss_edge = edge_l1(pred_uv, gt_uv, uv_mask)
        loss_total = (
            self.lambda_rgb * loss_rgb
            + self.lambda_alpha * loss_alpha
            + self.lambda_render * loss_render
            + self.lambda_edge * loss_edge
        )
        return {
            "loss_total": loss_total,
            "loss_rgb": loss_rgb,
            "loss_alpha": loss_alpha,
            "loss_render": loss_render,
            "loss_edge": loss_edge,
        }
