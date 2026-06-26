import sys
from pathlib import Path

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


def alpha_masked_rgb_l1(pred_uv, gt_uv):
    alpha_gt = gt_uv[:, 3:4].detach()
    denom = (alpha_gt.sum(dim=(1, 2, 3)) * 3.0).clamp_min(1.0)
    per_sample = ((pred_uv[:, :3] - gt_uv[:, :3]).abs() * alpha_gt).sum(dim=(1, 2, 3)) / denom
    return per_sample.mean()


def alpha_bce(pred_uv, gt_uv):
    pred_alpha = pred_uv[:, 3:4].clamp(1e-4, 1.0 - 1e-4)
    return F.binary_cross_entropy(pred_alpha, gt_uv[:, 3:4])


class InverseUVLoss(nn.Module):
    def __init__(
        self,
        mappings_dir=None,
        views="static_front,static_back",
        bg_color=(128 / 255, 128 / 255, 128 / 255),
        lambda_rgb=1.0,
        lambda_alpha=0.5,
        lambda_render=0.1,
        render_foreground_weight=1.0,
    ):
        super().__init__()
        self.lambda_rgb = lambda_rgb
        self.lambda_alpha = lambda_alpha
        self.lambda_render = lambda_render
        self.render_foreground_weight = render_foreground_weight

        self.renderer = DifferentiableRenderer(mappings_dir=mappings_dir, bg_color=bg_color)
        self.views = parse_views(views)
        missing_views = [view for view in self.views if view not in self.renderer.views]
        if missing_views:
            raise ValueError(
                f"Unknown renderer views {missing_views}. "
                f"Available views: {', '.join(self.renderer.views)}"
            )

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
        loss_rgb = alpha_masked_rgb_l1(pred_uv, gt_uv)
        loss_alpha = alpha_bce(pred_uv, gt_uv)
        loss_render = self.render_loss(pred_uv, gt_uv)
        loss_total = (
            self.lambda_rgb * loss_rgb
            + self.lambda_alpha * loss_alpha
            + self.lambda_render * loss_render
        )
        return {
            "loss_total": loss_total,
            "loss_rgb": loss_rgb,
            "loss_alpha": loss_alpha,
            "loss_render": loss_render,
        }
