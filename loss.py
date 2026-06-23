import os
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from renderer import DifferentiableRenderer
except ImportError:
    from .renderer import DifferentiableRenderer

class MinecraftLoss(nn.Module):
    def __init__(
        self,
        mappings_dir=None,
        bg_color=(128/255, 128/255, 128/255),
        lambda_uv=1.0,
        lambda_render=1.0,
        lambda_lpips=0.1,
        use_lpips=False,
        views=None,
        foreground_weight=1.0
    ):
        """
        Custom training loss combining flat UV loss and multi-view rendering loss.
        """
        super().__init__()
        self.renderer = DifferentiableRenderer(mappings_dir=mappings_dir, bg_color=bg_color)
        self.lambda_uv = lambda_uv
        self.lambda_render = lambda_render
        self.lambda_lpips = lambda_lpips
        self.foreground_weight = foreground_weight
        
        # Configure which views to compute loss for
        if views is not None:
            if isinstance(views, str):
                views = [v.strip() for v in views.split(",") if v.strip()]
            self.views = [v for v in views if v in self.renderer.views]
            missing_views = [v for v in views if v not in self.renderer.views]
            if missing_views:
                print(f"WARNING: Ignoring unknown Minecraft render views: {missing_views}")
        else:
            self.views = self.renderer.views
            
        self.view_count = max(1, len(self.views))
        
        # Optionally load LPIPS model
        self.lpips_loss_fn = None
        if use_lpips:
            try:
                import lpips
                # Use standard AlexNet back-end for fast perceptual distance
                self.lpips_loss_fn = lpips.LPIPS(net='alex')
                self.lpips_loss_fn.eval()
                # Do not train LPIPS weights
                self.lpips_loss_fn.requires_grad_(False)
                print("LPIPS loss module loaded successfully for training.")
            except ImportError:
                print("WARNING: 'lpips' package not found. Falling back to MSE/L1 rendering loss.")
                print("Please install lpips via: pip install lpips")

    def forward(self, skins_pred, skins_gt):
        """
        Calculates the combined training loss (UV Loss + Render Loss).
        Args:
            skins_pred: Predicted skin tensor of shape (B, 4, 64, 64) in range [0, 1].
            skins_gt: Ground truth skin tensor of shape (B, 4, 64, 64) in range [0, 1].
        Returns:
            dict containing individual loss terms and total loss.
        """
        assert skins_pred.shape == skins_gt.shape, f"Shape mismatch: {skins_pred.shape} vs {skins_gt.shape}"
        
        # 1. Flat UV Loss (Reconstruction)
        # We calculate MSE for RGB and Alpha channel of the flat 64x64 skin
        loss_uv = F.mse_loss(skins_pred, skins_gt)
        
        # 2. Rendering Loss across selected views
        loss_mse_render = torch.tensor(0.0, device=skins_pred.device, dtype=skins_pred.dtype)
        loss_lpips_render = torch.tensor(0.0, device=skins_pred.device, dtype=skins_pred.dtype)
        
        if self.lambda_render > 0 and self.view_count > 0:
            for view in self.views:
                # Render predicted skin and GT skin for the view
                pred_view = self.renderer.forward_view(skins_pred, view) # (B, 4, H, W)
                gt_view = self.renderer.forward_view(skins_gt, view)     # (B, 4, H, W)
                
                # Full frame MSE (on RGB channels)
                full_mse = F.mse_loss(pred_view[:, :3], gt_view[:, :3])
                
                # Foreground-focused MSE (places higher weights on the character, ignoring empty background)
                if self.foreground_weight > 0:
                    # Construct mask of the union of foreground pixels (where either predicted or GT is non-transparent)
                    fg_mask = torch.maximum(pred_view[:, 3:4], gt_view[:, 3:4]).detach()
                    fg_denom = fg_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
                    fg_mse = (((pred_view[:, :3] - gt_view[:, :3]) ** 2) * fg_mask).sum(dim=(1, 2, 3)) / (fg_denom * 3.0)
                    loss_mse_render += full_mse + self.foreground_weight * fg_mse.mean()
                else:
                    loss_mse_render += full_mse
                    
                # Optional LPIPS perceptual loss on rendered view
                if self.lpips_loss_fn is not None:
                    # LPIPS expects inputs in [-1, 1] range
                    pred_rgb = pred_view[:, :3] * 2.0 - 1.0
                    gt_rgb = gt_view[:, :3] * 2.0 - 1.0
                    loss_lpips_render += self.lpips_loss_fn(pred_rgb, gt_rgb).mean().to(dtype=skins_pred.dtype)
            
            # Average rendering losses over all views
            loss_mse_render = loss_mse_render / self.view_count
            loss_lpips_render = loss_lpips_render / self.view_count
            
        # Combine losses
        loss_total_render = loss_mse_render + self.lambda_lpips * loss_lpips_render
        loss_total = self.lambda_uv * loss_uv + self.lambda_render * loss_total_render
        
        return {
            "loss_total": loss_total,
            "loss_uv": loss_uv,
            "loss_render_mse": loss_mse_render,
            "loss_render_lpips": loss_lpips_render,
            "loss_render_total": loss_total_render
        }
