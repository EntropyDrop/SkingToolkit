import torch
import torch.nn as nn
import torch.nn.functional as F

from SkingToolkit.semantic_uv_reconstruction.dataset import load_uv_masks
from SkingToolkit.semantic_uv_reconstruction.losses import (
    alpha_dice_loss,
    alpha_masked_rgb_l1,
    edge_l1,
    minecraft_layer_rects,
)


IGNORE_INDEX = 255
AUTO_SEMANTIC_CLASSES = 13


def build_part_layer_masks():
    """Return exact Steve inner/outer atlas masks grouped into six body parts."""
    inner = torch.zeros(6, 1, 64, 64, dtype=torch.float32)
    outer = torch.zeros_like(inner)
    for rect_index, (inner_x, inner_y, width, height, decor_dx, decor_dy) in enumerate(
        minecraft_layer_rects(is_slim=False)
    ):
        part = rect_index // 6
        inner[part, :, inner_y : inner_y + height, inner_x : inner_x + width] = 1.0
        outer_x = inner_x + decor_dx
        outer_y = inner_y + decor_dy
        outer[part, :, outer_y : outer_y + height, outer_x : outer_x + width] = 1.0
    return inner, outer


def build_auto_semantic_uv_target(target_uv, inner_part_masks, outer_part_masks):
    """Build exact occupied layer/part labels from the source skin atlas.

    Class 0 is a valid but transparent atlas texel, classes 1..6 are occupied
    inner parts, and classes 7..12 are occupied outer parts. Invalid atlas
    space is ignored with class 255.
    """
    batch = target_uv.shape[0]
    alpha = target_uv[:, 3] > 0.5
    inner_masks = inner_part_masks[:, 0].bool()
    outer_masks = outer_part_masks[:, 0].bool()
    valid = inner_masks.any(dim=0) | outer_masks.any(dim=0)
    labels = torch.full(
        (batch, 64, 64),
        IGNORE_INDEX,
        dtype=torch.long,
        device=target_uv.device,
    )
    labels[:, valid] = 0
    for part in range(6):
        labels[alpha & inner_masks[part].unsqueeze(0)] = part + 1
        labels[alpha & outer_masks[part].unsqueeze(0)] = part + 7
    return labels


def build_semantic_attribute_targets(target_uv, inner_part_masks, outer_part_masks):
    alpha = target_uv[:, 3:4].float()
    rgb = target_uv[:, :3].float()
    outer_masks = outer_part_masks.unsqueeze(0).to(device=target_uv.device)
    outer_weight = alpha.unsqueeze(1) * outer_masks
    outer_area = outer_masks.sum(dim=(2, 3, 4)).clamp_min(1.0)
    outer_coverage = outer_weight.sum(dim=(2, 3, 4)) / outer_area
    outer_presence = (outer_coverage > 0.0).float()

    all_masks = torch.cat([inner_part_masks, outer_part_masks], dim=0)
    all_masks = all_masks.unsqueeze(0).to(device=target_uv.device)
    color_weight = alpha.unsqueeze(1) * all_masks
    color_denominator = color_weight.sum(dim=(3, 4)).clamp_min(1.0)
    part_colors = (rgb.unsqueeze(1) * color_weight).sum(dim=(3, 4)) / color_denominator
    color_known = color_weight.sum(dim=(2, 3, 4)) > 0.0
    return {
        "outer_presence": outer_presence,
        "outer_coverage": outer_coverage,
        "part_colors": part_colors,
        "part_colors_known": color_known,
    }


class SemanticUVReconstructionLoss(nn.Module):
    def __init__(
        self,
        lambda_uv_rgb=2.0,
        lambda_uv_edge=1.0,
        lambda_outer_alpha=1.0,
        lambda_outer_dice=0.5,
        lambda_semantic_uv=0.25,
        lambda_semantic_presence=0.25,
        lambda_semantic_coverage=0.25,
        lambda_semantic_color=0.25,
        lambda_render_rgb=0.5,
        lambda_render_alpha=0.5,
        lambda_siglip_render=0.1,
    ):
        super().__init__()
        self.lambda_uv_rgb = float(lambda_uv_rgb)
        self.lambda_uv_edge = float(lambda_uv_edge)
        self.lambda_outer_alpha = float(lambda_outer_alpha)
        self.lambda_outer_dice = float(lambda_outer_dice)
        self.lambda_semantic_uv = float(lambda_semantic_uv)
        self.lambda_semantic_presence = float(lambda_semantic_presence)
        self.lambda_semantic_coverage = float(lambda_semantic_coverage)
        self.lambda_semantic_color = float(lambda_semantic_color)
        self.lambda_render_rgb = float(lambda_render_rgb)
        self.lambda_render_alpha = float(lambda_render_alpha)
        self.lambda_siglip_render = float(lambda_siglip_render)

        weights = (
            self.lambda_uv_rgb,
            self.lambda_uv_edge,
            self.lambda_outer_alpha,
            self.lambda_outer_dice,
            self.lambda_semantic_uv,
            self.lambda_semantic_presence,
            self.lambda_semantic_coverage,
            self.lambda_semantic_color,
            self.lambda_render_rgb,
            self.lambda_render_alpha,
            self.lambda_siglip_render,
        )
        if any(weight < 0.0 for weight in weights):
            raise ValueError("Semantic UV loss weights must be non-negative.")

        masks = load_uv_masks()
        if masks is None:
            raise FileNotFoundError("skin-mask.png and skin-decor-mask.png are required.")
        base_mask, decor_mask = masks
        inner_part_masks, outer_part_masks = build_part_layer_masks()
        self.register_buffer("base_mask", base_mask.unsqueeze(0), persistent=False)
        self.register_buffer("decor_mask", decor_mask.unsqueeze(0), persistent=False)
        self.register_buffer("inner_part_masks", inner_part_masks, persistent=False)
        self.register_buffer("outer_part_masks", outer_part_masks, persistent=False)

    def forward(
        self,
        outputs,
        target_uv,
        gt_renders=None,
        renderer=None,
        views=None,
        semantic_uv_target=None,
        semantic_encoder=None,
        compute_siglip_render=True,
        siglip_render_scale=1.0,
    ):
        target_uv = target_uv.float()
        pred_uv = outputs["uv"]
        zero = pred_uv.new_zeros((), dtype=torch.float32)
        valid_mask = (self.base_mask + self.decor_mask).clamp(0.0, 1.0)
        loss_uv_rgb = alpha_masked_rgb_l1(pred_uv, target_uv, valid_mask)
        loss_uv_edge = edge_l1(pred_uv, target_uv, valid_mask)

        target_alpha = target_uv[:, 3:4]
        alpha_bce = F.binary_cross_entropy_with_logits(
            outputs["alpha_logits"].float(), target_alpha, reduction="none"
        )
        decor_mask = self.decor_mask.to(dtype=alpha_bce.dtype).expand_as(alpha_bce)
        loss_outer_alpha = (
            (alpha_bce * decor_mask).sum(dim=(1, 2, 3))
            / decor_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
        ).mean()
        loss_outer_dice = alpha_dice_loss(pred_uv, target_uv, self.decor_mask)

        attributes = build_semantic_attribute_targets(
            target_uv, self.inner_part_masks, self.outer_part_masks
        )
        loss_semantic_presence = F.binary_cross_entropy_with_logits(
            outputs["outer_presence_logits"].float(), attributes["outer_presence"]
        )
        loss_semantic_coverage = F.smooth_l1_loss(
            outputs["outer_coverage"].float(), attributes["outer_coverage"]
        )
        color_known = attributes["part_colors_known"].unsqueeze(-1).to(dtype=torch.float32)
        loss_semantic_color = (
            (outputs["part_colors"].float() - attributes["part_colors"]).abs() * color_known
        ).sum() / (color_known.sum() * 3.0).clamp_min(1.0)

        if "semantic_uv_logits" in outputs:
            if semantic_uv_target is None:
                if outputs["semantic_uv_logits"].shape[1] != AUTO_SEMANTIC_CLASSES:
                    raise ValueError(
                        "Automatic semantic UV supervision requires 13 classes; "
                        "provide --semantic_labels_dir for a custom vocabulary."
                    )
                semantic_uv_target = build_auto_semantic_uv_target(
                    target_uv, self.inner_part_masks, self.outer_part_masks
                )
            semantic_uv_target = semantic_uv_target.long()
            valid_semantic = semantic_uv_target != IGNORE_INDEX
            if valid_semantic.any():
                maximum = int(semantic_uv_target[valid_semantic].max().item())
                if maximum >= outputs["semantic_uv_logits"].shape[1]:
                    raise ValueError(
                        f"Semantic UV label {maximum} exceeds the configured "
                        f"{outputs['semantic_uv_logits'].shape[1]} classes."
                    )
            loss_semantic_uv = F.cross_entropy(
                outputs["semantic_uv_logits"].float(),
                semantic_uv_target,
                ignore_index=IGNORE_INDEX,
            )
        else:
            loss_semantic_uv = zero

        loss_render_rgb = zero
        loss_render_alpha = zero
        loss_siglip_render = zero
        render_losses_enabled = (
            self.lambda_render_rgb > 0.0
            or self.lambda_render_alpha > 0.0
            or (self.lambda_siglip_render > 0.0 and compute_siglip_render)
        )
        if render_losses_enabled:
            if renderer is None or views is None or gt_renders is None:
                raise ValueError("Render losses require renderer, views, and gt_renders.")
            render_rgb_total = zero
            render_alpha_total = zero
            predicted_renders = []
            for view_index, view in enumerate(views):
                pred_render = renderer.forward_view(pred_uv, view)
                predicted_renders.append(pred_render)
                gt_render = gt_renders[:, view_index].to(dtype=pred_render.dtype)
                foreground = gt_render[:, 3:4].detach()
                render_rgb_total = render_rgb_total + (
                    ((pred_render[:, :3] - gt_render[:, :3]).abs() * foreground).sum()
                    / (foreground.sum() * 3.0).clamp_min(1.0)
                )
                render_alpha_total = render_alpha_total + F.l1_loss(
                    pred_render[:, 3:4].float(), gt_render[:, 3:4].float()
                )
            view_count = max(len(views), 1)
            loss_render_rgb = render_rgb_total / view_count
            loss_render_alpha = render_alpha_total / view_count
            if self.lambda_siglip_render > 0.0 and compute_siglip_render:
                if semantic_encoder is None or "open_semantic_embedding" not in outputs:
                    raise ValueError(
                        "SigLIP render loss requires an open semantic backbone and "
                        "outputs['open_semantic_embedding']."
                    )
                predicted_renders = torch.stack(predicted_renders, dim=1)
                predicted_embedding = semantic_encoder(predicted_renders[:, :, :3])
                target_embedding = outputs.get(
                    "open_semantic_view_embeddings", outputs["open_semantic_embedding"]
                ).detach()
                loss_siglip_render = (
                    1.0
                    - F.cosine_similarity(
                        predicted_embedding.float(), target_embedding.float(), dim=-1
                    )
                ).mean()
                loss_siglip_render = loss_siglip_render * float(siglip_render_scale)

        loss_semantic = (
            self.lambda_semantic_uv * loss_semantic_uv
            + self.lambda_semantic_presence * loss_semantic_presence
            + self.lambda_semantic_coverage * loss_semantic_coverage
            + self.lambda_semantic_color * loss_semantic_color
        )
        loss_total = (
            self.lambda_uv_rgb * loss_uv_rgb
            + self.lambda_uv_edge * loss_uv_edge
            + self.lambda_outer_alpha * loss_outer_alpha
            + self.lambda_outer_dice * loss_outer_dice
            + loss_semantic
            + self.lambda_render_rgb * loss_render_rgb
            + self.lambda_render_alpha * loss_render_alpha
            + self.lambda_siglip_render * loss_siglip_render
        )

        outer_target = (target_alpha > 0.5) & self.decor_mask.bool()
        outer_pred = (pred_uv[:, 3:4] > 0.5) & self.decor_mask.bool()
        count_outer_tp = (outer_pred & outer_target).sum().float()
        count_outer_fp = (outer_pred & ~outer_target).sum().float()
        count_outer_fn = (~outer_pred & outer_target).sum().float()
        presence_pred = outputs["outer_presence_logits"] > 0.0
        presence_target = attributes["outer_presence"] > 0.5

        return {
            "loss_total": loss_total,
            "loss_uv_rgb": loss_uv_rgb,
            "loss_uv_edge": loss_uv_edge,
            "rgb_mae_255": loss_uv_rgb.detach() * 255.0,
            "loss_outer_alpha": loss_outer_alpha,
            "loss_outer_dice": loss_outer_dice,
            "loss_semantic": loss_semantic,
            "loss_semantic_uv": loss_semantic_uv,
            "loss_semantic_presence": loss_semantic_presence,
            "loss_semantic_coverage": loss_semantic_coverage,
            "loss_semantic_color": loss_semantic_color,
            "loss_render_rgb": loss_render_rgb,
            "loss_render_alpha": loss_render_alpha,
            "loss_siglip_render": loss_siglip_render,
            "acc_outer_part_presence": (presence_pred == presence_target).float().mean(),
            "count_outer_tp": count_outer_tp,
            "count_outer_fp": count_outer_fp,
            "count_outer_fn": count_outer_fn,
        }
