import torch
import torch.nn as nn
import torch.nn.functional as F

from SkingToolkit.dense_uv_parser.utils import IGNORE_INDEX


class DenseUVParserLoss(nn.Module):
    def __init__(
        self,
        lambda_foreground=1.0,
        lambda_layer=1.0,
        lambda_part=0.5,
        lambda_face=0.5,
        lambda_uv=0.25,
        lambda_uv_class=1.0,
        lambda_affine=1.0,
        lambda_surface=1.0,
        uv_size=64,
        foreground_pos_weight_max=20.0,
        use_uv=True,
        affine_translation_limit=0.06,
        affine_log_scale_limit=0.03,
    ):
        super().__init__()
        self.lambda_foreground = lambda_foreground
        self.lambda_layer = lambda_layer
        self.lambda_part = lambda_part
        self.lambda_face = lambda_face
        self.lambda_uv = lambda_uv
        self.lambda_uv_class = lambda_uv_class
        self.lambda_affine = lambda_affine
        self.lambda_surface = lambda_surface
        self.uv_size = uv_size
        self.foreground_pos_weight_max = foreground_pos_weight_max
        self.use_uv = bool(use_uv)
        self.affine_translation_limit = max(float(affine_translation_limit), 1e-6)
        self.affine_log_scale_limit = max(float(affine_log_scale_limit), 1e-6)

    def forward(self, outputs, targets):
        fg_target = targets["foreground"].float()
        pos_count = fg_target.sum().clamp_min(1.0)
        neg_count = (fg_target.numel() - fg_target.sum()).clamp_min(1.0)
        pos_weight = (neg_count / pos_count).clamp(max=self.foreground_pos_weight_max)
        loss_foreground_bce = F.binary_cross_entropy_with_logits(
            outputs["foreground"],
            fg_target,
            pos_weight=pos_weight,
        )
        fg_prob = torch.sigmoid(outputs["foreground"])
        dice_den = fg_prob.sum() + fg_target.sum()
        loss_foreground_dice = 1.0 - (2.0 * (fg_prob * fg_target).sum() + 1.0) / (dice_den + 1.0)
        loss_foreground = loss_foreground_bce + loss_foreground_dice

        loss_layer = F.cross_entropy(outputs["layer"], targets["layer"], ignore_index=IGNORE_INDEX)
        loss_part = F.cross_entropy(outputs["part"], targets["part"], ignore_index=IGNORE_INDEX)
        loss_face = F.cross_entropy(outputs["face"], targets["face"], ignore_index=IGNORE_INDEX)

        fg_mask = targets["layer"] != IGNORE_INDEX
        use_uv = self.use_uv and "uv" in outputs
        if use_uv and fg_mask.any():
            pred_uv_px = outputs["uv"].permute(0, 2, 3, 1)[fg_mask] * (self.uv_size - 1)
            target_uv_px = targets["uv"].permute(0, 2, 3, 1)[fg_mask] * (self.uv_size - 1)
            loss_uv = F.smooth_l1_loss(pred_uv_px, target_uv_px)
            loss_uv_l1_px = (pred_uv_px - target_uv_px).abs().mean()
        else:
            loss_uv = outputs["foreground"].new_tensor(0.0)
            loss_uv_l1_px = outputs["foreground"].new_tensor(0.0)

        if use_uv and "uv_x" in outputs and "uv_y" in outputs and fg_mask.any():
            target_x, target_y = uv_class_targets(targets["uv"], targets["layer"], self.uv_size)
            loss_uv_x = F.cross_entropy(outputs["uv_x"], target_x, ignore_index=IGNORE_INDEX)
            loss_uv_y = F.cross_entropy(outputs["uv_y"], target_y, ignore_index=IGNORE_INDEX)
            loss_uv_class = 0.5 * (loss_uv_x + loss_uv_y)
        else:
            loss_uv_x = outputs["foreground"].new_tensor(0.0)
            loss_uv_y = outputs["foreground"].new_tensor(0.0)
            loss_uv_class = outputs["foreground"].new_tensor(0.0)

        if "affine" in outputs and "affine" in targets:
            affine_error = outputs["affine"] - targets["affine"].to(outputs["affine"].dtype)
            loss_affine_translation = F.smooth_l1_loss(
                affine_error[:, :2] / self.affine_translation_limit,
                torch.zeros_like(affine_error[:, :2]),
            )
            loss_affine_scale = F.smooth_l1_loss(
                affine_error[:, 2] / self.affine_log_scale_limit,
                torch.zeros_like(affine_error[:, 2]),
            )
            loss_affine = 0.5 * (loss_affine_translation + loss_affine_scale)
            H, W = outputs["foreground"].shape[-2:]
            err_affine_translation_px = 0.5 * (
                affine_error[:, 0].abs().mean() * (W / 2.0)
                + affine_error[:, 1].abs().mean() * (H / 2.0)
            )
            err_affine_scale_pct = (
                outputs["affine"][:, 2].exp() - targets["affine"][:, 2].to(outputs["affine"].dtype).exp()
            ).abs().mean() * 100.0
        else:
            loss_affine_translation = outputs["foreground"].new_tensor(0.0)
            loss_affine_scale = outputs["foreground"].new_tensor(0.0)
            loss_affine = outputs["foreground"].new_tensor(0.0)
            err_affine_translation_px = outputs["foreground"].new_tensor(0.0)
            err_affine_scale_pct = outputs["foreground"].new_tensor(0.0)

        if "surface" in outputs and "surface" in targets:
            loss_surface = F.cross_entropy(outputs["surface"], targets["surface"], ignore_index=IGNORE_INDEX)
            acc_surface = _masked_accuracy(outputs["surface"], targets["surface"])
        else:
            loss_surface = outputs["foreground"].new_tensor(0.0)
            acc_surface = outputs["foreground"].new_tensor(0.0)

        loss_routing = loss_surface + loss_uv_class

        loss_total = (
            self.lambda_foreground * loss_foreground
            + self.lambda_layer * loss_layer
            + self.lambda_part * loss_part
            + self.lambda_face * loss_face
            + self.lambda_uv * loss_uv
            + self.lambda_uv_class * loss_uv_class
            + self.lambda_affine * loss_affine
            + self.lambda_surface * loss_surface
        )

        metrics = {
            "loss_total": loss_total,
            "loss_foreground": loss_foreground,
            "loss_foreground_bce": loss_foreground_bce,
            "loss_foreground_dice": loss_foreground_dice,
            "loss_layer": loss_layer,
            "loss_part": loss_part,
            "loss_face": loss_face,
            "loss_uv": loss_uv,
            "loss_uv_l1_px": loss_uv_l1_px,
            "loss_uv_class": loss_uv_class,
            "loss_uv_x": loss_uv_x,
            "loss_uv_y": loss_uv_y,
            "loss_affine": loss_affine,
            "loss_affine_translation": loss_affine_translation,
            "loss_affine_scale": loss_affine_scale,
            "err_affine_translation_px": err_affine_translation_px,
            "err_affine_scale_pct": err_affine_scale_pct,
            "loss_surface": loss_surface,
            "loss_routing": loss_routing,
            "acc_surface": acc_surface,
        }
        metrics.update(classification_metrics(outputs, targets, self.uv_size, use_uv=use_uv))
        return metrics


def uv_class_targets(uv, layer, uv_size):
    valid = layer != IGNORE_INDEX
    xy = (uv * (uv_size - 1)).round().long().clamp(0, uv_size - 1)
    target_x = torch.full_like(layer, IGNORE_INDEX)
    target_y = torch.full_like(layer, IGNORE_INDEX)
    target_x[valid] = xy[:, 0][valid]
    target_y[valid] = xy[:, 1][valid]
    return target_x, target_y


def _masked_accuracy(logits, target):
    mask = target != IGNORE_INDEX
    if not mask.any():
        return logits.new_tensor(0.0)
    pred = logits.argmax(dim=1)
    return (pred[mask] == target[mask]).float().mean()


def classification_metrics(outputs, targets, uv_size, use_uv=True):
    fg_pred = torch.sigmoid(outputs["foreground"]) > 0.5
    fg_target = targets["foreground"] > 0.5
    fg_acc = (fg_pred == fg_target).float().mean()
    fg_pred_flat = fg_pred[:, 0]
    fg_target_flat = fg_target[:, 0]
    tp = (fg_pred_flat & fg_target_flat).sum().float()
    fp = (fg_pred_flat & ~fg_target_flat).sum().float()
    fn = (~fg_pred_flat & fg_target_flat).sum().float()
    metrics = {
        "acc_foreground": fg_acc,
        "precision_foreground": tp / (tp + fp).clamp_min(1.0),
        "recall_foreground": tp / (tp + fn).clamp_min(1.0),
        "iou_foreground": tp / (tp + fp + fn).clamp_min(1.0),
        "acc_layer": _masked_accuracy(outputs["layer"], targets["layer"]),
        "acc_part": _masked_accuracy(outputs["part"], targets["part"]),
        "acc_face": _masked_accuracy(outputs["face"], targets["face"]),
    }
    if use_uv and "uv_x" in outputs and "uv_y" in outputs:
        target_x, target_y = uv_class_targets(targets["uv"], targets["layer"], uv_size)
        valid = target_x != IGNORE_INDEX
        if valid.any():
            pred_x = outputs["uv_x"].argmax(dim=1)
            pred_y = outputs["uv_y"].argmax(dim=1)
            err_x = (pred_x[valid] - target_x[valid]).abs()
            err_y = (pred_y[valid] - target_y[valid]).abs()
            metrics.update(
                {
                    "acc_uv_x": (pred_x[valid] == target_x[valid]).float().mean(),
                    "acc_uv_y": (pred_y[valid] == target_y[valid]).float().mean(),
                    "acc_uv_exact": ((pred_x[valid] == target_x[valid]) & (pred_y[valid] == target_y[valid])).float().mean(),
                    "acc_uv_within1": ((err_x <= 1) & (err_y <= 1)).float().mean(),
                    "err_uv_class_l1_px": 0.5 * (err_x.float().mean() + err_y.float().mean()),
                }
            )
    return metrics
