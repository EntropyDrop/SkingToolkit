import torch
import torch.nn as nn
import torch.nn.functional as F

from SkingToolkit.dense_uv_parser.utils import IGNORE_INDEX, combine_layer_face


def _balanced_cross_entropy(
    logits,
    target,
    max_weight=4.0,
    min_weight=0.0,
    class_weight_caps=None,
):
    """Give small projected faces useful gradient without letting them dominate."""
    if min_weight < 0 or max_weight < min_weight:
        raise ValueError("Class weights require 0 <= min_weight <= max_weight.")
    valid = target != IGNORE_INDEX
    if not valid.any():
        return logits.new_tensor(0.0)
    counts = torch.bincount(target[valid], minlength=logits.shape[1]).float()
    active = counts > 0
    weights = torch.where(active, counts.clamp_min(1.0).rsqrt(), torch.zeros_like(counts))
    weights[active] /= weights[active].mean().clamp_min(1e-6)
    weights[active] = weights[active].clamp(min=min_weight, max=max_weight)
    if class_weight_caps is not None:
        caps = torch.as_tensor(
            class_weight_caps,
            device=weights.device,
            dtype=weights.dtype,
        )
        if caps.numel() != logits.shape[1]:
            raise ValueError(
                f"Expected {logits.shape[1]} class-weight caps, got {caps.numel()}."
            )
        if (caps < 0).any():
            raise ValueError("Class-weight caps must be non-negative.")
        weights = torch.minimum(weights, caps)
    return F.cross_entropy(logits.float(), target, weight=weights, ignore_index=IGNORE_INDEX)


def outer_false_positive_loss(logits, target, outer_index=1, gamma=2.0):
    """Focal negative loss for non-outer pixels assigned outer probability.

    Ordinary class-balanced cross entropy can underweight abundant inner pixels
    while increasing the rare outer target weight.  This term restores the
    asymmetric cost that matters for UV reconstruction: confidently inventing
    an outer texel is worse than leaving an uncertain outer texel unknown.
    """
    if gamma < 0:
        raise ValueError("gamma must be non-negative.")
    if not 0 <= outer_index < logits.shape[1]:
        raise ValueError(f"outer_index={outer_index} is invalid for {logits.shape[1]} classes.")
    negative = (target != IGNORE_INDEX) & (target != outer_index)
    if not negative.any():
        return logits.new_zeros((), dtype=torch.float32)
    outer_probability = torch.softmax(logits.float(), dim=1)[:, outer_index]
    outer_probability = outer_probability.clamp(1e-6, 1.0 - 1e-6)
    focal_negative = outer_probability.pow(gamma) * (-torch.log1p(-outer_probability))
    return focal_negative[negative].mean()


def outer_false_negative_loss(logits, target, outer_index=1, gamma=2.0):
    """Focal positive loss for visible outer pixels assigned too little outer probability."""
    if gamma < 0:
        raise ValueError("gamma must be non-negative.")
    if not 0 <= outer_index < logits.shape[1]:
        raise ValueError(f"outer_index={outer_index} is invalid for {logits.shape[1]} classes.")
    positive = target == outer_index
    if not positive.any():
        return logits.new_zeros((), dtype=torch.float32)
    outer_probability = torch.softmax(logits.float(), dim=1)[:, outer_index]
    outer_probability = outer_probability.clamp(1e-6, 1.0 - 1e-6)
    focal_positive = (1.0 - outer_probability).pow(gamma) * (-outer_probability.log())
    return focal_positive[positive].mean()


class DenseUVParserLoss(nn.Module):
    def __init__(
        self,
        lambda_foreground=1.0,
        lambda_layer=1.0,
        lambda_part=0.5,
        lambda_face=0.5,
        lambda_layer_face=1.0,
        lambda_uv=0.25,
        lambda_uv_class=1.0,
        lambda_affine=1.0,
        lambda_surface=1.0,
        lambda_outer_false_positive=0.75,
        lambda_outer_false_negative=0.75,
        outer_false_positive_gamma=2.0,
        outer_false_negative_gamma=2.0,
        route_class_weight_floor=0.75,
        route_outer_class_weight_cap=1.0,
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
        self.lambda_layer_face = lambda_layer_face
        self.lambda_uv = lambda_uv
        self.lambda_uv_class = lambda_uv_class
        self.lambda_affine = lambda_affine
        self.lambda_surface = lambda_surface
        self.lambda_outer_false_positive = lambda_outer_false_positive
        self.lambda_outer_false_negative = lambda_outer_false_negative
        self.outer_false_positive_gamma = outer_false_positive_gamma
        self.outer_false_negative_gamma = outer_false_negative_gamma
        self.route_class_weight_floor = route_class_weight_floor
        self.route_outer_class_weight_cap = route_outer_class_weight_cap
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

        zero = outputs["foreground"].new_tensor(0.0)
        geometry_route_roles = outputs["layer"].shape[1] == 3 and "route_role" in targets
        layer_target = targets["route_role"] if geometry_route_roles else targets["layer"]
        loss_layer = (
            _balanced_cross_entropy(
                outputs["layer"],
                layer_target,
                min_weight=self.route_class_weight_floor,
                class_weight_caps=(
                    float("inf"),
                    self.route_outer_class_weight_cap,
                    float("inf"),
                ),
            )
            if geometry_route_roles
            else F.cross_entropy(outputs["layer"], layer_target, ignore_index=IGNORE_INDEX)
        )
        loss_outer_false_positive = (
            outer_false_positive_loss(
                outputs["layer"],
                layer_target,
                gamma=self.outer_false_positive_gamma,
            )
            if geometry_route_roles
            else zero
        )
        weighted_outer_false_positive = (
            self.lambda_outer_false_positive * loss_outer_false_positive
        )
        loss_outer_false_negative = (
            outer_false_negative_loss(
                outputs["layer"],
                layer_target,
                gamma=self.outer_false_negative_gamma,
            )
            if geometry_route_roles
            else zero
        )
        weighted_outer_false_negative = (
            self.lambda_outer_false_negative * loss_outer_false_negative
        )
        loss_part = (
            F.cross_entropy(outputs["part"], targets["part"], ignore_index=IGNORE_INDEX)
            if "part" in outputs
            else zero
        )
        loss_face = (
            F.cross_entropy(outputs["face"], targets["face"], ignore_index=IGNORE_INDEX)
            if "face" in outputs
            else zero
        )
        layer_face_target = combine_layer_face(targets["layer"], targets["face"])
        if "layer_face" in outputs:
            loss_layer_face = _balanced_cross_entropy(outputs["layer_face"], layer_face_target)
        else:
            loss_layer_face = zero

        fg_mask = targets["layer"] != IGNORE_INDEX
        use_uv = self.use_uv and "uv" in outputs
        if use_uv and fg_mask.any():
            pred_uv_px = outputs["uv"].permute(0, 2, 3, 1)[fg_mask] * (self.uv_size - 1)
            target_uv_px = targets["uv"].permute(0, 2, 3, 1)[fg_mask] * (self.uv_size - 1)
            loss_uv = F.smooth_l1_loss(pred_uv_px, target_uv_px)
            loss_uv_l1_px = (pred_uv_px - target_uv_px).abs().mean()
        else:
            loss_uv = zero
            loss_uv_l1_px = zero

        if use_uv and "uv_x" in outputs and "uv_y" in outputs and fg_mask.any():
            target_x, target_y = uv_class_targets(targets["uv"], targets["layer"], self.uv_size)
            loss_uv_x = F.cross_entropy(outputs["uv_x"], target_x, ignore_index=IGNORE_INDEX)
            loss_uv_y = F.cross_entropy(outputs["uv_y"], target_y, ignore_index=IGNORE_INDEX)
            loss_uv_class = 0.5 * (loss_uv_x + loss_uv_y)
        else:
            loss_uv_x = zero
            loss_uv_y = zero
            loss_uv_class = zero

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
            loss_affine_translation = zero
            loss_affine_scale = zero
            loss_affine = zero
            err_affine_translation_px = zero
            err_affine_scale_pct = zero

        if "surface" in outputs and "surface" in targets:
            loss_surface = _balanced_cross_entropy(outputs["surface"], targets["surface"])
            acc_surface = _masked_accuracy(outputs["surface"], targets["surface"])
        else:
            loss_surface = zero
            acc_surface = zero

        geometry_route_roles = outputs["layer"].shape[1] == 3 and "route_role" in targets
        loss_routing = (
            loss_layer
            + loss_affine
            + loss_surface
            + weighted_outer_false_positive
            + weighted_outer_false_negative
            if geometry_route_roles
            else loss_surface + loss_uv_class + loss_layer_face
        )
        loss_geometry = loss_foreground + loss_layer + loss_affine + (
            loss_surface if geometry_route_roles else zero
        ) + weighted_outer_false_positive + weighted_outer_false_negative

        loss_total = (
            self.lambda_foreground * loss_foreground
            + self.lambda_layer * loss_layer
            + self.lambda_part * loss_part
            + self.lambda_face * loss_face
            + self.lambda_layer_face * loss_layer_face
            + self.lambda_uv * loss_uv
            + self.lambda_uv_class * loss_uv_class
            + self.lambda_affine * loss_affine
            + self.lambda_surface * loss_surface
            + weighted_outer_false_positive
            + weighted_outer_false_negative
        )

        metrics = {
            "loss_total": loss_total,
            "loss_foreground": loss_foreground,
            "loss_foreground_bce": loss_foreground_bce,
            "loss_foreground_dice": loss_foreground_dice,
            "loss_layer": loss_layer,
            "loss_part": loss_part,
            "loss_face": loss_face,
            "loss_layer_face": loss_layer_face,
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
            "loss_outer_false_positive": loss_outer_false_positive,
            "loss_outer_false_positive_weighted": weighted_outer_false_positive,
            "loss_outer_false_negative": loss_outer_false_negative,
            "loss_outer_false_negative_weighted": weighted_outer_false_negative,
            "loss_routing": loss_routing,
            "loss_geometry": loss_geometry,
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
    geometry_route_roles = outputs["layer"].shape[1] == 3 and "route_role" in targets
    layer_target = targets["route_role"] if geometry_route_roles else targets["layer"]
    metrics = {
        "acc_foreground": fg_acc,
        "precision_foreground": tp / (tp + fp).clamp_min(1.0),
        "recall_foreground": tp / (tp + fn).clamp_min(1.0),
        "iou_foreground": tp / (tp + fp + fn).clamp_min(1.0),
        "acc_layer": _masked_accuracy(outputs["layer"], layer_target),
    }
    if geometry_route_roles:
        metrics["acc_route_role"] = metrics["acc_layer"]
        valid_role = layer_target != IGNORE_INDEX
        pred_role = outputs["layer"].argmax(dim=1)
        for name, role in (("outer", 1), ("secondary", 2)):
            target_role = layer_target == role
            predicted_role = pred_role == role
            role_tp = (predicted_role & target_role & valid_role).sum().float()
            role_fp = (predicted_role & ~target_role & valid_role).sum().float()
            role_fn = (~predicted_role & target_role & valid_role).sum().float()
            metrics[f"precision_{name}"] = role_tp / (role_tp + role_fp).clamp_min(1.0)
            metrics[f"recall_{name}"] = role_tp / (role_tp + role_fn).clamp_min(1.0)
            metrics[f"iou_{name}"] = role_tp / (role_tp + role_fp + role_fn).clamp_min(1.0)
            if name == "outer":
                metrics["count_outer_tp"] = role_tp
                metrics["count_outer_fp"] = role_fp
                metrics["count_outer_fn"] = role_fn
        valid_layer = torch.zeros_like(valid_role)
    else:
        valid_layer = targets["layer"] != IGNORE_INDEX
    if valid_layer.any():
        pred_outer = outputs["layer"].argmax(dim=1) == 1
        target_outer = targets["layer"] == 1
        outer_tp = (pred_outer & target_outer & valid_layer).sum().float()
        outer_fp = (pred_outer & ~target_outer & valid_layer).sum().float()
        outer_fn = (~pred_outer & target_outer & valid_layer).sum().float()
        metrics.update(
            {
                "precision_outer": outer_tp / (outer_tp + outer_fp).clamp_min(1.0),
                "recall_outer": outer_tp / (outer_tp + outer_fn).clamp_min(1.0),
                "iou_outer": outer_tp / (outer_tp + outer_fp + outer_fn).clamp_min(1.0),
                "count_outer_tp": outer_tp,
                "count_outer_fp": outer_fp,
                "count_outer_fn": outer_fn,
            }
        )
    if "part" in outputs:
        metrics["acc_part"] = _masked_accuracy(outputs["part"], targets["part"])
    if "face" in outputs:
        metrics["acc_face"] = _masked_accuracy(outputs["face"], targets["face"])
    if "layer_face" in outputs:
        metrics["acc_layer_face"] = _masked_accuracy(
            outputs["layer_face"],
            combine_layer_face(targets["layer"], targets["face"]),
        )
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
