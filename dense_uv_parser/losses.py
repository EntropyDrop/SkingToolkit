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
        lambda_uv=5.0,
    ):
        super().__init__()
        self.lambda_foreground = lambda_foreground
        self.lambda_layer = lambda_layer
        self.lambda_part = lambda_part
        self.lambda_face = lambda_face
        self.lambda_uv = lambda_uv

    def forward(self, outputs, targets):
        fg_target = targets["foreground"].float()
        loss_foreground = F.binary_cross_entropy_with_logits(outputs["foreground"], fg_target)

        loss_layer = F.cross_entropy(outputs["layer"], targets["layer"], ignore_index=IGNORE_INDEX)
        loss_part = F.cross_entropy(outputs["part"], targets["part"], ignore_index=IGNORE_INDEX)
        loss_face = F.cross_entropy(outputs["face"], targets["face"], ignore_index=IGNORE_INDEX)

        fg_mask = (targets["layer"] != IGNORE_INDEX).unsqueeze(1)
        if fg_mask.any():
            loss_uv = F.smooth_l1_loss(outputs["uv"][fg_mask.expand_as(outputs["uv"])], targets["uv"][fg_mask.expand_as(targets["uv"])])
        else:
            loss_uv = outputs["uv"].new_tensor(0.0)

        loss_total = (
            self.lambda_foreground * loss_foreground
            + self.lambda_layer * loss_layer
            + self.lambda_part * loss_part
            + self.lambda_face * loss_face
            + self.lambda_uv * loss_uv
        )

        metrics = {
            "loss_total": loss_total,
            "loss_foreground": loss_foreground,
            "loss_layer": loss_layer,
            "loss_part": loss_part,
            "loss_face": loss_face,
            "loss_uv": loss_uv,
        }
        metrics.update(classification_metrics(outputs, targets))
        return metrics


def _masked_accuracy(logits, target):
    mask = target != IGNORE_INDEX
    if not mask.any():
        return logits.new_tensor(0.0)
    pred = logits.argmax(dim=1)
    return (pred[mask] == target[mask]).float().mean()


def classification_metrics(outputs, targets):
    fg_pred = torch.sigmoid(outputs["foreground"]) > 0.5
    fg_target = targets["foreground"] > 0.5
    fg_acc = (fg_pred == fg_target).float().mean()
    return {
        "acc_foreground": fg_acc,
        "acc_layer": _masked_accuracy(outputs["layer"], targets["layer"]),
        "acc_part": _masked_accuracy(outputs["part"], targets["part"]),
        "acc_face": _masked_accuracy(outputs["face"], targets["face"]),
    }

