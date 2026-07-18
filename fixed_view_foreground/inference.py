import re
from pathlib import Path

import torch
from torchvision.utils import save_image

from SkingToolkit.fixed_view_foreground.model import FixedViewForegroundNet


RUN_PATTERN = re.compile(r"^fixed_view_foreground_v(\d+)$")


def find_latest_checkpoint(runs_dir=None, checkpoint_name="best.pt"):
    if runs_dir is None:
        runs_dir = Path(__file__).resolve().parent / "runs"
    runs_dir = Path(runs_dir)
    candidates = []
    if runs_dir.is_dir():
        for run_dir in runs_dir.iterdir():
            match = RUN_PATTERN.match(run_dir.name)
            checkpoint = run_dir / checkpoint_name
            if match and checkpoint.is_file():
                candidates.append((int(match.group(1)), checkpoint))
    return max(candidates, default=(-1, None), key=lambda item: item[0])[1]


def load_foreground_model(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("model_config", {})
    model = FixedViewForegroundNet(
        input_channels=config.get("input_channels", 3),
        base_channels=config.get("base_channels", 24),
        view_classes=config.get("view_classes", 2),
        coordinate_channels=config.get("coordinate_channels", True),
        dropout=config.get("dropout", 0.05),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint.get("args", {}), checkpoint


@torch.no_grad()
def predict_foreground(model, rendered, view_ids):
    probability = torch.sigmoid(model(rendered[:, :3], view_ids=view_ids).float())
    return probability.clamp(0.0, 1.0)


def save_foreground_outputs(
    rendered,
    probability,
    threshold,
    view_count,
    probability_output=None,
    mask_output=None,
    cutout_output=None,
    bg_color=(128, 128, 128),
):
    mask = probability >= threshold
    bg = rendered.new_tensor(bg_color).view(1, 3, 1, 1) / 255.0
    cutout = torch.where(mask.expand(-1, 3, -1, -1), rendered[:, :3], bg)
    for tensor, path in (
        (probability, probability_output),
        (mask.to(dtype=rendered.dtype), mask_output),
        (cutout, cutout_output),
    ):
        if path is None:
            continue
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_image(tensor.detach().cpu(), path, nrow=view_count)
    return mask[:, 0]
