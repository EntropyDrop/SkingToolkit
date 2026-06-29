import argparse
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TOOLKIT_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from SkingToolkit.foreground_alpha.dataset import parse_color  # noqa: E402
from SkingToolkit.foreground_alpha.model import ForegroundAlphaNet  # noqa: E402
from SkingToolkit.foreground_alpha.train import get_device  # noqa: E402


def image_to_tensor(path):
    image = Image.open(path).convert("RGB")
    return TF.to_tensor(image).clamp(0.0, 1.0)


def tensor_to_rgba_image(rgb, alpha):
    rgba = torch.cat([rgb, alpha], dim=0).clamp(0.0, 1.0)
    return TF.to_pil_image(rgba)


def uncompose_background(rgb, alpha, bg_color, min_alpha):
    bg = torch.tensor(bg_color, dtype=rgb.dtype, device=rgb.device).view(3, 1, 1) / 255.0
    fg = (rgb - bg * (1.0 - alpha)) / alpha.clamp_min(min_alpha)
    return torch.where(alpha > min_alpha, fg, rgb).clamp(0.0, 1.0)


def output_path_for(input_path, args):
    input_path = Path(input_path)
    if args.output:
        return Path(args.output)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{input_path.stem}_rgba.png"


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Extract foreground alpha from RGB render PNGs.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", default=None, help="Single input RGB/RGBA image.")
    parser.add_argument("--inputs", nargs="*", default=None, help="Multiple input RGB/RGBA images.")
    parser.add_argument("--output", default=None, help="Output RGBA PNG for --input.")
    parser.add_argument("--output_dir", default="foreground_alpha_outputs", help="Output folder for --inputs.")
    parser.add_argument("--bg_color", default="0,0,0", help="Known input background for --uncompose, as r,g,b.")
    parser.add_argument("--uncompose", action="store_true", help="Recover foreground RGB from a known solid background.")
    parser.add_argument("--min_alpha", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=None, help="Optional hard alpha threshold.")
    parser.add_argument("--device", default="auto")
    return parser


def main():
    args = build_arg_parser().parse_args()
    input_paths = []
    if args.input:
        input_paths.append(args.input)
    if args.inputs:
        input_paths.extend(args.inputs)
    if not input_paths:
        raise ValueError("Provide --input or --inputs.")
    if args.output and len(input_paths) != 1:
        raise ValueError("--output can only be used with a single --input.")

    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_args = checkpoint.get("args", {})
    base_channels = checkpoint_args.get("base_channels", 32)
    model = ForegroundAlphaNet(base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    bg_color = parse_color(args.bg_color)
    for input_path in input_paths:
        rgb = image_to_tensor(input_path).to(device)
        with torch.no_grad():
            alpha = model(rgb.unsqueeze(0))[0].clamp(0.0, 1.0)
        if args.threshold is not None:
            alpha = (alpha >= args.threshold).to(dtype=alpha.dtype)
        out_rgb = uncompose_background(rgb, alpha, bg_color, args.min_alpha) if args.uncompose else rgb
        output_path = output_path_for(input_path, args)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tensor_to_rgba_image(out_rgb.cpu(), alpha.cpu()).save(output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
