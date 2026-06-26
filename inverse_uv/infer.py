import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TOOLKIT_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from SkingToolkit.inverse_uv.dataset import parse_views, tensor_to_rgba_image  # noqa: E402
from SkingToolkit.inverse_uv.model import InverseUVNet  # noqa: E402
from SkingToolkit.inverse_uv.train import get_device  # noqa: E402


def image_to_condition_tensor(image, render_size, include_alpha):
    mode = "RGBA" if include_alpha else "RGB"
    tensor = TF.to_tensor(image.convert(mode))
    tensor = F.interpolate(
        tensor.unsqueeze(0),
        size=(render_size, render_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return tensor.clamp(0.0, 1.0)


def load_conditioning(args, checkpoint_args, input_channels):
    views = parse_views(checkpoint_args.get("views", "static_front,static_back"))
    include_alpha = checkpoint_args.get("include_alpha", False)
    render_size = args.render_size or checkpoint_args.get("render_size", 256)

    images = []
    if args.combined:
        combined = Image.open(args.combined)
        width, height = combined.size
        if width % len(views) != 0:
            raise ValueError(f"Combined image width {width} is not divisible by {len(views)} views.")
        view_width = width // len(views)
        images = [combined.crop((i * view_width, 0, (i + 1) * view_width, height)) for i in range(len(views))]
    elif args.view_images:
        if len(args.view_images) != len(views):
            raise ValueError(f"Expected {len(views)} --view_images, got {len(args.view_images)}.")
        images = [Image.open(path) for path in args.view_images]
    elif args.front and args.back:
        images = [Image.open(args.front), Image.open(args.back)]
    else:
        raise ValueError("Provide --combined, --view_images, or both --front and --back.")

    channels = [image_to_condition_tensor(image, render_size, include_alpha) for image in images]
    conditioning = torch.cat(channels, dim=0)
    if conditioning.shape[0] != input_channels:
        raise ValueError(f"Conditioning has {conditioning.shape[0]} channels, checkpoint expects {input_channels}.")
    return conditioning.unsqueeze(0)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Infer Minecraft UV from fixed render views.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt/latest.pt.")
    parser.add_argument("--output", required=True, help="Output RGBA UV PNG path.")
    parser.add_argument("--front", default=None, help="Front render image.")
    parser.add_argument("--back", default=None, help="Back render image.")
    parser.add_argument("--combined", default=None, help="Combined side-by-side front/back image.")
    parser.add_argument("--view_images", nargs="*", default=None, help="Images matching checkpoint view order.")
    parser.add_argument("--render_size", type=int, default=None, help="Override input render resize.")
    parser.add_argument("--device", default="auto")
    return parser


def main():
    args = build_arg_parser().parse_args()
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_args = checkpoint.get("args", {})
    input_channels = checkpoint.get("input_channels", checkpoint_args.get("input_channels", 6))
    base_channels = checkpoint_args.get("base_channels", 64)

    model = InverseUVNet(input_channels=input_channels, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    conditioning = load_conditioning(args, checkpoint_args, input_channels).to(device)
    with torch.no_grad():
        pred_uv = model(conditioning)[0]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_rgba_image(pred_uv).save(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
