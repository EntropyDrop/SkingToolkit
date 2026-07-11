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

from SkingToolkit.inverse_uv.dataset import (  # noqa: E402
    finalize_minecraft_alpha,
    parse_views,
    tensor_to_rgba_image,
    unproject_renders_to_uv,
    view_native_size,
)
from SkingToolkit.inverse_uv.model import InverseUVNet, LightInverseUVNet  # noqa: E402
from SkingToolkit.inverse_uv.train import get_device  # noqa: E402
from SkingToolkit.renderer import DifferentiableRenderer  # noqa: E402


def image_to_render_tensor(image, view_size):
    tensor = TF.to_tensor(image.convert("RGBA"))
    if tuple(tensor.shape[-2:]) != tuple(view_size):
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=view_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return tensor.clamp(0.0, 1.0)


def load_conditioning(args, checkpoint_args, input_channels):
    conditioning_mode = checkpoint_args.get("conditioning_mode")
    if conditioning_mode not in (None, "uv_unproject_inpaint"):
        raise ValueError(f"Unsupported checkpoint conditioning_mode={conditioning_mode!r}.")
    if input_channels != 10:
        raise ValueError(
            f"Checkpoint expects {input_channels} input channels. "
            "This inference path requires a uv_unproject_inpaint checkpoint; retrain with the current train.py."
        )

    views = parse_views(checkpoint_args.get("views", "static_front,static_back"))
    mappings_dir = args.mappings_dir or checkpoint_args.get("mappings_dir")
    renderer = DifferentiableRenderer(mappings_dir=mappings_dir)
    missing_views = [view for view in views if view not in renderer.views]
    if missing_views:
        raise ValueError(f"Unknown renderer views {missing_views}. Available views: {', '.join(renderer.views)}")

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
        if len(views) != 2:
            raise ValueError(f"--front/--back only works for 2-view checkpoints, got {len(views)} views.")
        images = [Image.open(args.front), Image.open(args.back)]
    else:
        raise ValueError("Provide --combined, --view_images, or both --front and --back.")

    rendered_views = [
        image_to_render_tensor(image, view_native_size(renderer, view))
        for image, view in zip(images, views)
    ]
    unproject_mode = args.unproject_mode
    if unproject_mode is None:
        unproject_mode = checkpoint_args.get("unproject_mode", "mode")

    conditioning = unproject_renders_to_uv(rendered_views, renderer, views, unproject_mode=unproject_mode)
    if conditioning.shape[0] != input_channels:
        raise ValueError(f"Conditioning has {conditioning.shape[0]} channels, checkpoint expects {input_channels}.")
    return conditioning.unsqueeze(0)


def model_config_from_checkpoint(checkpoint, checkpoint_args, input_channels):
    model_config = checkpoint.get("model_config")
    if model_config is not None:
        model_type = model_config.get("model_type", "full")
        result = {
            "model_type": model_type,
            "input_channels": model_config.get("input_channels", input_channels),
            "base_channels": model_config.get("base_channels", checkpoint_args.get("base_channels", 64)),
            "use_coordconv": model_config.get("use_coordconv", False),
        }
        if model_type == "light":
            result["use_pixelshuffle"] = model_config.get("use_pixelshuffle", False)
        else:
            result.update({
                "use_attention": model_config.get("use_attention", False),
                "attention_heads": model_config.get("attention_heads", 4),
                "use_resnet": model_config.get("use_resnet", False),
                "multi_scale_coord": model_config.get("multi_scale_coord", False),
            })
        return result

    if "coordconv" in checkpoint_args or "bottleneck_attention" in checkpoint_args:
        return {
            "model_type": checkpoint_args.get("model", "full"),
            "input_channels": input_channels,
            "base_channels": checkpoint_args.get("base_channels", 64),
            "use_coordconv": checkpoint_args.get("coordconv", False),
            "use_attention": checkpoint_args.get("bottleneck_attention", False),
            "attention_heads": checkpoint_args.get("attention_heads", 4),
            "use_resnet": checkpoint_args.get("resnet", False),
            "multi_scale_coord": checkpoint_args.get("multi_scale_coord", False),
        }

    return {
        "model_type": "full",
        "input_channels": input_channels,
        "base_channels": checkpoint_args.get("base_channels", 64),
        "use_coordconv": False,
        "use_attention": False,
        "attention_heads": 4,
        "use_resnet": False,
        "multi_scale_coord": False,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Infer Minecraft UV from fixed render views.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt/latest.pt.")
    parser.add_argument("--output", required=True, help="Output RGBA UV PNG path.")
    parser.add_argument("--front", default=None, help="Front render image.")
    parser.add_argument("--back", default=None, help="Back render image.")
    parser.add_argument("--combined", default=None, help="Combined side-by-side image matching checkpoint view order.")
    parser.add_argument("--view_images", nargs="*", default=None, help="Images matching checkpoint view order.")
    parser.add_argument("--alpha_threshold", type=float, default=0.5, help="Threshold used to binarize predicted alpha.")
    parser.add_argument(
        "--no_enforce_base_alpha",
        action="store_true",
        help="Do not force the Minecraft base layer alpha to opaque in the saved PNG.",
    )
    parser.add_argument("--mappings_dir", default=None, help="Override renderer mappings directory from checkpoint.")
    parser.add_argument(
        "--unproject_mode",
        choices=["mode", "mean", "medoid"],
        default=None,
        help="Method to aggregate render pixels into UV texels. Defaults to checkpoint's unproject_mode (or 'mode').",
    )
    parser.add_argument(
        "--render_size",
        type=int,
        default=None,
        help="Deprecated compatibility option; inference uses each view's native mapping size.",
    )
    parser.add_argument("--device", default="auto")
    return parser


def main():
    args = build_arg_parser().parse_args()
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_args = checkpoint.get("args", {})
    input_channels = checkpoint.get("input_channels", checkpoint_args.get("input_channels", 6))
    model_config = model_config_from_checkpoint(checkpoint, checkpoint_args, input_channels)

    model = InverseUVNet(
        input_channels=model_config["input_channels"],
        base_channels=model_config["base_channels"],
        preserve_known=checkpoint_args.get("preserve_known", True),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    conditioning = load_conditioning(args, checkpoint_args, input_channels).to(device)
    with torch.no_grad():
        pred_uv = finalize_minecraft_alpha(
            model(conditioning)[0],
            alpha_threshold=args.alpha_threshold,
            enforce_base_alpha=not args.no_enforce_base_alpha,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_rgba_image(pred_uv).save(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
