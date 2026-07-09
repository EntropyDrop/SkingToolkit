import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.utils import save_image

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TOOLKIT_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from SkingToolkit.dense_uv_parser.model import DenseUVParserNet  # noqa: E402
from SkingToolkit.dense_uv_parser.utils import parse_views, splat_predictions_to_uv_conditioning  # noqa: E402
from SkingToolkit.inverse_uv.dataset import finalize_minecraft_alpha, tensor_to_rgba_image, view_native_size  # noqa: E402
from SkingToolkit.inverse_uv.model import InverseUVNet  # noqa: E402
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


def load_parser(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_args = checkpoint.get("args", {})
    model_config = checkpoint.get("model_config", {})
    model = DenseUVParserNet(
        base_channels=model_config.get("base_channels", checkpoint_args.get("base_channels", 32)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint_args


def load_inpaint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_args = checkpoint.get("args", {})
    input_channels = checkpoint.get("input_channels", checkpoint_args.get("input_channels", 10))
    model = InverseUVNet(
        input_channels=input_channels,
        base_channels=checkpoint_args.get("base_channels", 64),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def load_view_images(args, views, renderer):
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

    tensors = [
        image_to_render_tensor(image, view_native_size(renderer, view))
        for image, view in zip(images, views)
    ]
    return torch.stack(tensors, dim=0)


def save_conditioning_preview(conditioning, output_path):
    inner_rgb = conditioning[:, 0:3]
    outer_rgb = conditioning[:, 5:8]
    preview = torch.cat([inner_rgb, outer_rgb], dim=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(preview.clamp(0.0, 1.0), output_path, nrow=conditioning.shape[0])


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Infer UV conditioning with a dense UV parser.")
    parser.add_argument("--parser_checkpoint", required=True)
    parser.add_argument("--inpaint_checkpoint", default=None, help="Optional inverse_uv checkpoint used to inpaint final skin.")
    parser.add_argument("--output", default=None, help="Final RGBA UV PNG path; requires --inpaint_checkpoint.")
    parser.add_argument("--conditioning_output", default=None, help="Optional preview image for parser-splatted conditioning.")
    parser.add_argument("--front", default=None)
    parser.add_argument("--back", default=None)
    parser.add_argument("--combined", default=None)
    parser.add_argument("--view_images", nargs="*", default=None)
    parser.add_argument("--mappings_dir", default=None)
    parser.add_argument("--fg_threshold", type=float, default=0.5)
    parser.add_argument("--alpha_threshold", type=float, default=0.5)
    parser.add_argument("--no_enforce_base_alpha", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.output and not args.inpaint_checkpoint:
        raise ValueError("--output requires --inpaint_checkpoint.")
    if not args.output and not args.conditioning_output:
        raise ValueError("Provide --output and/or --conditioning_output.")

    device = get_device(args.device)
    parser_model, parser_args = load_parser(args.parser_checkpoint, device)
    views = parse_views(parser_args.get("views", "walk_front_both_layer_ortho,walk_back_both_layer_ortho"))
    mappings_dir = args.mappings_dir or parser_args.get("mappings_dir")
    renderer = DifferentiableRenderer(mappings_dir=mappings_dir)
    missing_views = [view for view in views if view not in renderer.views]
    if missing_views:
        raise ValueError(f"Unknown renderer views {missing_views}. Available views: {', '.join(renderer.views)}")

    rendered = load_view_images(args, views, renderer).to(device)
    with torch.no_grad():
        outputs = parser_model(rendered)
        conditioning = splat_predictions_to_uv_conditioning(
            rendered,
            outputs,
            group_size=len(views),
            fg_threshold=args.fg_threshold,
        )

    if args.conditioning_output:
        save_conditioning_preview(conditioning.detach().cpu(), Path(args.conditioning_output))

    if args.output:
        inpaint_model = load_inpaint(args.inpaint_checkpoint, device)
        with torch.no_grad():
            pred_uv = finalize_minecraft_alpha(
                inpaint_model(conditioning)[0],
                alpha_threshold=args.alpha_threshold,
                enforce_base_alpha=not args.no_enforce_base_alpha,
            )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tensor_to_rgba_image(pred_uv.detach().cpu()).save(output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()

