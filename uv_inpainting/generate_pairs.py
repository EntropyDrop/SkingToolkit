import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TOOLKIT_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from SkingToolkit.uv_inpainting.dataset import IMAGE_EXTENSIONS, load_skin, parse_views, tensor_to_rgba_image  # noqa: E402
from SkingToolkit.renderer import DifferentiableRenderer  # noqa: E402


def render_skin(renderer, skin, view, include_alpha):
    with torch.no_grad():
        rendered = renderer.forward_view(skin.unsqueeze(0), view)[0]
    if not include_alpha:
        rendered = rendered[:3]
    return tensor_to_rgba_image(rendered) if include_alpha else Image.fromarray(
        (rendered.permute(1, 2, 0).cpu().clamp(0.0, 1.0).numpy() * 255).astype("uint8"),
        mode="RGB",
    )


def save_combined(images, path):
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    combined = Image.new(images[0].mode, (width, height))
    offset = 0
    for image in images:
        combined.paste(image, (offset, 0))
        offset += image.width
    combined.save(path)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Generate fixed-view render pairs from GT Minecraft UV skins.")
    parser.add_argument("--data_dir", required=True, help="Folder containing GT 64x64 RGBA skin PNGs.")
    parser.add_argument("--output_dir", required=True, help="Output render folder.")
    parser.add_argument("--mappings_dir", default=None)
    parser.add_argument("--views", default="static_front,static_back")
    parser.add_argument("--unproject_mode", choices=["mode", "mean", "medoid"], default="mode")
    parser.add_argument("--include_alpha", action="store_true")
    parser.add_argument("--combined", action="store_true", help="Also save side-by-side combined images.")
    parser.add_argument("--max_samples", type=int, default=None)
    return parser


def main():
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    views = parse_views(args.views)
    renderer = DifferentiableRenderer(mappings_dir=args.mappings_dir)
    missing_views = [view for view in views if view not in renderer.views]
    if missing_views:
        raise ValueError(f"Unknown renderer views {missing_views}. Available views: {', '.join(renderer.views)}")

    skin_paths = sorted(
        os.path.join(args.data_dir, filename)
        for filename in os.listdir(args.data_dir)
        if filename.lower().endswith(IMAGE_EXTENSIONS) and not filename.startswith("half_")
    )
    if args.max_samples is not None:
        skin_paths = skin_paths[: args.max_samples]

    for view in views:
        (output_dir / view).mkdir(exist_ok=True)
    if args.combined:
        (output_dir / "combined").mkdir(exist_ok=True)

    for index, skin_path in enumerate(skin_paths, start=1):
        stem = Path(skin_path).stem
        skin = load_skin(skin_path)
        images = []
        for view in views:
            image = render_skin(renderer, skin, view, args.include_alpha)
            image.save(output_dir / view / f"{stem}.png")
            images.append(image)
        if args.combined:
            save_combined(images, output_dir / "combined" / f"{stem}.png")
        if index % 100 == 0 or index == len(skin_paths):
            print(f"rendered {index}/{len(skin_paths)}")


if __name__ == "__main__":
    main()
