import argparse
import sys
from pathlib import Path
from PIL import Image

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TOOLKIT_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from SkingToolkit.foreground_alpha_simple.floodfill import (  # noqa: E402
    remove_background_simple,
)


def parse_seed(seed_str):
    parts = [int(p.strip()) for p in seed_str.split(",")]
    if len(parts) != 2:
        raise ValueError("Seed must be in 'x,y' format, e.g., '0,0'")
    return (parts[0], parts[1])


def is_merged_image(pil_img, split_mode="auto"):
    if split_mode == "always":
        return True
    if split_mode == "never":
        return False
    w, h = pil_img.size
    return (w / h) >= 0.75


def output_path_for(input_path, args):
    input_path = Path(input_path)
    if args.output:
        return Path(args.output)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{input_path.stem}_rgba.png"


def process_single_image(pil_img, args, seed=(0, 0), extra_seeds=None):
    fixed_range = not args.floating_range
    return remove_background_simple(
        pil_img,
        seed=seed,
        tolerance=args.tolerance,
        color_space=args.color_space,
        fixed_range=fixed_range,
        extra_seeds=extra_seeds,
        fill_holes=args.fill_holes,
        opening_size=args.opening_size,
        uncompose=args.uncompose,
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Simple seed flood-fill background removal tool (magic wand at 0,0)."
    )
    parser.add_argument("--input", default=None, help="Single input RGB/RGBA image file.")
    parser.add_argument("--inputs", nargs="*", default=None, help="Multiple input RGB/RGBA image files.")
    parser.add_argument("--output", default=None, help="Output RGBA PNG path (for single --input).")
    parser.add_argument(
        "--output_dir", default="foreground_alpha_simple_outputs", help="Output folder for batch inputs."
    )
    parser.add_argument("--output_front", default=None, help="Explicit output path for split front view.")
    parser.add_argument("--output_back", default=None, help="Explicit output path for split back view.")
    parser.add_argument("--seed", default="0,0", help="Seed pixel coordinate (x,y). Default is '0,0'.")
    parser.add_argument(
        "--tolerance", type=float, default=15.0, help="Color tolerance for magic wand flood fill (0-255). Default: 15.0"
    )
    parser.add_argument(
        "--color_space", choices=["RGB", "LAB"], default="RGB", help="Color space for matching. Default: RGB"
    )
    parser.add_argument(
        "--floating_range",
        action="store_true",
        help="Use floating/gradient range (compare to adjacent pixel) instead of fixed range (compare to seed color).",
    )
    parser.add_argument(
        "--uncompose",
        action="store_true",
        help="Recover uncomposed foreground RGB from background color blending on anti-aliased edges.",
    )
    parser.add_argument(
        "--fill_holes",
        action="store_true",
        help="Fill interior transparent holes inside the character body mask.",
    )
    parser.add_argument(
        "--opening_size",
        type=int,
        default=0,
        help="Kernel size for morphological opening to remove thin edge noise (0 to disable).",
    )
    parser.add_argument(
        "--split_merged",
        choices=["auto", "always", "never"],
        default="auto",
        help="Automatically split merged front/back image (left half front, right half back).",
    )
    parser.add_argument(
        "--save_split",
        action="store_true",
        help="When processing merged images, save separated front and back RGBA images.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    input_paths = []
    if args.input:
        input_paths.append(args.input)
    if args.inputs:
        input_paths.extend(args.inputs)
    if not input_paths:
        raise ValueError("Please provide --input or --inputs.")
    if (args.output or args.output_front or args.output_back) and len(input_paths) != 1:
        raise ValueError("--output, --output_front, and --output_back can only be used with a single input image.")

    base_seed = parse_seed(args.seed)

    for input_path in input_paths:
        input_path = Path(input_path)
        if not input_path.exists():
            print(f"Skipping non-existent file: {input_path}")
            continue

        pil_img = Image.open(input_path).convert("RGB")

        if is_merged_image(pil_img, args.split_merged):
            w, h = pil_img.size
            split_w = w // 2
            front_img = pil_img.crop((0, 0, split_w, h))
            back_img = pil_img.crop((split_w, 0, w, h))

            front_rgba = process_single_image(front_img, args, seed=base_seed)
            # For the right half (back view), seed point is relative to left top of right image (base_seed)
            back_rgba = process_single_image(back_img, args, seed=base_seed)

            # Re-combine into merged RGBA output
            merged_rgba = Image.new("RGBA", (w, h))
            merged_rgba.paste(front_rgba, (0, 0))
            merged_rgba.paste(back_rgba, (split_w, 0))

            output_path = output_path_for(input_path, args)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            merged_rgba.save(output_path)
            print(f"Saved merged output: {output_path}")

            if args.output_front:
                front_out_path = Path(args.output_front)
            elif args.save_split:
                output_dir = output_path.parent if args.output else Path(args.output_dir)
                front_out_path = output_dir / f"{input_path.stem}_front_rgba.png"
            else:
                front_out_path = None

            if args.output_back:
                back_out_path = Path(args.output_back)
            elif args.save_split:
                output_dir = output_path.parent if args.output else Path(args.output_dir)
                back_out_path = output_dir / f"{input_path.stem}_back_rgba.png"
            else:
                back_out_path = None

            if front_out_path:
                front_out_path.parent.mkdir(parents=True, exist_ok=True)
                front_rgba.save(front_out_path)
                print(f"Saved front split output: {front_out_path}")

            if back_out_path:
                back_out_path.parent.mkdir(parents=True, exist_ok=True)
                back_rgba.save(back_out_path)
                print(f"Saved back split output: {back_out_path}")
        else:
            out_rgba = process_single_image(pil_img, args, seed=base_seed)
            output_path = output_path_for(input_path, args)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            out_rgba.save(output_path)
            print(f"Saved RGBA output: {output_path}")


if __name__ == "__main__":
    main()
