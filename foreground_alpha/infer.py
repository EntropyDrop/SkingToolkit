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


def image_to_tensor(path_or_image):
    if isinstance(path_or_image, (str, Path)):
        image = Image.open(path_or_image).convert("RGB")
    else:
        image = path_or_image.convert("RGB")
    return TF.to_tensor(image).clamp(0.0, 1.0)


def tensor_to_rgba_image(rgb, alpha):
    rgba = torch.cat([rgb, alpha], dim=0).clamp(0.0, 1.0)
    return TF.to_pil_image(rgba)


def clean_alpha_mask(alpha, threshold=0.5, fill_holes=True, clean_noise=True, min_component_size=100):
    # alpha shape: [1, H, W]
    alpha_np = alpha.squeeze(0).cpu().numpy()
    binary_mask = alpha_np > threshold
    try:
        import numpy as np
        from scipy.ndimage import binary_fill_holes, label

        if clean_noise and binary_mask.any():
            labeled_array, num_features = label(binary_mask)
            if num_features > 0:
                component_sizes = np.bincount(labeled_array.ravel())
                component_sizes[0] = 0  # Ignore background label 0
                if min_component_size is None or min_component_size <= 0:
                    # Keep only the largest single component (main character body)
                    largest_label = component_sizes.argmax()
                    binary_mask = (labeled_array == largest_label)
                else:
                    # Keep components larger than min_component_size (or largest if none)
                    valid_labels = np.where(component_sizes >= min_component_size)[0]
                    if len(valid_labels) > 0:
                        binary_mask = np.isin(labeled_array, valid_labels)
                    else:
                        largest_label = component_sizes.argmax()
                        binary_mask = (labeled_array == largest_label)

        if fill_holes and binary_mask.any():
            binary_mask = binary_fill_holes(binary_mask)

        out_np = np.where(binary_mask, np.maximum(alpha_np, 1.0 if threshold is not None else alpha_np), 0.0)
        return torch.from_numpy(out_np).unsqueeze(0).to(device=alpha.device, dtype=alpha.dtype)
    except ImportError:
        # Fallback when scipy is not installed
        import torch.nn.functional as F

        kernel_size = 5
        pad = kernel_size // 2
        tensor_mask = (alpha > threshold).float().unsqueeze(0)  # [1, 1, H, W]
        dilated = F.max_pool2d(tensor_mask, kernel_size=kernel_size, stride=1, padding=pad)
        eroded = -F.max_pool2d(-dilated, kernel_size=kernel_size, stride=1, padding=pad)
        return torch.max(alpha, eroded.squeeze(0))


def fill_alpha_holes(alpha, threshold=0.5):
    return clean_alpha_mask(alpha, threshold=threshold, fill_holes=True, clean_noise=False)


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


def is_merged_image(pil_img, split_mode="auto"):
    if split_mode == "always":
        return True
    if split_mode == "never":
        return False
    w, h = pil_img.size
    # Merged front+back renders have side-by-side layout (w/h >= 0.75, typically ~1.0 for two 1:2 views)
    return (w / h) >= 0.75


def process_single_image(model, image, device, args, bg_color):
    rgb = image_to_tensor(image).to(device)
    with torch.no_grad():
        alpha = model(rgb.unsqueeze(0))[0].clamp(0.0, 1.0)
    if getattr(args, "bg_threshold", None) is not None and args.bg_threshold > 0.0:
        alpha = torch.where(alpha < args.bg_threshold, torch.zeros_like(alpha), alpha)
    clean_noise = getattr(args, "clean_noise", False)
    fill_holes = getattr(args, "fill_holes", False)
    if clean_noise or fill_holes:
        thresh = args.threshold if args.threshold is not None else args.hole_threshold
        alpha = clean_alpha_mask(
            alpha,
            threshold=thresh,
            fill_holes=fill_holes,
            clean_noise=clean_noise,
            min_component_size=getattr(args, "min_component_size", 100),
        )
    elif args.threshold is not None:
        alpha = (alpha >= args.threshold).to(dtype=alpha.dtype)
    out_rgb = uncompose_background(rgb, alpha, bg_color, args.min_alpha) if args.uncompose else rgb
    return out_rgb, alpha


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Extract foreground alpha from RGB render PNGs.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", default=None, help="Single input RGB/RGBA image.")
    parser.add_argument("--inputs", nargs="*", default=None, help="Multiple input RGB/RGBA images.")
    parser.add_argument("--output", default=None, help="Output RGBA PNG for --input.")
    parser.add_argument("--output_dir", default="foreground_alpha_outputs", help="Output folder for --inputs.")
    parser.add_argument("--output_front", default=None, help="Explicit output file path for split front RGBA image.")
    parser.add_argument("--output_back", default=None, help="Explicit output file path for split back RGBA image.")
    parser.add_argument("--bg_color", default="0,0,0", help="Known input background for --uncompose, as r,g,b.")
    parser.add_argument("--uncompose", action="store_true", help="Recover foreground RGB from a known solid background.")
    parser.add_argument("--fill_holes", action="store_true", help="Fill interior transparent holes inside predicted alpha mask.")
    parser.add_argument("--clean_noise", action="store_true", help="Remove isolated floating background noise islands using connected component analysis.")
    parser.add_argument("--min_component_size", type=int, default=100, help="Minimum pixel area for connected components (0 or <=0 keeps only the single largest main body component).")
    parser.add_argument("--hole_threshold", type=float, default=0.5, help="Binary threshold for hole filling and noise cleaning.")
    parser.add_argument("--bg_threshold", type=float, default=0.15, help="Alpha threshold below which background noise is suppressed to 0.0.")
    parser.add_argument("--min_alpha", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=None, help="Optional hard alpha threshold.")
    parser.add_argument(
        "--split_merged",
        choices=["auto", "always", "never"],
        default="auto",
        help="Automatically split merged front/back image (left half front, right half back).",
    )
    parser.add_argument(
        "--save_split",
        action="store_true",
        help="When processing merged images, also save separated front and back RGBA images.",
    )
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
    if (args.output or args.output_front or args.output_back) and len(input_paths) != 1:
        raise ValueError("--output, --output_front, and --output_back can only be used with a single input image.")

    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_args = checkpoint.get("args", {})
    base_channels = checkpoint_args.get("base_channels", 32)
    model = ForegroundAlphaNet(base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    bg_color = parse_color(args.bg_color)
    for input_path in input_paths:
        input_path = Path(input_path)
        pil_img = Image.open(input_path).convert("RGB")

        if is_merged_image(pil_img, args.split_merged):
            w, h = pil_img.size
            split_w = w // 2
            front_img = pil_img.crop((0, 0, split_w, h))
            back_img = pil_img.crop((split_w, 0, w, h))

            front_rgb, front_alpha = process_single_image(model, front_img, device, args, bg_color)
            back_rgb, back_alpha = process_single_image(model, back_img, device, args, bg_color)

            merged_rgb = torch.cat([front_rgb, back_rgb], dim=2)
            merged_alpha = torch.cat([front_alpha, back_alpha], dim=2)

            output_path = output_path_for(input_path, args)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tensor_to_rgba_image(merged_rgb.cpu(), merged_alpha.cpu()).save(output_path)
            print(f"Saved merged output {output_path}")

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
                tensor_to_rgba_image(front_rgb.cpu(), front_alpha.cpu()).save(front_out_path)
                print(f"Saved front split output {front_out_path}")

            if back_out_path:
                back_out_path.parent.mkdir(parents=True, exist_ok=True)
                tensor_to_rgba_image(back_rgb.cpu(), back_alpha.cpu()).save(back_out_path)
                print(f"Saved back split output {back_out_path}")
        else:
            out_rgb, alpha = process_single_image(model, pil_img, device, args, bg_color)
            output_path = output_path_for(input_path, args)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tensor_to_rgba_image(out_rgb.cpu(), alpha.cpu()).save(output_path)
            print(f"Saved {output_path}")


if __name__ == "__main__":
    main()


