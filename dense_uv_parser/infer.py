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
from SkingToolkit.dense_uv_parser.utils import (  # noqa: E402
    IGNORE_INDEX,
    LAYER_PALETTE,
    PART_PALETTE,
    SPLAT_COLOR_AGGREGATIONS,
    colorize_foreground,
    colorize_labels,
    colorize_surface,
    colorize_uv,
    flat_uv_to_uv01,
    parse_views,
    prediction_uv01,
    splat_parser_predictions_to_uv_conditioning,
    surface_class_count,
)
from SkingToolkit.inverse_uv.dataset import finalize_minecraft_alpha, tensor_to_rgba_image, view_native_size  # noqa: E402
from SkingToolkit.inverse_uv.model import InverseUVNet  # noqa: E402
from SkingToolkit.inverse_uv.train import get_device  # noqa: E402
from SkingToolkit.renderer import DifferentiableRenderer  # noqa: E402


def image_to_render_tensor(image, view_size, bg_color=(128, 128, 128)):
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        alpha = TF.to_tensor(rgba)[3:4]
        rgb = TF.to_tensor(rgba.convert("RGB"))
        bg = rgb.new_tensor(bg_color).view(3, 1, 1) / 255.0
        tensor = alpha * rgb + (1.0 - alpha) * bg
    else:
        tensor = TF.to_tensor(image.convert("RGB"))

    if tuple(tensor.shape[-2:]) != tuple(view_size):
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=view_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    tensor = tensor.clamp(0.0, 1.0)
    return torch.cat([tensor, torch.ones_like(tensor[:1])], dim=0)


def load_parser(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_args = checkpoint.get("args", {})
    model_config = checkpoint.get("model_config", {})
    state_dict = checkpoint["model"]
    has_uv_classification = any(key.startswith("uv_x.") or key.startswith("uv_y.") for key in state_dict)
    uv_classification = model_config.get("uv_classification", has_uv_classification)
    parser_mode = model_config.get("parser_mode", checkpoint_args.get("parser_mode", "dense"))
    predict_affine = model_config.get("predict_affine", parser_mode == "global_affine")
    model = DenseUVParserNet(
        base_channels=model_config.get("base_channels", checkpoint_args.get("base_channels", 32)),
        uv_size=model_config.get("uv_size", 64),
        uv_classification=uv_classification,
        view_classes=model_config.get("view_classes", 0),
        predict_affine=predict_affine,
        affine_translation_scale=model_config.get(
            "affine_translation_scale", checkpoint_args.get("translation_scale", 0.03)
        ),
        affine_scale_range=model_config.get("affine_scale_range", checkpoint_args.get("scale_range", 0.03)),
        surface_classes=model_config.get(
            "surface_classes",
            checkpoint_args.get("surface_classes", 2 if predict_affine else 0),
        ),
    ).to(device)
    model.load_state_dict(state_dict)
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
    return model, checkpoint_args


def checkpoint_run_id(path):
    path = Path(path)
    return f"{path.parent.name}/{path.name}"


def load_view_images(args, views, renderer, bg_color=(128, 128, 128)):
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
        image_to_render_tensor(image, view_native_size(renderer, view), bg_color=bg_color)
        for image, view in zip(images, views)
    ]
    return torch.stack(tensors, dim=0)


def save_conditioning_preview(conditioning, output_path):
    inner_rgb = conditioning[:, 0:3]
    outer_rgb = conditioning[:, 5:8]
    preview = torch.cat([inner_rgb, outer_rgb], dim=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(preview.clamp(0.0, 1.0), output_path, nrow=conditioning.shape[0])


def save_debug_preview(
    rendered,
    outputs,
    view_count,
    output_path,
    fg_threshold,
    bg_color=(128, 128, 128),
    routing=None,
):
    pred_fg = routing["foreground"] if routing is not None else torch.sigmoid(outputs["foreground"])[:, 0] > fg_threshold
    pred_part = torch.where(
        pred_fg,
        outputs["part"].argmax(dim=1),
        torch.full_like(outputs["part"].argmax(dim=1), IGNORE_INDEX),
    )
    pred_layer_values = routing["layer"] if routing is not None else outputs["layer"].argmax(dim=1)
    pred_layer = torch.where(
        pred_fg,
        pred_layer_values,
        torch.full_like(outputs["layer"].argmax(dim=1), IGNORE_INDEX),
    )
    pred_uv = (
        flat_uv_to_uv01(routing["flat_uv"], rendered.dtype)
        if routing is not None
        else prediction_uv01(outputs)
    )

    debug_images = [
        rendered[:, :3],
        colorize_foreground(pred_fg, bg_color, rendered),
        colorize_labels(pred_part, PART_PALETTE, bg_color, rendered),
        colorize_labels(pred_layer, LAYER_PALETTE, bg_color, rendered),
    ]
    if routing is not None:
        pred_surface = torch.where(
            pred_fg,
            routing["surface"],
            torch.full_like(routing["surface"], IGNORE_INDEX),
        )
        debug_images.append(colorize_surface(pred_surface, bg_color, rendered))
    debug_images.append(colorize_uv(pred_uv, pred_fg, bg_color))
    debug_preview = torch.cat(debug_images, dim=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(debug_preview.clamp(0.0, 1.0).detach().cpu(), output_path, nrow=view_count)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Infer UV conditioning with a dense UV parser.")
    parser.add_argument("--parser_checkpoint", required=True)
    parser.add_argument("--inpaint_checkpoint", default=None, help="Optional inverse_uv checkpoint used to inpaint final skin.")
    parser.add_argument("--output", default=None, help="Final RGBA UV PNG path; requires --inpaint_checkpoint.")
    parser.add_argument("--conditioning_output", default=None, help="Optional preview image for parser-splatted conditioning.")
    parser.add_argument("--debug_output", default=None, help="Optional path to write a debug preview grid of predictions.")
    parser.add_argument("--front", default=None)
    parser.add_argument("--back", default=None)
    parser.add_argument("--combined", default=None)
    parser.add_argument("--view_images", nargs="*", default=None)
    parser.add_argument("--mappings_dir", default=None)
    parser.add_argument("--fg_threshold", type=float, default=0.5)
    parser.add_argument("--no_semantic_gate", dest="semantic_gate", action="store_false", default=None)
    parser.add_argument("--semantic_gate_radius", type=int, default=None)
    parser.add_argument("--splat_color_aggregation", choices=SPLAT_COLOR_AGGREGATIONS, default=None)
    parser.add_argument("--splat_color_mode_bits", type=int, default=None)
    parser.add_argument("--splat_color_mode_confidence_ratio", type=float, default=None)
    parser.add_argument("--alpha_threshold", type=float, default=0.5)
    parser.add_argument("--no_enforce_base_alpha", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.output and not args.inpaint_checkpoint:
        raise ValueError("--output requires --inpaint_checkpoint.")
    if not args.output and not args.conditioning_output and not args.debug_output:
        raise ValueError("Provide --output, --conditioning_output, and/or --debug_output.")

    device = get_device(args.device)
    parser_model, parser_args = load_parser(args.parser_checkpoint, device)
    views = parse_views(parser_args.get("views", "walk_front_both_layer_ortho,walk_back_both_layer_ortho"))
    if parser_model.view_classes not in (0, len(views)):
        raise ValueError(
            f"Parser checkpoint expects {parser_model.view_classes} views, but its metadata lists {len(views)}: {views}"
        )
    mappings_dir = args.mappings_dir or parser_args.get("mappings_dir")
    renderer = DifferentiableRenderer(mappings_dir=mappings_dir)
    missing_views = [view for view in views if view not in renderer.views]
    if missing_views:
        raise ValueError(f"Unknown renderer views {missing_views}. Available views: {', '.join(renderer.views)}")
    if parser_model.predict_affine:
        mapping_surface_classes = surface_class_count(renderer, views)
        if parser_model.surface_classes != mapping_surface_classes:
            raise ValueError(
                "Parser/mapping surface-slot mismatch: "
                f"checkpoint={parser_model.surface_classes}, mappings={mapping_surface_classes}."
            )

    bg_color = parser_args.get("bg_color", (128, 128, 128))
    semantic_gate = parser_args.get("semantic_gate", True) if args.semantic_gate is None else args.semantic_gate
    checkpoint_gate_radius = parser_args.get("semantic_gate_radius")
    semantic_gate_radius = (
        args.semantic_gate_radius
        if args.semantic_gate_radius is not None
        else 1 if checkpoint_gate_radius is None else checkpoint_gate_radius
    )
    color_aggregation = args.splat_color_aggregation or parser_args.get("splat_color_aggregation") or "quantized_mode"
    color_mode_bits = (
        args.splat_color_mode_bits
        if args.splat_color_mode_bits is not None
        else parser_args.get("splat_color_mode_bits") or 5
    )
    color_mode_confidence_ratio = (
        args.splat_color_mode_confidence_ratio
        if args.splat_color_mode_confidence_ratio is not None
        else parser_args.get("splat_color_mode_confidence_ratio") or 0.85
    )

    inpaint_model = None
    inpaint_args = None
    if args.output:
        inpaint_model, inpaint_args = load_inpaint(args.inpaint_checkpoint, device)
        inpaint_views = parse_views(inpaint_args.get("views", ""))
        if inpaint_views and inpaint_views != views:
            raise ValueError(f"Parser/inpaint view mismatch: parser={views}, inpaint={inpaint_views}")
        expected_parser = inpaint_args.get("parser_checkpoint")
        if expected_parser and checkpoint_run_id(expected_parser) != checkpoint_run_id(args.parser_checkpoint):
            raise ValueError(
                "The inverse_uv checkpoint was trained with a different parser: "
                f"expected {checkpoint_run_id(expected_parser)}, got {checkpoint_run_id(args.parser_checkpoint)}."
            )
        expected_semantic_gate = inpaint_args.get("parser_semantic_gate")
        if expected_semantic_gate is not None and bool(expected_semantic_gate) != semantic_gate:
            raise ValueError(
                "Parser semantic-gate setting does not match the inverse_uv checkpoint: "
                f"checkpoint={expected_semantic_gate}, requested={semantic_gate}."
            )
        expected_gate_radius = inpaint_args.get("parser_semantic_gate_radius")
        if expected_gate_radius is not None and int(expected_gate_radius) != semantic_gate_radius:
            raise ValueError(
                "Parser semantic-gate radius does not match the inverse_uv checkpoint: "
                f"checkpoint={expected_gate_radius}, requested={semantic_gate_radius}."
            )
        expected_aggregation = inpaint_args.get("parser_splat_color_aggregation", "best")
        if expected_aggregation != color_aggregation:
            raise ValueError(
                "Parser conditioning color aggregation does not match the inverse_uv checkpoint: "
                f"checkpoint={expected_aggregation}, requested={color_aggregation}. "
                "Use the checkpoint's mode for a legacy comparison, or retrain inverse_uv with quantized_mode."
            )
        if expected_aggregation == "quantized_mode":
            expected_bits = int(inpaint_args.get("parser_splat_color_mode_bits", 5))
            expected_ratio = float(inpaint_args.get("parser_splat_color_mode_confidence_ratio", 0.85))
            if expected_bits != color_mode_bits or abs(expected_ratio - color_mode_confidence_ratio) > 1e-9:
                raise ValueError(
                    "Parser conditioning color-mode settings do not match the inverse_uv checkpoint: "
                    f"checkpoint=bits:{expected_bits}, ratio:{expected_ratio}; "
                    f"requested=bits:{color_mode_bits}, ratio:{color_mode_confidence_ratio}."
                )

    rendered = load_view_images(args, views, renderer, bg_color=bg_color).to(device)
    view_ids = torch.arange(len(views), device=device)
    with torch.no_grad():
        outputs = parser_model(rendered, view_ids=view_ids)
        conditioning, routing_details = splat_parser_predictions_to_uv_conditioning(
            rendered,
            outputs,
            renderer=renderer,
            views=views,
            group_size=len(views),
            fg_threshold=args.fg_threshold,
            bg_color=bg_color,
            semantic_gate=semantic_gate,
            semantic_gate_radius=semantic_gate_radius,
            color_aggregation=color_aggregation,
            color_mode_bits=color_mode_bits,
            color_mode_confidence_ratio=color_mode_confidence_ratio,
            return_details=True,
        )

    if args.debug_output:
        save_debug_preview(
            routing_details["rendered"],
            routing_details["outputs"],
            len(views),
            Path(args.debug_output),
            args.fg_threshold,
            bg_color=bg_color,
            routing=routing_details["routing"],
        )

    if args.conditioning_output:
        save_conditioning_preview(conditioning.detach().cpu(), Path(args.conditioning_output))

    if args.output:
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
