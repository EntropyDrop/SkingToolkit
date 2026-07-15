import argparse
import json
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
    FACE_PALETTE,
    IGNORE_INDEX,
    LAYER_FACE_PALETTE,
    LAYER_PALETTE,
    PART_PALETTE,
    ROUTE_ROLE_PALETTE,
    SPLAT_COLOR_AGGREGATIONS,
    combine_layer_face,
    build_geometry_grid_debug,
    fill_geometry_grid_debug,
    overlay_geometry_grid_debug,
    colorize_foreground,
    colorize_labels,
    colorize_surface,
    colorize_uv,
    conditioning_to_pred_uv,
    flat_uv_to_uv01,
    parse_views,
    prediction_uv01,
    splat_parser_predictions_to_uv_conditioning,
    surface_class_count,
)
from SkingToolkit.uv_inpainting.dataset import finalize_minecraft_alpha, tensor_to_rgba_image, view_native_size  # noqa: E402
from SkingToolkit.uv_inpainting.model import UVInpaintingNet  # noqa: E402
from SkingToolkit.uv_inpainting.train import get_device  # noqa: E402
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
            mode="nearest",
        ).squeeze(0)
    tensor = tensor.clamp(0.0, 1.0)
    return torch.cat([tensor, torch.ones_like(tensor[:1])], dim=0)


def load_parser(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_args = checkpoint.get("args", {})
    model_config = checkpoint.get("model_config", {})
    if model_config.get("arm_model", "steve") != "steve":
        raise ValueError("Geometry parser only supports standard Steve arms.")
    state_dict = checkpoint["model"]
    has_uv_classification = any(key.startswith("uv_x.") or key.startswith("uv_y.") for key in state_dict)
    has_layer_face = any(key.startswith("layer_face.") for key in state_dict)
    uv_classification = model_config.get("uv_classification", has_uv_classification)
    parser_mode = model_config.get("parser_mode", checkpoint_args.get("parser_mode", "dense"))
    predict_affine = model_config.get("predict_affine", parser_mode in ("global_affine", "geometry_fit"))
    geometry_only = model_config.get("geometry_only", parser_mode == "geometry_fit")
    layer_classes = model_config.get("layer_classes", 2)
    if geometry_only and layer_classes != 3:
        raise ValueError(
            "This geometry parser predates the secondary/backface route class. "
            "Train a new dense_uv_parser checkpoint."
        )
    model = DenseUVParserNet(
        base_channels=model_config.get("base_channels", checkpoint_args.get("base_channels", 32)),
        uv_size=model_config.get("uv_size", 64),
        uv_classification=uv_classification,
        layer_classes=layer_classes,
        layer_face_classes=model_config.get("layer_face_classes", 12 if has_layer_face else 0),
        view_classes=model_config.get("view_classes", 0),
        predict_affine=predict_affine,
        affine_translation_scale=model_config.get(
            "affine_translation_scale", checkpoint_args.get("translation_scale", 0.03)
        ),
        affine_scale_range=model_config.get("affine_scale_range", checkpoint_args.get("scale_range", 0.03)),
        surface_classes=model_config.get(
            "surface_classes",
            checkpoint_args.get("surface_classes", 0 if geometry_only else 2 if predict_affine else 0),
        ),
        geometry_only=geometry_only,
        feature_dropout=model_config.get(
            "feature_dropout", checkpoint_args.get("feature_dropout", 0.0)
        ),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint_args


def load_inpaint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_args = checkpoint.get("args", {})
    input_channels = checkpoint.get("input_channels", checkpoint_args.get("input_channels", 10))
    model = UVInpaintingNet(
        input_channels=input_channels,
        base_channels=checkpoint_args.get("base_channels", 64),
        preserve_known=checkpoint_args.get("preserve_known", True),
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


def save_parser_uv(
    conditioning,
    output_path,
    alpha_threshold=0.5,
    enforce_base_alpha=True,
):
    parser_uv = conditioning_to_pred_uv(conditioning)
    if parser_uv.dim() != 4 or parser_uv.shape[0] != 1:
        raise ValueError(
            "Parser UV PNG output requires exactly one conditioning sample, "
            f"got {tuple(parser_uv.shape)}."
        )
    parser_uv = finalize_minecraft_alpha(
        parser_uv[0],
        alpha_threshold=alpha_threshold,
        enforce_base_alpha=enforce_base_alpha,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_rgba_image(parser_uv.detach().cpu()).save(output_path)
    print(f"Saved {output_path}")


def save_debug_preview(
    rendered,
    outputs,
    view_count,
    output_path,
    fg_threshold,
    bg_color=(128, 128, 128),
    routing=None,
    overlay_output=None,
    overlay_alpha=0.45,
    inner_cutout_output=None,
    outer_cutout_output=None,
    secondary_cutout_output=None,
    face_output=None,
    layer_face_output=None,
    raw_face_output=None,
    raw_layer_face_output=None,
    geometry_grid_output=None,
    geometry_overlay_output=None,
    geometry_routed_overlay_output=None,
    geometry_fill_output=None,
    renderer=None,
    views=None,
):
    if not 0.0 <= overlay_alpha <= 1.0:
        raise ValueError(f"overlay_alpha must be in [0, 1], got {overlay_alpha}.")
    pred_fg = routing["foreground"] if routing is not None else torch.sigmoid(outputs["foreground"])[:, 0] > fg_threshold
    pred_part_values = (
        outputs["part"].argmax(dim=1)
        if "part" in outputs
        else routing["part"]
    )
    pred_part = torch.where(pred_fg, pred_part_values, torch.full_like(pred_part_values, IGNORE_INDEX))
    raw_layer_values = outputs["layer"].argmax(dim=1)
    raw_fg = torch.sigmoid(outputs["foreground"])[:, 0] > fg_threshold
    pred_layer_values = routing["layer"] if routing is not None else raw_layer_values
    pred_layer = torch.where(
        pred_fg,
        pred_layer_values,
        torch.full_like(outputs["layer"].argmax(dim=1), IGNORE_INDEX),
    )
    raw_face_values = outputs["face"].argmax(dim=1) if "face" in outputs else routing["face"]
    raw_face = torch.where(
        pred_fg,
        raw_face_values,
        torch.full_like(raw_face_values, IGNORE_INDEX),
    )
    pred_face_values = routing["face"] if routing is not None else raw_face_values
    pred_face = torch.where(
        pred_fg,
        pred_face_values,
        torch.full_like(pred_face_values, IGNORE_INDEX),
    )
    if "layer_face" in outputs:
        raw_layer_face_values = outputs["layer_face"].argmax(dim=1)
        raw_layer_face = torch.where(
            pred_fg,
            raw_layer_face_values,
            torch.full_like(raw_layer_face_values, IGNORE_INDEX),
        )
    else:
        raw_layer = pred_layer if outputs["layer"].shape[1] == 3 else torch.where(
            pred_fg, raw_layer_values, torch.full_like(raw_layer_values, IGNORE_INDEX)
        )
        raw_layer_face = combine_layer_face(raw_layer, raw_face)
    pred_layer_face = combine_layer_face(pred_layer, pred_face)
    pred_uv = flat_uv_to_uv01(routing["flat_uv"], rendered.dtype) if routing is not None else prediction_uv01(outputs)

    part_color = colorize_labels(pred_part, PART_PALETTE, bg_color, rendered)
    layer_color = colorize_labels(pred_layer, LAYER_PALETTE, bg_color, rendered)
    route_role_values = routing.get("route_role", raw_layer_values) if routing is not None else raw_layer_values
    route_role_mask = (
        pred_fg | routing.get("secondary", torch.zeros_like(pred_fg))
        if routing is not None
        else raw_fg
    )
    route_role = torch.where(
        route_role_mask,
        route_role_values,
        torch.full_like(route_role_values, IGNORE_INDEX),
    )
    route_role_color = colorize_labels(route_role, ROUTE_ROLE_PALETTE, bg_color, rendered)
    raw_face_color = colorize_labels(raw_face, FACE_PALETTE, bg_color, rendered)
    face_color = colorize_labels(pred_face, FACE_PALETTE, bg_color, rendered)
    raw_layer_face_color = colorize_labels(raw_layer_face, LAYER_FACE_PALETTE, bg_color, rendered)
    layer_face_color = colorize_labels(pred_layer_face, LAYER_FACE_PALETTE, bg_color, rendered)
    geometry_images = None
    geometry_overlays = None
    geometry_routed_overlays = None
    if renderer is not None and views is not None and routing is not None:
        geometry_debug = build_geometry_grid_debug(
            renderer, views, rendered.shape[0], rendered, bg_color=bg_color
        )
        inner_grid, outer_grid = geometry_debug[:2]
        inner_fill, outer_fill = fill_geometry_grid_debug(
            rendered, pred_fg, pred_layer_values, geometry_debug, bg_color=bg_color
        )
        geometry_images = (inner_grid, outer_grid, inner_fill, outer_fill)
        geometry_overlays = overlay_geometry_grid_debug(rendered, geometry_debug)
        geometry_routed_overlays = overlay_geometry_grid_debug(
            rendered,
            geometry_debug,
            base_images=(inner_fill, outer_fill),
        )
    debug_images = [
        rendered[:, :3],
        colorize_foreground(pred_fg, bg_color, rendered),
        part_color,
        layer_color,
        route_role_color,
        raw_face_color,
        face_color,
        raw_layer_face_color,
        layer_face_color,
    ]
    if geometry_images is not None:
        debug_images.extend(geometry_overlays)
        debug_images.extend(geometry_routed_overlays)
        debug_images.extend(geometry_images)
    surface_color = None
    if routing is not None:
        pred_surface = torch.where(
            pred_fg,
            routing["surface"],
            torch.full_like(routing["surface"], IGNORE_INDEX),
        )
        surface_color = colorize_surface(pred_surface, bg_color, rendered)
        debug_images.append(surface_color)
    debug_images.append(colorize_uv(pred_uv, pred_fg, bg_color))
    if output_path is not None:
        debug_preview = torch.cat(debug_images, dim=0)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(debug_preview.clamp(0.0, 1.0).detach().cpu(), output_path, nrow=view_count)

    for colorized, path in (
        (face_color, face_output),
        (layer_face_color, layer_face_output),
        (raw_face_color, raw_face_output),
        (raw_layer_face_color, raw_layer_face_output),
    ):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            save_image(colorized.clamp(0.0, 1.0).detach().cpu(), path, nrow=view_count)

    if geometry_images is not None:
        for images, path in (
            (geometry_overlays, geometry_overlay_output),
            (geometry_routed_overlays, geometry_routed_overlay_output),
            (geometry_images[:2], geometry_grid_output),
            (geometry_images[2:], geometry_fill_output),
        ):
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                save_image(
                    torch.cat(images, dim=0).clamp(0.0, 1.0).detach().cpu(),
                    path,
                    nrow=view_count,
                )

    if (
        overlay_output is not None
        or inner_cutout_output is not None
        or outer_cutout_output is not None
        or secondary_cutout_output is not None
    ):
        rgb = rendered[:, :3]
        mask = pred_fg.unsqueeze(1)
        bg = rgb.new_tensor(bg_color).view(1, 3, 1, 1) / 255.0
        routed_original = torch.where(mask, rgb, bg.expand_as(rgb))
        inner_mask = (pred_fg & (pred_layer_values == 0)).unsqueeze(1)
        outer_mask = (pred_fg & (pred_layer_values == 1)).unsqueeze(1)
        secondary_mask = (
            routing.get("secondary", raw_fg & (raw_layer_values == 2))
            if routing is not None
            else raw_fg & (raw_layer_values == 2)
        ).unsqueeze(1)
        inner_cutout = torch.where(inner_mask, rgb, bg.expand_as(rgb))
        outer_cutout = torch.where(outer_mask, rgb, bg.expand_as(rgb))
        secondary_cutout = torch.where(secondary_mask, rgb, bg.expand_as(rgb))

        def save_cutout(cutout, path):
            if path is None:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            save_image(cutout.clamp(0.0, 1.0).detach().cpu(), path, nrow=view_count)

        save_cutout(inner_cutout, inner_cutout_output)
        save_cutout(outer_cutout, outer_cutout_output)
        save_cutout(secondary_cutout, secondary_cutout_output)

        def overlay(colorized):
            blended = rgb * (1.0 - overlay_alpha) + colorized * overlay_alpha
            return torch.where(mask, blended, rgb)

        if overlay_output is not None:
            overlay_images = [
                rgb,
                routed_original,
                *(geometry_overlays or ()),
                *(geometry_routed_overlays or ()),
                inner_cutout,
                outer_cutout,
                secondary_cutout,
                overlay(part_color),
                overlay(layer_color),
                overlay(route_role_color),
                overlay(face_color),
                overlay(layer_face_color),
            ]
            if surface_color is not None:
                overlay_images.append(overlay(surface_color))
            overlay_preview = torch.cat(overlay_images, dim=0)
            overlay_output.parent.mkdir(parents=True, exist_ok=True)
            save_image(overlay_preview.clamp(0.0, 1.0).detach().cpu(), overlay_output, nrow=view_count)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Infer UV conditioning with a dense UV parser.")
    parser.add_argument("--parser_checkpoint", required=True)
    parser.add_argument("--inpaint_checkpoint", default=None, help="Optional uv_inpainting checkpoint used to inpaint final skin.")
    parser.add_argument("--output", default=None, help="Final RGBA UV PNG path; requires --inpaint_checkpoint.")
    parser.add_argument("--conditioning_output", default=None, help="Optional preview image for parser-splatted conditioning.")
    parser.add_argument(
        "--parser_uv_output",
        default=None,
        help="Optional preliminary RGBA skin merged directly from parser conditioning.",
    )
    parser.add_argument("--debug_output", default=None, help="Optional path to write a debug preview grid of predictions.")
    parser.add_argument("--overlay_output", default=None, help="Optional path for segmentation overlays on canonicalized input views.")
    parser.add_argument("--overlay_alpha", type=float, default=0.45)
    parser.add_argument("--inner_cutout_output", default=None, help="Original-color cutout for routed inner-layer pixels.")
    parser.add_argument("--outer_cutout_output", default=None, help="Original-color cutout for routed outer/decor pixels.")
    parser.add_argument(
        "--secondary_cutout_output",
        default=None,
        help="Original-color cutout for secondary/deeper surface pixels.",
    )
    parser.add_argument("--face_output", default=None, help="Six-class routed cube-face visualization.")
    parser.add_argument("--layer_face_output", default=None, help="Twelve-class inner/outer-by-face visualization.")
    parser.add_argument("--raw_face_output", default=None, help="Six-class raw face-head visualization.")
    parser.add_argument("--raw_layer_face_output", default=None, help="Twelve-class raw joint-head visualization.")
    parser.add_argument("--geometry_grid_output", default=None, help="Fitted inner/outer cuboid UV grid preview.")
    parser.add_argument(
        "--geometry_overlay_output",
        default=None,
        help="Inner/outer fitted UV texel grids overlaid on canonicalized source views.",
    )
    parser.add_argument(
        "--geometry_routed_overlay_output",
        default=None,
        help="Inner/outer UV grids overlaid on only the pixels routed to that layer.",
    )
    parser.add_argument("--geometry_fill_output", default=None, help="Classified RGB filled onto inner/outer cuboid grids.")
    parser.add_argument("--front", default=None)
    parser.add_argument("--back", default=None)
    parser.add_argument("--combined", default=None)
    parser.add_argument("--view_images", nargs="*", default=None)
    parser.add_argument("--mappings_dir", default=None)
    parser.add_argument("--fg_threshold", type=float, default=0.5)
    parser.add_argument("--route_confidence_threshold", type=float, default=0.0)
    parser.add_argument("--route_margin_threshold", type=float, default=0.0)
    parser.add_argument("--outer_route_confidence_threshold", type=float, default=0.55)
    parser.add_argument("--outer_route_margin_threshold", type=float, default=0.35)
    parser.add_argument(
        "--outer_uv_min_coverage",
        type=float,
        default=None,
        help="Reject outer UV texels supported by less than this fraction of their projected footprint.",
    )
    parser.add_argument(
        "--geometry_route_texel_consensus",
        dest="geometry_route_texel_consensus",
        action="store_true",
        default=None,
        help="Use projected UV-cell voting instead of semantic-first per-pixel routing.",
    )
    parser.add_argument(
        "--no_geometry_route_texel_consensus",
        dest="geometry_route_texel_consensus",
        action="store_false",
    )
    parser.add_argument(
        "--color_aggregation",
        choices=SPLAT_COLOR_AGGREGATIONS,
        default="exact_mode",
        help="How source colors mapping to the same UV texel are selected.",
    )
    parser.add_argument(
        "--allow_semantic_fallback",
        action="store_true",
        help="Keep pixels whose strict semantic routing had no valid candidate.",
    )
    parser.add_argument("--no_semantic_gate", dest="semantic_gate", action="store_false", default=None)
    parser.add_argument("--affine_refine", dest="affine_refine", action="store_true", default=None)
    parser.add_argument("--no_affine_refine", dest="affine_refine", action="store_false")
    parser.add_argument("--affine_refine_translation_px", type=float, default=None)
    parser.add_argument("--affine_refine_scale", type=float, default=None)
    parser.add_argument("--alpha_threshold", type=float, default=0.5)
    parser.add_argument("--no_enforce_base_alpha", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.output and not args.inpaint_checkpoint:
        raise ValueError("--output requires --inpaint_checkpoint.")
    if not any(
        (
            args.output,
            args.conditioning_output,
            args.parser_uv_output,
            args.debug_output,
            args.overlay_output,
            args.inner_cutout_output,
            args.outer_cutout_output,
            args.secondary_cutout_output,
            args.face_output,
            args.layer_face_output,
            args.raw_face_output,
            args.raw_layer_face_output,
            args.geometry_grid_output,
            args.geometry_overlay_output,
            args.geometry_routed_overlay_output,
            args.geometry_fill_output,
        )
    ):
        raise ValueError(
            "Provide --output, --conditioning_output, --parser_uv_output, --debug_output, --overlay_output, "
            "--inner_cutout_output, --outer_cutout_output, and/or --secondary_cutout_output."
        )

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
    if parser_model.predict_affine and parser_model.surface_classes > 0:
        mapping_surface_classes = surface_class_count(renderer, views)
        if parser_model.surface_classes != mapping_surface_classes:
            raise ValueError(
                "Parser/mapping surface-slot mismatch: "
                f"checkpoint={parser_model.surface_classes}, mappings={mapping_surface_classes}."
            )

    bg_color = parser_args.get("bg_color", (128, 128, 128))
    semantic_gate = parser_args.get("semantic_gate", True) if args.semantic_gate is None else args.semantic_gate
    affine_refine = parser_args.get("affine_refine", True) if args.affine_refine is None else args.affine_refine
    checkpoint_translation_px = parser_args.get("affine_refine_translation_px")
    affine_refine_translation_px = (
        args.affine_refine_translation_px
        if args.affine_refine_translation_px is not None
        else 8.0 if checkpoint_translation_px is None else checkpoint_translation_px
    )
    checkpoint_scale = parser_args.get("affine_refine_scale")
    affine_refine_scale = (
        args.affine_refine_scale
        if args.affine_refine_scale is not None
        else 0.0 if checkpoint_scale is None else checkpoint_scale
    )
    geometry_route_texel_consensus = (
        parser_args.get("geometry_route_texel_consensus", False)
        if args.geometry_route_texel_consensus is None
        else args.geometry_route_texel_consensus
    )
    outer_uv_min_coverage = (
        parser_args.get("outer_uv_min_coverage", 0.0)
        if args.outer_uv_min_coverage is None
        else args.outer_uv_min_coverage
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
            affine_refine=affine_refine,
            affine_refine_translation_px=affine_refine_translation_px,
            affine_refine_scale=affine_refine_scale,
            route_confidence_threshold=args.route_confidence_threshold,
            route_margin_threshold=args.route_margin_threshold,
            outer_route_confidence_threshold=args.outer_route_confidence_threshold,
            outer_route_margin_threshold=args.outer_route_margin_threshold,
            outer_uv_min_coverage=outer_uv_min_coverage,
            color_aggregation=args.color_aggregation,
            geometry_route_texel_consensus=geometry_route_texel_consensus,
            reject_semantic_fallback=not args.allow_semantic_fallback,
            return_details=True,
        )

    routing = routing_details.get("routing")
    if routing is not None:
        raw_count = int(routing["raw_foreground"].sum().item())
        rejected_count = int(routing["rejected"].sum().item())
        kept = routing["foreground"]
        kept_inner_count = int((kept & (routing["layer"] == 0)).sum().item())
        kept_outer_count = int((kept & (routing["layer"] == 1)).sum().item())
        raw_inner = routing["raw_foreground"] & (routing["layer"] == 0)
        raw_outer = routing["raw_foreground"] & (routing["layer"] == 1)
        rejected_inner = routing["rejected"] & (routing["layer"] == 0)
        rejected_outer = routing["rejected"] & (routing["layer"] == 1)
        coverage_rejected_outer = raw_outer & (
            routing.get("outer_uv_coverage", torch.ones_like(routing["confidence"]))
            < outer_uv_min_coverage
        )
        secondary_count = int(routing.get("secondary", torch.zeros_like(raw_outer)).sum().item())
        routed_secondary_count = int(
            routing.get("secondary_routed", torch.zeros_like(raw_outer)).sum().item()
        )
        rejected_secondary_count = int(
            routing.get("secondary_rejected", torch.zeros_like(raw_outer)).sum().item()
        )
        background_rejected_count = int(
            routing.get("background_rejected", torch.zeros_like(raw_outer)).sum().item()
        )
        raw_secondary_count = int(
            (
                (routing.get("raw_route_role", routing.get("route_role")) == 2)
                & (torch.sigmoid(routing_details["outputs"]["foreground"])[:, 0] > args.fg_threshold)
            ).sum().item()
        )
        print(
            "routing_filter="
            + json.dumps(
                {
                    "raw_pixels": raw_count,
                    "kept_pixels": raw_count - rejected_count,
                    "kept_inner_pixels": kept_inner_count,
                    "kept_outer_pixels": kept_outer_count,
                    "kept_outer_percent": round(
                        100.0 * kept_outer_count / max(kept_inner_count + kept_outer_count, 1),
                        3,
                    ),
                    "rejected_pixels": rejected_count,
                    "rejected_percent": round(100.0 * rejected_count / max(raw_count, 1), 3),
                    "inner_rejected_percent": round(
                        100.0 * int(rejected_inner.sum().item()) / max(int(raw_inner.sum().item()), 1),
                        3,
                    ),
                    "outer_rejected_percent": round(
                        100.0 * int(rejected_outer.sum().item()) / max(int(raw_outer.sum().item()), 1),
                        3,
                    ),
                    "outer_coverage_rejected_pixels": int(coverage_rejected_outer.sum().item()),
                    "background_rejected_pixels": background_rejected_count,
                    "secondary_backface_pixels": secondary_count,
                    "routed_secondary_pixels": routed_secondary_count,
                    "rejected_secondary_pixels": rejected_secondary_count,
                    "raw_secondary_backface_pixels": raw_secondary_count,
                },
                sort_keys=True,
            )
        )

    alignment = routing_details.get("alignment")
    if alignment is not None:
        translation = alignment["translation_px"].detach().cpu()
        scale_percent = alignment["scale_percent"].detach().cpu()
        score_before = alignment["score_before"].detach().cpu()
        score_after = alignment["score_after"].detach().cpu()
        accepted = alignment["accepted"].detach().cpu()
        for index in range(translation.shape[0]):
            print(
                "affine_refinement="
                + json.dumps(
                    {
                        "view": views[index % len(views)],
                        "accepted": bool(accepted[index]),
                        "dx_px": round(float(translation[index, 0]), 3),
                        "dy_px": round(float(translation[index, 1]), 3),
                        "scale_percent": round(float(scale_percent[index]), 4),
                        "score_before": round(float(score_before[index]), 6),
                        "score_after": round(float(score_after[index]), 6),
                    },
                    sort_keys=True,
                )
            )

    if any(
        (
            args.debug_output,
            args.overlay_output,
            args.inner_cutout_output,
            args.outer_cutout_output,
            args.secondary_cutout_output,
            args.face_output,
            args.layer_face_output,
            args.raw_face_output,
            args.raw_layer_face_output,
            args.geometry_grid_output,
            args.geometry_overlay_output,
            args.geometry_routed_overlay_output,
            args.geometry_fill_output,
        )
    ):
        save_debug_preview(
            routing_details["rendered"],
            routing_details["outputs"],
            len(views),
            Path(args.debug_output) if args.debug_output else None,
            args.fg_threshold,
            bg_color=bg_color,
            routing=routing_details["routing"],
            overlay_output=Path(args.overlay_output) if args.overlay_output else None,
            overlay_alpha=args.overlay_alpha,
            inner_cutout_output=Path(args.inner_cutout_output) if args.inner_cutout_output else None,
            outer_cutout_output=Path(args.outer_cutout_output) if args.outer_cutout_output else None,
            secondary_cutout_output=(
                Path(args.secondary_cutout_output) if args.secondary_cutout_output else None
            ),
            face_output=Path(args.face_output) if args.face_output else None,
            layer_face_output=Path(args.layer_face_output) if args.layer_face_output else None,
            raw_face_output=Path(args.raw_face_output) if args.raw_face_output else None,
            raw_layer_face_output=Path(args.raw_layer_face_output) if args.raw_layer_face_output else None,
            geometry_grid_output=Path(args.geometry_grid_output) if args.geometry_grid_output else None,
            geometry_overlay_output=(
                Path(args.geometry_overlay_output) if args.geometry_overlay_output else None
            ),
            geometry_routed_overlay_output=(
                Path(args.geometry_routed_overlay_output)
                if args.geometry_routed_overlay_output
                else None
            ),
            geometry_fill_output=Path(args.geometry_fill_output) if args.geometry_fill_output else None,
            renderer=renderer,
            views=views,
        )

    if args.conditioning_output:
        save_conditioning_preview(conditioning.detach().cpu(), Path(args.conditioning_output))

    if args.parser_uv_output:
        save_parser_uv(
            conditioning.detach().cpu(),
            Path(args.parser_uv_output),
            alpha_threshold=args.alpha_threshold,
            enforce_base_alpha=not args.no_enforce_base_alpha,
        )

    if args.output:
        inpaint_model, inpaint_args = load_inpaint(args.inpaint_checkpoint, device)
        print(
            "inpaint_config="
            + json.dumps(
                {
                    "checkpoint": checkpoint_run_id(args.inpaint_checkpoint),
                    "preserve_known": bool(inpaint_args.get("preserve_known", True)),
                },
                sort_keys=True,
            )
        )
        inpaint_views = parse_views(inpaint_args.get("views", ""))
        if inpaint_views and inpaint_views != views:
            raise ValueError(f"Parser/inpaint view mismatch: parser={views}, inpaint={inpaint_views}")
        expected_parser = inpaint_args.get("parser_checkpoint")
        if expected_parser and checkpoint_run_id(expected_parser) != checkpoint_run_id(args.parser_checkpoint):
            raise ValueError(
                "The uv_inpainting checkpoint was trained with a different parser: "
                f"expected {checkpoint_run_id(expected_parser)}, got {checkpoint_run_id(args.parser_checkpoint)}."
            )
        expected_refine = inpaint_args.get("parser_affine_refine")
        if expected_refine is not None and bool(expected_refine) != affine_refine:
            raise ValueError(
                "Parser affine-refinement setting does not match the uv_inpainting checkpoint: "
                f"checkpoint={expected_refine}, requested={affine_refine}."
            )
        expected_translation_px = inpaint_args.get("parser_affine_refine_translation_px")
        if expected_translation_px is not None and abs(float(expected_translation_px) - affine_refine_translation_px) > 1e-9:
            print(
                "inpaint_warning="
                + json.dumps(
                    {
                        "message": "parser affine-refinement translation range differs from checkpoint",
                        "checkpoint_translation_px": float(expected_translation_px),
                        "requested_translation_px": float(affine_refine_translation_px),
                    },
                    sort_keys=True,
                )
            )
        expected_scale = inpaint_args.get("parser_affine_refine_scale")
        if expected_scale is not None and abs(float(expected_scale) - affine_refine_scale) > 1e-9:
            raise ValueError(
                "Parser affine-refinement scale range does not match the uv_inpainting checkpoint: "
                f"checkpoint={expected_scale}, requested={affine_refine_scale}."
            )
        expected_outer_coverage = inpaint_args.get("parser_outer_uv_min_coverage")
        if expected_outer_coverage is not None and abs(
            float(expected_outer_coverage) - outer_uv_min_coverage
        ) > 1e-9:
            raise ValueError(
                "Parser outer UV coverage threshold does not match the uv_inpainting checkpoint: "
                f"checkpoint={expected_outer_coverage}, requested={outer_uv_min_coverage}."
            )
        expected_consensus = inpaint_args.get("parser_geometry_route_texel_consensus")
        if (
            expected_consensus is not None
            and bool(expected_consensus) != geometry_route_texel_consensus
        ):
            raise ValueError(
                "Parser routing mode does not match the uv_inpainting checkpoint: "
                f"checkpoint texel_consensus={bool(expected_consensus)}, "
                f"requested={geometry_route_texel_consensus}."
            )
        expected_color_aggregation = inpaint_args.get("parser_splat_color_aggregation")
        if expected_color_aggregation is not None and expected_color_aggregation != args.color_aggregation:
            raise ValueError(
                "Parser color aggregation does not match the uv_inpainting checkpoint: "
                f"checkpoint={expected_color_aggregation}, requested={args.color_aggregation}."
            )
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
