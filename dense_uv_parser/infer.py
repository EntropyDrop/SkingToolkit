import argparse
import inspect
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
from SkingToolkit.dense_uv_parser.foreground import (  # noqa: E402
    build_parser_input,
    save_flood_outputs,
)
from SkingToolkit.dense_uv_parser.semantic import attach_siglip_runtime  # noqa: E402
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
    estimate_top_left_flood_foreground,
    flat_uv_to_uv01,
    parse_views,
    prediction_uv01,
    splat_parser_predictions_to_uv_conditioning,
    surface_class_count,
)
from SkingToolkit.semantic_uv_reconstruction.dataset import finalize_minecraft_alpha, tensor_to_rgba_image, view_native_size  # noqa: E402
from SkingToolkit.semantic_uv_reconstruction.model import UVInpaintingNet  # noqa: E402
from SkingToolkit.semantic_uv_reconstruction.topology_model import TopologyAwareUVCompletionNet  # noqa: E402
from SkingToolkit.semantic_uv_reconstruction.topology import (  # noqa: E402
    FACE_COUNT,
    LAYER_COUNT,
    PART_COUNT,
    build_uv_topology,
    simple_symmetry_nearest_inpaint,
)
from SkingToolkit.semantic_uv_reconstruction.train import get_device  # noqa: E402
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
            # PyTorch's legacy nearest mode samples the top/left member of an
            # integer resize block.  Inputs are commonly rendered at exactly
            # 2x the mapping resolution, so that introduces a stable half-pixel
            # phase shift before otherwise-correct UV routing. nearest-exact
            # follows pixel-center semantics while retaining unblended colors.
            mode="nearest-exact",
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
    has_route_prior = "route_role_prior" in state_dict
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
        semantic_feature_dim=model_config.get("semantic_feature_dim", 0),
        semantic_channels=model_config.get("semantic_channels", 128),
        semantic_attention_heads=model_config.get("semantic_attention_heads", 4),
        semantic_layers=model_config.get("semantic_layers", 1),
        semantic_dropout=model_config.get("semantic_dropout", 0.05),
        predict_confidence=model_config.get(
            "predict_confidence",
            any(key.startswith("route_confidence.") for key in state_dict),
        ),
        route_role_spatial_prior=model_config.get(
            "route_role_spatial_prior", has_route_prior
        ),
        route_prior_height=model_config.get(
            "route_prior_height",
            state_dict["route_role_prior"].shape[-2] if has_route_prior else 32,
        ),
        route_prior_width=model_config.get(
            "route_prior_width",
            state_dict["route_role_prior"].shape[-1] if has_route_prior else 16,
        ),
        route_prior_logit_cap=model_config.get("route_prior_logit_cap", 1.5),
        route_prior_dropout=model_config.get("route_prior_dropout", 0.0),
    ).to(device)
    model.load_state_dict(state_dict)
    if model.semantic_feature_dim > 0:
        attach_siglip_runtime(
            model,
            model_config.get(
                "siglip_model",
                checkpoint_args.get(
                    "siglip_model", "google/siglip2-base-patch16-224"
                ),
            ),
            device,
            local_files_only=bool(
                checkpoint_args.get("siglip_local_files_only", False)
            ),
        )
    model.eval()
    return model, checkpoint_args


def load_inpaint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_args = checkpoint.get("args", {})
    model_config = checkpoint.get("model_config", {})
    model_type = model_config.get(
        "model_type", checkpoint_args.get("completion_model", "unet")
    )
    input_channels = model_config.get(
        "input_channels", checkpoint.get("input_channels", checkpoint_args.get("input_channels", 10))
    )
    if model_type == "topology_maskgit":
        model = TopologyAwareUVCompletionNet(
            input_channels=input_channels,
            hidden_channels=model_config.get(
                "hidden_channels", checkpoint_args.get("topology_channels", 128)
            ),
            layers=model_config.get("layers", checkpoint_args.get("topology_layers", 4)),
            attention_heads=model_config.get(
                "attention_heads", checkpoint_args.get("topology_attention_heads", 4)
            ),
            dropout=model_config.get("dropout", checkpoint_args.get("topology_dropout", 0.05)),
            preserve_known=model_config.get(
                "preserve_known", checkpoint_args.get("preserve_known", True)
            ),
            hard_lock_threshold=model_config.get(
                "hard_lock_threshold",
                checkpoint_args.get("topology_hard_lock_threshold", 0.85),
            ),
        ).to(device)
    elif model_type == "unet":
        model = UVInpaintingNet(
            input_channels=input_channels,
            base_channels=model_config.get(
                "base_channels", checkpoint_args.get("base_channels", 64)
            ),
            preserve_known=model_config.get(
                "preserve_known", checkpoint_args.get("preserve_known", True)
            ),
        ).to(device)
    else:
        raise ValueError(f"Unsupported inpaint model_type={model_type!r}.")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    checkpoint_args = dict(checkpoint_args)
    checkpoint_args["completion_model"] = model_type
    return model, checkpoint_args


def checkpoint_run_id(path):
    path = Path(path)
    return f"{path.parent.name}/{path.name}"


def generate_topology_completion(
    model,
    conditioning,
    steps,
    temperature,
    seed,
    palette_snap=True,
    palette_min_confidence=0.5,
    context_min_confidence=None,
    rgb_decode="mean",
):
    """Call new and legacy topology generators without source-version crashes."""
    parameters = inspect.signature(model.generate).parameters
    kwargs = {
        "steps": steps,
        "temperature": temperature,
        "seed": seed,
    }
    palette_supported = "palette_snap" in parameters
    if palette_supported:
        kwargs.update(
            palette_snap=palette_snap,
            palette_min_confidence=palette_min_confidence,
        )
        if "context_min_confidence" in parameters:
            kwargs["context_min_confidence"] = context_min_confidence
    elif palette_snap:
        print(
            "inpaint_warning="
            + json.dumps(
                {
                    "message": (
                        "palette snapping is unavailable because "
                        "semantic_uv_reconstruction/topology_model.py is older "
                        "than dense_uv_parser/infer.py; update both files"
                    ),
                    "palette_snap_applied": False,
                },
                sort_keys=True,
            )
        )
    if "rgb_decode" in parameters:
        kwargs["rgb_decode"] = rgb_decode
    return model.generate(conditioning, **kwargs)


def lock_completed_parser_evidence(completed, conditioning, confidence_threshold=0.0):
    """Copy trusted parser RGBA back after completion, including legacy generators."""
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1].")
    if completed.dim() != 4 or completed.shape[1:] != (4, 64, 64):
        raise ValueError(f"Expected completed Bx4x64x64 UV, got {tuple(completed.shape)}.")
    if conditioning.dim() != 4 or conditioning.shape[1] not in (10, 12):
        raise ValueError(
            "Expected 10- or 12-channel conditioning, "
            f"got {tuple(conditioning.shape)}."
        )
    if completed.shape[0] != conditioning.shape[0]:
        raise ValueError("Completed UV and conditioning batch sizes must match.")

    inner_rgba = conditioning[:, 0:4]
    inner_evidence = conditioning[:, 4:5] > 0.5
    outer_offset = 6 if conditioning.shape[1] == 12 else 5
    outer_rgba = conditioning[:, outer_offset : outer_offset + 4]
    outer_evidence = conditioning[:, outer_offset + 4 : outer_offset + 5] > 0.5
    if conditioning.shape[1] == 12:
        inner_evidence = inner_evidence & (
            conditioning[:, 5:6] >= float(confidence_threshold)
        )
        outer_evidence = outer_evidence & (
            conditioning[:, 11:12] >= float(confidence_threshold)
        )

    locked_rgba = torch.where(outer_evidence.expand_as(outer_rgba), outer_rgba, inner_rgba)
    locked_mask = inner_evidence | outer_evidence
    difference = (completed - locked_rgba).abs().amax(dim=1, keepdim=True)
    overwritten = locked_mask & (difference > (0.5 / 255.0))
    locked = torch.where(locked_mask.expand_as(completed), locked_rgba, completed)
    return locked, {
        "locked_evidence_texels": int(locked_mask.sum().item()),
        "model_overwrote_locked_texels": int(overwritten.sum().item()),
        "max_locked_rgba_error": (
            float(difference[locked_mask].max().item()) if locked_mask.any() else 0.0
        ),
    }


def propagate_completed_unknown_colors(
    completed,
    conditioning,
    min_confidence=0.75,
    context_min_confidence=None,
    context_alpha_rescue_mask=None,
):
    """Replace generated opaque RGB with stable nearby parser-observed colors."""
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1].")
    if context_min_confidence is None:
        context_min_confidence = min_confidence
    if not 0.0 <= context_min_confidence <= 1.0:
        raise ValueError("context_min_confidence must be in [0, 1].")
    if completed.dim() != 4 or completed.shape[1:] != (4, 64, 64):
        raise ValueError(f"Expected completed Bx4x64x64 UV, got {tuple(completed.shape)}.")
    if conditioning.dim() != 4 or conditioning.shape[1] not in (10, 12):
        raise ValueError("Expected 10- or 12-channel parser conditioning.")
    if context_alpha_rescue_mask is not None:
        if context_alpha_rescue_mask.shape != (
            completed.shape[0],
            1,
            64,
            64,
        ):
            raise ValueError(
                "context_alpha_rescue_mask must have shape Bx1x64x64."
            )

    topology = build_uv_topology()
    device = completed.device
    valid = topology.valid.reshape(-1).to(device=device)
    layer = topology.layer.reshape(-1).to(device=device)
    part = topology.part.reshape(-1).to(device=device)
    face = topology.face.reshape(-1).to(device=device)
    coordinates = topology.local_uv.reshape(-1, 2).to(device=device).float()

    flat = conditioning.flatten(2)
    inner_rgba = flat[:, 0:4].transpose(1, 2)
    inner_evidence = flat[:, 4] > 0.5
    outer_offset = 6 if conditioning.shape[1] == 12 else 5
    outer_rgba = flat[:, outer_offset : outer_offset + 4].transpose(1, 2)
    outer_evidence = flat[:, outer_offset + 4] > 0.5
    is_inner = (layer == 0).view(1, -1)
    observed = torch.where(is_inner.unsqueeze(-1), inner_rgba, outer_rgba)
    evidence = torch.where(is_inner, inner_evidence, outer_evidence) & valid.view(1, -1)
    if conditioning.shape[1] == 12:
        confidence = torch.where(
            is_inner,
            flat[:, 5],
            flat[:, 11],
        )
    else:
        confidence = evidence.to(dtype=completed.dtype)

    # Rejected-but-plausible pixels need a lower threshold when copied back at
    # their exact UV coordinate.  Keep them out of the shared propagation
    # palette unless they also pass the stricter palette threshold, otherwise
    # one uncertain pixel can recolor unrelated generated texels.
    context_source = (~evidence) & (
        confidence >= float(context_min_confidence)
    )
    palette_context_source = context_source & (
        confidence >= float(min_confidence)
    )
    result = completed.flatten(2).transpose(1, 2).clone()
    model_generated = (
        (~evidence) & valid.view(1, -1) & (result[..., 3] > 0.5)
    )
    if context_alpha_rescue_mask is None:
        alpha_rescue_mask = torch.zeros_like(context_source)
    else:
        alpha_rescue_mask = (
            context_alpha_rescue_mask.flatten(2)[:, 0]
            .to(device=result.device)
            .bool()
        )
    alpha_rescue_eligible = (
        context_source
        & alpha_rescue_mask
        & valid.view(1, -1)
        & (observed[..., 3] > 0.5)
    )
    alpha_restored = alpha_rescue_eligible & ~model_generated
    result[..., 3] = torch.where(
        alpha_rescue_eligible,
        observed[..., 3].to(dtype=result.dtype),
        result[..., 3],
    )
    generated = (~evidence) & valid.view(1, -1) & (result[..., 3] > 0.5)
    opaque_source = (evidence | palette_context_source) & (observed[..., 3] > 0.5)
    strong_source = opaque_source & (confidence >= float(min_confidence))
    direct_context = context_source & generated & (observed[..., 3] > 0.5)
    result[..., :3] = torch.where(
        direct_context.unsqueeze(-1),
        observed[..., :3].to(dtype=result.dtype),
        result[..., :3],
    )
    generated_for_propagation = generated & ~direct_context

    def stable_indices(mask, colors):
        indices = mask.nonzero(as_tuple=False).flatten()
        if indices.numel() < 2:
            return None, indices
        rgb8 = colors[indices, :3].clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
        _, inverse, counts = torch.unique(
            rgb8,
            dim=0,
            return_inverse=True,
            return_counts=True,
        )
        stable = counts[inverse] >= 2
        return (indices[stable] if stable.any() else None), indices

    recolored = 0
    for batch_index in range(result.shape[0]):
        strong = strong_source[batch_index]
        fallback = opaque_source[batch_index]
        if not fallback.any():
            continue
        colors = observed[batch_index]
        for part_index in range(PART_COUNT):
            part_mask = part == part_index
            for layer_index in range(LAYER_COUNT):
                group_target = (
                    generated_for_propagation[batch_index]
                    & part_mask
                    & (layer == layer_index)
                )
                if not group_target.any():
                    continue
                for face_index in range(FACE_COUNT):
                    target = group_target & (face == face_index)
                    if not target.any():
                        continue
                    candidates = (
                        strong & part_mask & (layer == layer_index) & (face == face_index),
                        fallback & part_mask & (layer == layer_index) & (face == face_index),
                        strong & part_mask & (layer == layer_index),
                        fallback & part_mask & (layer == layer_index),
                        strong & part_mask,
                        fallback & part_mask,
                        strong,
                        fallback,
                    )
                    selected_reference = None
                    first_nonempty = None
                    for candidate in candidates:
                        stable, indices = stable_indices(candidate, colors)
                        if first_nonempty is None and indices.numel() > 0:
                            first_nonempty = indices
                        if stable is not None and stable.numel() > 0:
                            selected_reference = stable
                            break
                    if selected_reference is None:
                        selected_reference = first_nonempty
                    if selected_reference is None or selected_reference.numel() == 0:
                        continue
                    target_indices = target.nonzero(as_tuple=False).flatten()
                    color_distance = torch.cdist(
                        result[batch_index, target_indices, :3].float(),
                        colors[selected_reference, :3].float(),
                    )
                    spatial_distance = torch.cdist(
                        coordinates[target_indices],
                        coordinates[selected_reference],
                    )
                    nearest = (
                        color_distance + 0.05 * spatial_distance
                    ).argmin(dim=1)
                    source_indices = selected_reference[nearest]
                    result[batch_index, target_indices, :3] = colors[source_indices, :3]
                    recolored += int(target_indices.numel())

    result = result.transpose(1, 2).reshape_as(completed)
    return result, {
        "model_generated_opaque_texels": int(model_generated.sum().item()),
        "generated_opaque_texels": int(generated.sum().item()),
        "available_context_texels": int(
            (context_source & valid.view(1, -1) & (observed[..., 3] > 0.5))
            .sum()
            .item()
        ),
        "palette_context_texels": int(
            (
                palette_context_source
                & valid.view(1, -1)
                & (observed[..., 3] > 0.5)
            )
            .sum()
            .item()
        ),
        "context_alpha_rescue_eligible_texels": int(
            alpha_rescue_eligible.sum().item()
        ),
        "context_alpha_restored_texels": int(alpha_restored.sum().item()),
        "direct_context_texels": int(direct_context.sum().item()),
        "model_context_alpha_rejected_texels": int(
            (
                context_source
                & valid.view(1, -1)
                & (observed[..., 3] > 0.5)
                & ~model_generated
            )
            .sum()
            .item()
        ),
        "context_alpha_rejected_texels": int(
            (
                context_source
                & valid.view(1, -1)
                & (observed[..., 3] > 0.5)
                & ~generated
            )
            .sum()
            .item()
        ),
        "topology_color_propagated_texels": recolored,
        "uncolored_generated_texels": (
            int(generated.sum().item())
            - int(direct_context.sum().item())
            - recolored
        ),
    }


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
    outer_offset = 6 if conditioning.shape[1] == 12 else 5
    outer_rgb = conditioning[:, outer_offset : outer_offset + 3]
    preview = torch.cat([inner_rgb, outer_rgb], dim=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(preview.clamp(0.0, 1.0), output_path, nrow=conditioning.shape[0])


def save_parser_uv(
    conditioning,
    output_path,
    alpha_threshold=0.5,
    enforce_base_alpha=False,
):
    """Save the partial parser atlas with unknown texels left transparent."""
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
    # A partial parser atlas is a diagnostic artifact, not a complete skin.
    # Clear placeholder RGB under transparent texels so viewers that mishandle
    # base-layer alpha cannot display the conditioning background as predicted
    # skin color.
    opaque = parser_uv[3:4] > 0.5
    parser_uv[:3] = torch.where(
        opaque.expand_as(parser_uv[:3]),
        parser_uv[:3],
        torch.zeros_like(parser_uv[:3]),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_rgba_image(parser_uv.detach().cpu()).save(output_path)
    print(
        "parser_uv_stats="
        + json.dumps(
            {
                "opaque_pixels": int(opaque.sum().item()),
                "transparent_pixels": int((~opaque).sum().item()),
            },
            sort_keys=True,
        )
    )
    print(f"Saved parser_partial_uv={output_path}")


def save_simple_inpaint_uv(conditioning, output_path, alpha_threshold=0.5):
    """Repair inner UV holes while preserving the parser's outer layer."""
    parser_uv = conditioning_to_pred_uv(conditioning)
    if parser_uv.dim() != 4 or parser_uv.shape[0] != 1:
        raise ValueError(
            "Simple parser UV inpainting requires exactly one conditioning "
            f"sample, got {tuple(parser_uv.shape)}."
        )
    repaired, stats = simple_symmetry_nearest_inpaint(
        parser_uv[0], alpha_threshold=alpha_threshold
    )
    repaired = finalize_minecraft_alpha(
        repaired,
        alpha_threshold=alpha_threshold,
        enforce_base_alpha=False,
    )
    opaque = repaired[3:4] > 0.5
    repaired[:3] = torch.where(
        opaque.expand_as(repaired[:3]),
        repaired[:3],
        torch.zeros_like(repaired[:3]),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_rgba_image(repaired.detach().cpu()).save(output_path)
    print("simple_inpaint_stats=" + json.dumps(stats, sort_keys=True))
    print(f"Saved parser_simple_inpaint_uv={output_path}")


def _raw_debug_foreground(outputs, routing, fg_threshold):
    """Mask raw semantic heads with observed input foreground when available.

    The routed foreground is deliberately stricter: it excludes pixels rejected
    by route confidence, margin, and UV coverage filters.  Using it for a raw
    head preview makes those routing rejections look like holes in the semantic
    prediction, even though the head produced a class at every foreground pixel.
    """
    model_foreground = (
        torch.sigmoid(outputs["foreground"])[:, 0] > fg_threshold
    )
    if routing is None:
        return model_foreground
    return routing.get(
        "observed_foreground",
        routing.get("raw_foreground", model_foreground),
    )


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
    color_source_output=None,
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
    raw_fg = _raw_debug_foreground(outputs, routing, fg_threshold)
    pred_layer_values = routing["layer"] if routing is not None else raw_layer_values
    pred_layer = torch.where(
        pred_fg,
        pred_layer_values,
        torch.full_like(outputs["layer"].argmax(dim=1), IGNORE_INDEX),
    )
    raw_face_values = outputs["face"].argmax(dim=1) if "face" in outputs else routing["face"]
    raw_face = torch.where(
        raw_fg,
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
            raw_fg,
            raw_layer_face_values,
            torch.full_like(raw_layer_face_values, IGNORE_INDEX),
        )
    else:
        raw_layer_values_for_debug = (
            routing["layer"]
            if routing is not None and outputs["layer"].shape[1] == 3
            else raw_layer_values
        )
        raw_layer = torch.where(
            raw_fg,
            raw_layer_values_for_debug,
            torch.full_like(raw_layer_values_for_debug, IGNORE_INDEX),
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
        or color_source_output is not None
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
        color_source_mask = (
            routing.get("color_foreground", pred_fg)
            if routing is not None
            else pred_fg
        ).unsqueeze(1)
        color_source_cutout = torch.where(
            color_source_mask, rgb, bg.expand_as(rgb)
        )

        def save_cutout(cutout, path):
            if path is None:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            save_image(cutout.clamp(0.0, 1.0).detach().cpu(), path, nrow=view_count)

        save_cutout(inner_cutout, inner_cutout_output)
        save_cutout(outer_cutout, outer_cutout_output)
        save_cutout(secondary_cutout, secondary_cutout_output)
        save_cutout(color_source_cutout, color_source_output)

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
    parser.add_argument(
        "--foreground_method",
        choices=["flood", "legacy"],
        default="flood",
        help=(
            "Background removal before dense parsing. 'flood' uses the top-left "
            "pixel as a connected-color seed; 'legacy' leaves removal to the "
            "former routing fallback."
        ),
    )
    parser.add_argument(
        "--foreground_flood_tolerance",
        type=float,
        default=0.03,
        help="Maximum per-channel RGB distance from the top-left flood seed.",
    )
    parser.add_argument(
        "--foreground_parser_background",
        choices=["adaptive", "neutral"],
        default="adaptive",
        help="Solid background used for the masked RGB passed to dense parser.",
    )
    parser.add_argument(
        "--foreground_probability_output",
        default="outputs/foreground_probability.png",
        help=(
            "Grayscale foreground score produced before dense parsing; this is "
            "a binary mask when top-left flood fill is selected."
        ),
    )
    parser.add_argument(
        "--foreground_mask_output",
        default="outputs/foreground_mask.png",
        help="Thresholded fixed-view foreground mask used by dense parsing.",
    )
    parser.add_argument(
        "--foreground_raw_mask_output",
        default="outputs/foreground_mask_raw.png",
        help="Binary foreground mask produced directly by top-left flood fill.",
    )
    parser.add_argument(
        "--foreground_cutout_output",
        default="outputs/foreground_cutout.png",
        help="Input views with the predicted background removed.",
    )
    parser.add_argument(
        "--foreground_parser_input_output",
        default="outputs/foreground_parser_input.png",
        help="Exact adaptive-background RGB images passed to dense parser.",
    )
    parser.add_argument("--inpaint_checkpoint", default=None, help="Optional semantic_uv_reconstruction checkpoint used to inpaint final skin.")
    parser.add_argument("--inpaint_steps", type=int, default=4, help="Masked-generation steps for topology checkpoints.")
    parser.add_argument("--inpaint_temperature", type=float, default=0.0, help="0 uses deterministic decoding; positive values sample unknown texels.")
    parser.add_argument("--inpaint_seed", type=int, default=1234)
    parser.add_argument(
        "--inpaint_rgb_decode",
        choices=["mean", "argmax"],
        default="mean",
        help="Deterministic RGB decoder. mean matches the continuous training objective; argmax is legacy behavior.",
    )
    parser.add_argument(
        "--inpaint_palette_snap",
        dest="inpaint_palette_snap",
        action="store_true",
        default=True,
        help="Snap generated RGB to observed per-part/layer character palettes.",
    )
    parser.add_argument(
        "--no_inpaint_palette_snap",
        dest="inpaint_palette_snap",
        action="store_false",
    )
    parser.add_argument("--inpaint_palette_min_confidence", type=float, default=0.5)
    parser.add_argument(
        "--inpaint_context_min_confidence",
        type=float,
        default=0.35,
        help=(
            "Minimum confidence for copying rejected context at the same UV texel. "
            "This does not lower the shared palette threshold."
        ),
    )
    parser.add_argument(
        "--inpaint_evidence_lock_threshold",
        type=float,
        default=0.0,
        help="Lock parser evidence at or above this confidence; zero preserves every routed texel.",
    )
    parser.add_argument("--output", default=None, help="Final RGBA UV PNG path; requires --inpaint_checkpoint.")
    parser.add_argument("--conditioning_output", default=None, help="Optional preview image for parser-splatted conditioning.")
    parser.add_argument(
        "--parser_uv_output",
        default=None,
        help="Optional preliminary RGBA skin merged directly from parser conditioning.",
    )
    parser.add_argument(
        "--simple_inpaint_output",
        default=None,
        help=(
            "Optional deterministic inner-layer repair: use known left/right "
            "symmetry first, then the nearest known texel in 3D character "
            "space from the same body part. Traverse each face top-down and "
            "horizontal-centre-out; preserve the outer layer unchanged."
        ),
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
    parser.add_argument(
        "--color_source_output",
        default=None,
        help="Original RGB pixels permitted to contribute colors to parser UV output.",
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
    parser.add_argument(
        "--background_color_tolerance",
        type=float,
        default=48.0 / 255.0,
        help="RGB distance used to reject solid-background and antialiased edge pixels.",
    )
    parser.add_argument(
        "--color_background_tolerance",
        type=float,
        default=8.0 / 255.0,
        help=(
            "Reject background-like RGB candidates only on the foreground "
            "boundary before inverse UV color sampling."
        ),
    )
    parser.add_argument(
        "--color_foreground_inset",
        type=int,
        default=1,
        help="Foreground boundary width demoted during texel-center color selection.",
    )
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
        "--outer_uv_min_source_pixels",
        type=int,
        default=15,
        help="Minimum routed source pixels required to keep an outer UV texel.",
    )
    parser.add_argument(
        "--outer_geometry_rescue",
        dest="outer_geometry_rescue",
        action="store_true",
        default=True,
        help="Relax outer gates only for UV texels proven by outer-only silhouette or an exact secondary slot.",
    )
    parser.add_argument(
        "--no_outer_geometry_rescue",
        dest="outer_geometry_rescue",
        action="store_false",
    )
    parser.add_argument(
        "--outer_semantic_rescue",
        dest="outer_semantic_rescue",
        action="store_true",
        default=True,
        help="Relax outer gates only on parts whose global semantic heads predict a substantial outer layer.",
    )
    parser.add_argument(
        "--no_outer_semantic_rescue",
        dest="outer_semantic_rescue",
        action="store_false",
    )
    parser.add_argument("--outer_semantic_presence_threshold", type=float, default=0.80)
    parser.add_argument("--outer_semantic_coverage_threshold", type=float, default=0.20)
    parser.add_argument("--outer_rescue_confidence_threshold", type=float, default=0.60)
    parser.add_argument("--outer_rescue_margin_threshold", type=float, default=0.25)
    parser.add_argument("--outer_rescue_min_coverage", type=float, default=0.10)
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
        default="grid_mode",
        help="How colors inside each fitted layer/UV grid cell are selected.",
    )
    parser.add_argument(
        "--allow_semantic_fallback",
        action="store_true",
        help="Keep pixels whose strict semantic routing had no valid candidate.",
    )
    parser.add_argument(
        "--rejected_context",
        dest="include_rejected_context",
        action="store_true",
        default=True,
        help="Pass moderately confident rejected pixels to topology completion as unlocked RGB context.",
    )
    parser.add_argument(
        "--no_rejected_context",
        dest="include_rejected_context",
        action="store_false",
    )
    parser.add_argument("--rejected_context_confidence_threshold", type=float, default=0.35)
    parser.add_argument("--rejected_context_margin_threshold", type=float, default=0.10)
    parser.add_argument(
        "--inpaint_context_alpha_rescue",
        dest="inpaint_context_alpha_rescue",
        action="store_true",
        default=False,
        help=(
            "Restore opacity only for rejected outer context backed by "
            "part semantics or outer geometry."
        ),
    )
    parser.add_argument(
        "--no_inpaint_context_alpha_rescue",
        dest="inpaint_context_alpha_rescue",
        action="store_false",
    )
    parser.add_argument(
        "--inpaint_context_alpha_min_confidence", type=float, default=0.50
    )
    parser.add_argument(
        "--inpaint_context_alpha_min_margin", type=float, default=0.10
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
    if args.inpaint_steps < 1 or args.inpaint_temperature < 0.0:
        raise ValueError("Inpaint generation requires positive steps and non-negative temperature.")
    if not 0.0 <= args.background_color_tolerance <= 1.0:
        raise ValueError("--background_color_tolerance must be in [0, 1].")
    if not 0.0 <= args.color_background_tolerance <= 1.0:
        raise ValueError("--color_background_tolerance must be in [0, 1].")
    if args.color_foreground_inset < 0:
        raise ValueError("--color_foreground_inset must be non-negative.")
    if args.outer_uv_min_source_pixels < 1:
        raise ValueError("--outer_uv_min_source_pixels must be positive.")
    if not 0.0 <= args.foreground_flood_tolerance <= 1.0:
        raise ValueError("--foreground_flood_tolerance must be in [0, 1].")
    if not 0.0 <= args.inpaint_palette_min_confidence <= 1.0:
        raise ValueError("--inpaint_palette_min_confidence must be in [0, 1].")
    if not 0.0 <= args.inpaint_context_min_confidence <= 1.0:
        raise ValueError("--inpaint_context_min_confidence must be in [0, 1].")
    if not 0.0 <= args.inpaint_context_alpha_min_confidence <= 1.0:
        raise ValueError(
            "--inpaint_context_alpha_min_confidence must be in [0, 1]."
        )
    if not 0.0 <= args.inpaint_context_alpha_min_margin <= 1.0:
        raise ValueError("--inpaint_context_alpha_min_margin must be in [0, 1].")
    if not 0.0 <= args.inpaint_evidence_lock_threshold <= 1.0:
        raise ValueError("--inpaint_evidence_lock_threshold must be in [0, 1].")
    if args.output and not args.inpaint_checkpoint:
        raise ValueError("--output requires --inpaint_checkpoint.")
    if not any(
        (
            args.output,
            args.conditioning_output,
            args.parser_uv_output,
            args.simple_inpaint_output,
            args.debug_output,
            args.overlay_output,
            args.inner_cutout_output,
            args.outer_cutout_output,
            args.secondary_cutout_output,
            args.color_source_output,
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
            "Provide --output, --conditioning_output, --parser_uv_output, --simple_inpaint_output, "
            "--debug_output, --overlay_output, "
            "--inner_cutout_output, --outer_cutout_output, --secondary_cutout_output, "
            "and/or --color_source_output."
        )

    device = get_device(args.device)
    parser_model, parser_args = load_parser(args.parser_checkpoint, device)
    inpaint_model = None
    inpaint_args = None
    if args.output:
        inpaint_model, inpaint_args = load_inpaint(args.inpaint_checkpoint, device)
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
        observed_foreground = None
        parser_rendered = rendered
        if args.foreground_method == "flood":
            observed_foreground = estimate_top_left_flood_foreground(
                rendered,
                color_tolerance=args.foreground_flood_tolerance,
            )
            foreground_log = {
                "method": "top_left_flood",
                "seed_rgb": [
                    [int(round(channel * 255.0)) for channel in color]
                    for color in rendered[:, :3, 0, 0].detach().cpu().tolist()
                ],
                "tolerance": round(float(args.foreground_flood_tolerance), 6),
            }
            observed_foreground = save_flood_outputs(
                rendered,
                observed_foreground,
                view_count=len(views),
                probability_output=args.foreground_probability_output,
                raw_mask_output=args.foreground_raw_mask_output,
                mask_output=args.foreground_mask_output,
                cutout_output=args.foreground_cutout_output,
            )
            (
                parser_rendered,
                parser_background_rgb,
                parser_background_indices,
            ) = build_parser_input(
                rendered,
                observed_foreground,
                bg_color=bg_color,
                background_mode=args.foreground_parser_background,
                return_background=True,
            )
            if args.foreground_parser_input_output:
                parser_input_path = Path(args.foreground_parser_input_output)
                parser_input_path.parent.mkdir(parents=True, exist_ok=True)
                save_image(
                    parser_rendered[:, :3].detach().cpu(),
                    parser_input_path,
                    nrow=len(views),
                )
            foreground_log.update(
                {
                    "kept_pixels": int(observed_foreground.sum().item()),
                    "rejected_background_pixels": int(
                        (~observed_foreground).sum().item()
                    ),
                    "parser_background_mode": args.foreground_parser_background,
                    "parser_background_rgb": [
                        [int(round(channel * 255.0)) for channel in color]
                        for color in parser_background_rgb.detach().cpu().tolist()
                    ],
                    "parser_background_indices": parser_background_indices,
                }
            )
            print(
                "foreground_filter="
                + json.dumps(foreground_log, sort_keys=True)
            )
        # Background removal affects parser features, while routing/splatting
        # below still reads exact foreground colors from the original input.
        outputs = parser_model(parser_rendered, view_ids=view_ids)
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
            outer_uv_min_source_pixels=args.outer_uv_min_source_pixels,
            outer_geometry_rescue=args.outer_geometry_rescue,
            outer_semantic_rescue=args.outer_semantic_rescue,
            outer_semantic_presence_threshold=args.outer_semantic_presence_threshold,
            outer_semantic_coverage_threshold=args.outer_semantic_coverage_threshold,
            outer_rescue_confidence_threshold=args.outer_rescue_confidence_threshold,
            outer_rescue_margin_threshold=args.outer_rescue_margin_threshold,
            outer_rescue_min_coverage=args.outer_rescue_min_coverage,
            color_aggregation=args.color_aggregation,
            geometry_route_texel_consensus=geometry_route_texel_consensus,
            observed_foreground=observed_foreground,
            background_color_tolerance=args.background_color_tolerance,
            color_background_tolerance=args.color_background_tolerance,
            color_foreground_inset=args.color_foreground_inset,
            reject_semantic_fallback=not args.allow_semantic_fallback,
            include_rejected_context=args.include_rejected_context,
            rejected_context_confidence_threshold=args.rejected_context_confidence_threshold,
            rejected_context_margin_threshold=args.rejected_context_margin_threshold,
            rejected_context_alpha_confidence_threshold=(
                args.inpaint_context_alpha_min_confidence
            ),
            rejected_context_alpha_margin_threshold=(
                args.inpaint_context_alpha_min_margin
            ),
            include_confidence=(
                getattr(inpaint_model, "input_channels", 10) == 12
            ),
            return_details=True,
        )

    routing = routing_details.get("routing")
    if routing is not None:
        observed_routing_foreground = routing.get(
            "observed_foreground", routing["raw_foreground"]
        )
        observed_count = int(observed_routing_foreground.sum().item())
        unrouted_observed_count = int(
            (
                observed_routing_foreground
                & ~routing["raw_foreground"]
            ).sum().item()
        )
        raw_count = int(routing["raw_foreground"].sum().item())
        rejected_count = int(routing["rejected"].sum().item())
        kept = routing["foreground"]
        kept_inner_count = int((kept & (routing["layer"] == 0)).sum().item())
        kept_outer_count = int((kept & (routing["layer"] == 1)).sum().item())
        raw_inner = routing["raw_foreground"] & (routing["layer"] == 0)
        raw_outer = routing["raw_foreground"] & (routing["layer"] == 1)
        rejected_inner = routing["rejected"] & (routing["layer"] == 0)
        rejected_outer = routing["rejected"] & (routing["layer"] == 1)
        outer_confidence = routing["confidence"][raw_outer].float()
        outer_margin_ratio = routing["confidence_margin_ratio"][raw_outer].float()

        def quantile_or_zero(values, quantile):
            return (
                float(torch.quantile(values, quantile).item())
                if values.numel() > 0
                else 0.0
            )
        coverage_rejected_outer = raw_outer & (
            routing.get("outer_uv_coverage", torch.ones_like(routing["confidence"]))
            < routing.get(
                "outer_required_coverage",
                torch.full_like(routing["confidence"], outer_uv_min_coverage),
            )
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
        color_rejected_count = int(
            routing.get("color_rejected", torch.zeros_like(raw_outer)).sum().item()
        )
        outer_source_rejected_count = int(
            routing.get(
                "outer_source_rejected", torch.zeros_like(raw_outer)
            ).sum().item()
        )
        outer_geometry_supported_count = int(
            routing.get("outer_geometry_supported", torch.zeros_like(raw_outer)).sum().item()
        )
        outer_geometry_rescued_count = int(
            routing.get("outer_geometry_rescued", torch.zeros_like(raw_outer)).sum().item()
        )
        outer_semantic_supported_count = int(
            routing.get("outer_semantic_supported", torch.zeros_like(raw_outer)).sum().item()
        )
        outer_semantic_rescued_count = int(
            routing.get("outer_semantic_rescued", torch.zeros_like(raw_outer)).sum().item()
        )
        rejected_context_count = int(
            routing.get("rejected_context", torch.zeros_like(raw_outer)).sum().item()
        )
        rejected_context_alpha_supported_count = int(
            routing.get(
                "rejected_context_alpha_supported",
                torch.zeros_like(raw_outer),
            ).sum().item()
        )
        outer_geometry_supported_rejected_count = int(
            (
                routing.get("outer_geometry_supported", torch.zeros_like(raw_outer))
                & routing["rejected"]
            ).sum().item()
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
                    "observed_foreground_pixels": observed_count,
                    "unrouted_observed_pixels": unrouted_observed_count,
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
                    "outer_confidence_p50": round(
                        quantile_or_zero(outer_confidence, 0.50), 6
                    ),
                    "outer_confidence_p75": round(
                        quantile_or_zero(outer_confidence, 0.75), 6
                    ),
                    "outer_confidence_p90": round(
                        quantile_or_zero(outer_confidence, 0.90), 6
                    ),
                    "outer_margin_p50": round(
                        quantile_or_zero(outer_margin_ratio, 0.50), 6
                    ),
                    "outer_margin_p75": round(
                        quantile_or_zero(outer_margin_ratio, 0.75), 6
                    ),
                    "outer_coverage_rejected_pixels": int(coverage_rejected_outer.sum().item()),
                    "outer_source_rejected_pixels": outer_source_rejected_count,
                    "outer_uv_min_source_pixels": int(
                        args.outer_uv_min_source_pixels
                    ),
                    "outer_geometry_supported_pixels": outer_geometry_supported_count,
                    "outer_geometry_rescued_pixels": outer_geometry_rescued_count,
                    "outer_geometry_supported_rejected_pixels": outer_geometry_supported_rejected_count,
                    "outer_semantic_supported_pixels": outer_semantic_supported_count,
                    "outer_semantic_rescued_pixels": outer_semantic_rescued_count,
                    "rejected_context_pixels": rejected_context_count,
                    "rejected_context_alpha_supported_pixels": (
                        rejected_context_alpha_supported_count
                    ),
                    "background_rejected_pixels": background_rejected_count,
                    "background_color_tolerance": round(
                        float(args.background_color_tolerance), 6
                    ),
                    "color_source_rejected_pixels": color_rejected_count,
                    "color_background_tolerance": round(
                        float(args.color_background_tolerance), 6
                    ),
                    "color_foreground_inset": int(args.color_foreground_inset),
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
            args.color_source_output,
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
            color_source_output=(
                Path(args.color_source_output) if args.color_source_output else None
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
            enforce_base_alpha=False,
        )

    if args.simple_inpaint_output:
        save_simple_inpaint_uv(
            conditioning.detach().cpu(),
            Path(args.simple_inpaint_output),
            alpha_threshold=args.alpha_threshold,
        )

    if args.output:
        print(
            "inpaint_config="
            + json.dumps(
                {
                    "checkpoint": checkpoint_run_id(args.inpaint_checkpoint),
                    "completion_model": inpaint_args.get("completion_model", "unet"),
                    "preserve_known": bool(inpaint_args.get("preserve_known", True)),
                    "generation_steps": args.inpaint_steps,
                    "generation_temperature": args.inpaint_temperature,
                    "rgb_decode": args.inpaint_rgb_decode,
                    "palette_snap": args.inpaint_palette_snap,
                    "palette_min_confidence": args.inpaint_palette_min_confidence,
                    "context_min_confidence": args.inpaint_context_min_confidence,
                    "context_alpha_rescue": args.inpaint_context_alpha_rescue,
                    "context_alpha_min_confidence": (
                        args.inpaint_context_alpha_min_confidence
                    ),
                    "context_alpha_min_margin": (
                        args.inpaint_context_alpha_min_margin
                    ),
                    "evidence_lock_threshold": args.inpaint_evidence_lock_threshold,
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
                "The semantic_uv_reconstruction checkpoint was trained with a different parser: "
                f"expected {checkpoint_run_id(expected_parser)}, got {checkpoint_run_id(args.parser_checkpoint)}."
            )
        expected_refine = inpaint_args.get("parser_affine_refine")
        if expected_refine is not None and bool(expected_refine) != affine_refine:
            raise ValueError(
                "Parser affine-refinement setting does not match the semantic_uv_reconstruction checkpoint: "
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
                "Parser affine-refinement scale range does not match the semantic_uv_reconstruction checkpoint: "
                f"checkpoint={expected_scale}, requested={affine_refine_scale}."
            )
        expected_outer_coverage = inpaint_args.get("parser_outer_uv_min_coverage")
        if expected_outer_coverage is not None and abs(
            float(expected_outer_coverage) - outer_uv_min_coverage
        ) > 1e-9:
            print(
                "inpaint_warning="
                + json.dumps(
                    {
                        "message": "using stricter inference outer-coverage filtering",
                        "checkpoint_outer_uv_min_coverage": float(expected_outer_coverage),
                        "requested_outer_uv_min_coverage": float(outer_uv_min_coverage),
                    },
                    sort_keys=True,
                )
            )
        expected_outer_source_pixels = inpaint_args.get(
            "parser_outer_uv_min_source_pixels"
        )
        if (
            expected_outer_source_pixels is not None
            and int(expected_outer_source_pixels)
            != int(args.outer_uv_min_source_pixels)
        ):
            print(
                "inpaint_warning="
                + json.dumps(
                    {
                        "message": "parser outer source-pixel filter differs from checkpoint",
                        "checkpoint_outer_uv_min_source_pixels": int(
                            expected_outer_source_pixels
                        ),
                        "requested_outer_uv_min_source_pixels": int(
                            args.outer_uv_min_source_pixels
                        ),
                    },
                    sort_keys=True,
                )
            )
        expected_consensus = inpaint_args.get("parser_geometry_route_texel_consensus")
        if (
            expected_consensus is not None
            and bool(expected_consensus) != geometry_route_texel_consensus
        ):
            print(
                "inpaint_warning="
                + json.dumps(
                    {
                        "message": "using inference-time projected-texel consensus",
                        "checkpoint_texel_consensus": bool(expected_consensus),
                        "requested_texel_consensus": bool(geometry_route_texel_consensus),
                    },
                    sort_keys=True,
                )
            )
        expected_color_aggregation = inpaint_args.get("parser_splat_color_aggregation")
        if expected_color_aggregation is not None and expected_color_aggregation != args.color_aggregation:
            print(
                "inpaint_warning="
                + json.dumps(
                    {
                        "message": "using inference-time parser color aggregation",
                        "checkpoint_color_aggregation": expected_color_aggregation,
                        "requested_color_aggregation": args.color_aggregation,
                    },
                    sort_keys=True,
                )
            )
        with torch.no_grad():
            if hasattr(inpaint_model, "hard_lock_threshold"):
                inpaint_model.hard_lock_threshold = float(
                    args.inpaint_evidence_lock_threshold
                )
            if hasattr(inpaint_model, "generate"):
                completed = generate_topology_completion(
                    inpaint_model,
                    conditioning,
                    steps=args.inpaint_steps,
                    temperature=args.inpaint_temperature,
                    seed=args.inpaint_seed,
                    rgb_decode=args.inpaint_rgb_decode,
                    # Final color propagation is applied below for both current
                    # and legacy generator signatures.
                    palette_snap=False,
                    palette_min_confidence=args.inpaint_palette_min_confidence,
                    context_min_confidence=args.inpaint_context_min_confidence,
                )[0]
            else:
                completed = inpaint_model(conditioning)[0]
            if args.inpaint_palette_snap:
                completed_batch, color_stats = propagate_completed_unknown_colors(
                    completed.unsqueeze(0),
                    conditioning,
                    min_confidence=args.inpaint_palette_min_confidence,
                    context_min_confidence=args.inpaint_context_min_confidence,
                    context_alpha_rescue_mask=(
                        routing_details.get("context_alpha_rescue_uv")
                        if args.inpaint_context_alpha_rescue
                        else None
                    ),
                )
                completed = completed_batch[0]
                print(
                    "inpaint_color_propagation="
                    + json.dumps(color_stats, sort_keys=True)
                )
            completed_batch, lock_stats = lock_completed_parser_evidence(
                completed.unsqueeze(0),
                conditioning,
                confidence_threshold=args.inpaint_evidence_lock_threshold,
            )
            completed = completed_batch[0]
            print("inpaint_evidence_lock=" + json.dumps(lock_stats, sort_keys=True))
            pred_uv = finalize_minecraft_alpha(
                completed,
                alpha_threshold=args.alpha_threshold,
                enforce_base_alpha=not args.no_enforce_base_alpha,
            )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tensor_to_rgba_image(pred_uv.detach().cpu()).save(output_path)
        print(f"Saved completed_uv={output_path}")


if __name__ == "__main__":
    main()
