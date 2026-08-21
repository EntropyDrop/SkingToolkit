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
from SkingToolkit.dense_uv_parser.foreground import (  # noqa: E402
    build_parser_input,
    save_flood_outputs,
)
from SkingToolkit.dense_uv_parser.inference_config import (  # noqa: E402
    PRODUCTION_PREPROCESSING_DEFAULTS,
    PRODUCTION_SPLAT_DEFAULTS,
)
from SkingToolkit.dense_uv_parser.semantic import attach_semantic_runtime  # noqa: E402
from SkingToolkit.dense_uv_parser.semantic_targets import (  # noqa: E402
    head_outer_face_values_to_uv,
)
from SkingToolkit.dense_uv_parser.differentiable_hypothesis_refiner import (  # noqa: E402
    refine_uv_by_analysis_by_synthesis,
)
from SkingToolkit.dense_uv_parser.runtime import get_device  # noqa: E402
from SkingToolkit.dense_uv_parser.simple_inpainting import (  # noqa: E402
    build_basic_minecraft_metadata,
    simple_symmetry_nearest_inpaint,
)
from SkingToolkit.dense_uv_parser.uv_layout import (  # noqa: E402
    UV_SIZE,
    finalize_minecraft_alpha,
    tensor_to_rgba_image,
    view_native_size,
)
from SkingToolkit.dense_uv_parser.utils import (  # noqa: E402
    FACE_PALETTE,
    IGNORE_INDEX,
    LAYER_FACE_PALETTE,
    LAYER_PALETTE,
    PART_PALETTE,
    ROUTE_ROLE_PALETTE,
    SPLAT_COLOR_AGGREGATIONS,
    aggregate_direct_inner_values_by_view,
    aggregate_direct_outer_values_by_view,
    attach_projected_head_outer_structure,
    attach_projected_outer_uv_occupancy,
    build_static_surface_routing,
    canonicalize_parser_outputs,
    combine_layer_face,
    build_geometry_grid_debug,
    fill_geometry_grid_debug,
    overlay_geometry_grid_debug,
    outer_uv_topology_hysteresis,
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


def project_head_eye_semantic_outer_probability(
    outputs,
    renderer,
    views,
    center_power=2.0,
):
    """Project v3 dense outer semantics into the physical head eye-band UV.

    The result is evidence, not an occupancy prediction.  It is consumed only
    by the one-hop bilateral completion rule, which additionally requires high
    accessory symmetry plus either observed mirrored outer evidence or a
    mirrored pair of observed inner colors.
    """
    canonical = canonicalize_parser_outputs(outputs)
    logits = canonical.get("dense_semantic_logits")
    if logits is None or logits.dim() != 4 or logits.shape[1] != 5:
        return None, {
            "available": False,
            "reason": "requires_dense_semantic_target_v3",
        }

    probabilities = logits.float().softmax(dim=1)
    outer_probability = probabilities[:, :3].sum(dim=1, keepdim=True)
    outer_pooled_by_view, outer_supported_by_view = (
        aggregate_direct_outer_values_by_view(
            renderer,
            views,
            outer_probability,
            center_power=float(center_power),
        )
    )
    inner_pooled_by_view, inner_supported_by_view = (
        aggregate_direct_inner_values_by_view(
            renderer,
            views,
            outer_probability,
            center_power=float(center_power),
        )
    )

    def collapse_views(pooled_by_view, supported_by_view):
        supported = supported_by_view.unsqueeze(2)
        pooled_supported = torch.where(
            supported,
            pooled_by_view,
            torch.full_like(pooled_by_view, -1.0),
        )
        collapsed = pooled_supported.amax(dim=1)[:, 0]
        return torch.where(
            supported_by_view.any(dim=1),
            collapsed,
            torch.zeros_like(collapsed),
        ), supported_by_view.any(dim=1)

    direct_outer, direct_outer_supported = collapse_views(
        outer_pooled_by_view, outer_supported_by_view
    )
    direct_inner, direct_inner_supported = collapse_views(
        inner_pooled_by_view, inner_supported_by_view
    )

    metadata = build_basic_minecraft_metadata(
        device=outer_probability.device
    )
    counterpart = metadata["counterpart_texel"]
    outer_head = (
        metadata["valid"]
        & (metadata["layer"] == 1)
        & (metadata["part"] == 0)
        & (counterpart >= 0)
    )
    source_inner = counterpart[outer_head]
    inner_as_outer = torch.zeros_like(direct_outer)
    inner_as_outer_supported = torch.zeros_like(
        direct_outer_supported
    )
    inner_as_outer[:, outer_head] = direct_inner[:, source_inner]
    inner_as_outer_supported[:, outer_head] = direct_inner_supported[
        :, source_inner
    ]
    projected = torch.maximum(direct_outer, inner_as_outer).reshape(
        -1, UV_SIZE, UV_SIZE
    )
    projected_supported = (
        direct_outer_supported | inner_as_outer_supported
    ).reshape(-1, UV_SIZE, UV_SIZE)

    eye_faces = projected.new_zeros(projected.shape[0], 6, 8, 8)
    eye_faces[:, (0, 2, 3), 1:6] = 1.0
    eye_band = head_outer_face_values_to_uv(eye_faces)[:, 0] > 0.5
    projected = torch.where(eye_band, projected, torch.zeros_like(projected))
    supported_eye = eye_band & projected_supported
    stats = {
        "available": True,
        "supported_eye_texels": int(supported_eye.sum().item()),
        "mean_supported_probability": round(
            float(projected[supported_eye].mean().item()), 6
        )
        if supported_eye.any()
        else 0.0,
        "max_probability": round(float(projected.max().item()), 6),
    }
    prompt_scores = outputs.get("text_prompt_scores")
    if (
        prompt_scores is not None
        and prompt_scores.dim() == 2
        and prompt_scores.shape[1] >= 4
        and prompt_scores.shape[0] % len(views) == 0
    ):
        eye_over_inner = (
            prompt_scores[:, 1].float() - prompt_scores[:, 3].float()
        ).reshape(-1, len(views))
        stats["global_eye_over_inner_margin_min"] = round(
            float(eye_over_inner.amin(dim=1)[0].item()), 6
        )
        stats["global_eye_over_inner_margin_mean"] = round(
            float(eye_over_inner.mean(dim=1)[0].item()), 6
        )
    return projected, stats


def load_parser(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_args = dict(checkpoint.get("args", {}))
    checkpoint_metrics = checkpoint.get("metrics", {})
    checkpoint_metric_source = (
        checkpoint_metrics.get("val")
        or checkpoint_metrics.get("train")
        or {}
    )
    for metric_name in (
        "outer_uv_occupancy_precision",
        "outer_uv_occupancy_recall",
        "head_outer_occupancy_precision",
        "head_outer_occupancy_recall",
    ):
        if metric_name in checkpoint_metric_source:
            checkpoint_args[f"_checkpoint_{metric_name}"] = float(
                checkpoint_metric_source[metric_name]
            )
    model_config = checkpoint.get("model_config", {})
    if model_config.get("arm_model", "steve") != "steve":
        raise ValueError("Geometry parser only supports standard Steve arms.")
    state_dict = checkpoint["model"]
    has_uv_classification = any(key.startswith("uv_x.") or key.startswith("uv_y.") for key in state_dict)
    has_layer_face = any(key.startswith("layer_face.") for key in state_dict)
    has_route_prior = "route_role_prior" in state_dict
    has_outer_uv_occupancy = any(
        key.startswith("outer_uv_occupancy_head.") for key in state_dict
    )
    has_global_head_outer_structure = any(
        key.startswith("head_outer_face_occupancy_head.")
        for key in state_dict
    )
    has_projected_head_outer_structure = any(
        key.startswith("head_outer_projected_head.") for key in state_dict
    )
    has_head_outer_structure = (
        has_global_head_outer_structure
        or has_projected_head_outer_structure
    )
    text_prompt_key = (
        "semantic_text_prompt_fusion.prompt_embeddings"
    )
    has_text_prompt_fusion = text_prompt_key in state_dict
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
        semantic_spatial_feature_dim=model_config.get(
            "semantic_spatial_feature_dim", 0
        ),
        semantic_spatial_channels=model_config.get(
            "semantic_spatial_channels", 64
        ),
        semantic_text_prompt_count=model_config.get(
            "semantic_text_prompt_count",
            state_dict[text_prompt_key].shape[0]
            if has_text_prompt_fusion
            else 0,
        ),
        semantic_text_prompt_feature_dim=model_config.get(
            "semantic_text_prompt_feature_dim",
            state_dict[text_prompt_key].shape[1]
            if has_text_prompt_fusion
            else 0,
        ),
        semantic_text_prompt_channels=model_config.get(
            "semantic_text_prompt_channels", 32
        ),
        semantic_text_logit_scale=model_config.get(
            "semantic_text_logit_scale", 1.0
        ),
        semantic_text_logit_bias=model_config.get(
            "semantic_text_logit_bias", 0.0
        ),
        dense_semantic_target_version=model_config.get(
            "dense_semantic_target_version", 1
        ),
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
        predict_outer_uv_occupancy=model_config.get(
            "predict_outer_uv_occupancy", has_outer_uv_occupancy
        ),
        predict_head_outer_structure=model_config.get(
            "predict_head_outer_structure", has_head_outer_structure
        ),
        head_outer_structure_mode=model_config.get(
            "head_outer_structure_mode",
            "projected" if has_projected_head_outer_structure else "global",
        ),
        head_outer_projected_input_version=model_config.get(
            "head_outer_projected_input_version", 1
        ),
        outer_uv_feature_channels=model_config.get(
            "outer_uv_feature_channels", 32
        ),
        outer_uv_topology_channels=model_config.get(
            "outer_uv_topology_channels", 64
        ),
        outer_uv_topology_layers=model_config.get(
            "outer_uv_topology_layers", 3
        ),
        outer_uv_topology_dropout=model_config.get(
            "outer_uv_topology_dropout", 0.05
        ),
        outer_uv_route_evidence_dropout=model_config.get(
            "outer_uv_route_evidence_dropout", 1.0
        ),
        cross_view_spatial_fusion=model_config.get(
            "cross_view_spatial_fusion", False
        ),
    ).to(device)
    model.load_state_dict(state_dict)
    object.__setattr__(
        model,
        "semantic_text_prompts",
        tuple(model_config.get("siglip_text_prompts", ())),
    )
    if (
        model.semantic_feature_dim > 0
        or model.semantic_spatial_feature_dim > 0
    ):
        semantic_backbone = model_config.get("semantic_backbone", "siglip2")
        semantic_model = model_config.get(
            "semantic_model",
            model_config.get(
                "tipsv2_model"
                if semantic_backbone == "tipsv2"
                else "siglip_model",
                checkpoint_args.get(
                    "tipsv2_model"
                    if semantic_backbone == "tipsv2"
                    else "siglip_model",
                    "google/tipsv2-b14"
                    if semantic_backbone == "tipsv2"
                    else "google/siglip2-base-patch16-224",
                ),
            ),
        )
        attach_semantic_runtime(
            model,
            semantic_backbone,
            semantic_model,
            device,
            local_files_only=bool(
                checkpoint_args.get(
                    "tipsv2_local_files_only"
                    if semantic_backbone == "tipsv2"
                    else "siglip_local_files_only",
                    False,
                )
            ),
            runtime_batch_size=int(
                checkpoint_args.get("semantic_runtime_batch_size", 32)
            ),
        )
    model.eval()
    return model, checkpoint_args


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


def simple_inpaint_uv(
    conditioning,
    alpha_threshold=0.5,
    head_outer_probability=None,
    head_outer_threshold=0.65,
    head_outer_min_component_seeds=2,
    head_outer_symmetric_candidate_probability=None,
    head_outer_symmetry_probability=None,
    head_outer_symmetry_threshold=0.80,
    head_outer_symmetry_candidate_threshold=0.20,
    head_outer_closed_ring_probability=None,
    head_outer_closed_ring_threshold=0.70,
    head_outer_open_top_probability=None,
    head_outer_open_top_threshold=0.70,
    head_outer_open_top_max_gap=3,
):
    """Repair inner UV holes while preserving the parser's outer layer."""
    parser_uv = conditioning_to_pred_uv(conditioning)
    if parser_uv.dim() != 4 or parser_uv.shape[0] != 1:
        raise ValueError(
            "Simple parser UV inpainting requires exactly one conditioning "
            f"sample, got {tuple(parser_uv.shape)}."
        )
    repaired, stats = simple_symmetry_nearest_inpaint(
        parser_uv[0],
        alpha_threshold=alpha_threshold,
        head_outer_probability=head_outer_probability,
        head_outer_threshold=head_outer_threshold,
        head_outer_min_component_seeds=head_outer_min_component_seeds,
        head_outer_symmetric_candidate_probability=(
            head_outer_symmetric_candidate_probability
        ),
        head_outer_symmetry_probability=head_outer_symmetry_probability,
        head_outer_symmetry_threshold=head_outer_symmetry_threshold,
        head_outer_symmetry_candidate_threshold=(
            head_outer_symmetry_candidate_threshold
        ),
        head_outer_closed_ring_probability=(
            head_outer_closed_ring_probability
        ),
        head_outer_closed_ring_threshold=head_outer_closed_ring_threshold,
        head_outer_open_top_probability=head_outer_open_top_probability,
        head_outer_open_top_threshold=head_outer_open_top_threshold,
        head_outer_open_top_max_gap=head_outer_open_top_max_gap,
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
    return repaired, stats


def save_simple_inpaint_uv(conditioning, output_path, alpha_threshold=0.5):
    repaired, stats = simple_inpaint_uv(
        conditioning,
        alpha_threshold=alpha_threshold,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_rgba_image(repaired.detach().cpu()).save(output_path)
    print("simple_inpaint_stats=" + json.dumps(stats, sort_keys=True))
    print(f"Saved parser_simple_inpaint_uv={output_path}")
    return repaired, stats


def log_and_save_semantic_diagnostics(
    outputs,
    views,
    parser_model,
    output_json_path=None,
):
    """Format, display, and optionally save comprehensive semantic diagnostics."""
    semantic_report = {}

    # 1. SigLIP2 Text Prompt Semantics
    if "text_prompt_scores" in outputs:
        prompts = getattr(parser_model, "semantic_text_prompts", ())
        prompt_results = []
        for view, scores in zip(views, outputs["text_prompt_scores"]):
            scores_f = scores.float().cpu()
            sorted_indices = torch.argsort(scores_f, descending=True)
            view_prompts = []
            for rank, idx in enumerate(sorted_indices[:5]):
                idx_val = int(idx.item())
                p_text = (
                    prompts[idx_val]
                    if idx_val < len(prompts)
                    else f"prompt_{idx_val}"
                )
                p_score = float(scores_f[idx_val].item())
                view_prompts.append({
                    "rank": rank + 1,
                    "index": idx_val,
                    "prompt": p_text,
                    "score": round(p_score, 4),
                })
            prompt_results.append({"view": view, "top_prompts": view_prompts})
        semantic_report["text_prompts"] = prompt_results

    # 2. Part-wise Outer Presence & Coverage
    part_names = ["head", "body", "left_arm", "right_arm", "left_leg", "right_leg"]
    if "outer_presence" in outputs:
        presence = torch.sigmoid(outputs["outer_presence"].float()).cpu()[0]
        coverage = (
            torch.sigmoid(outputs["outer_coverage"].float()).cpu()[0]
            if "outer_coverage" in outputs
            else None
        )
        part_stats = {}
        for p_idx, p_name in enumerate(part_names):
            part_stats[p_name] = {
                "outer_presence_prob": round(float(presence[p_idx].item()), 4),
                "outer_coverage": (
                    round(float(coverage[p_idx].item()), 4)
                    if coverage is not None
                    else None
                ),
            }
        semantic_report["part_outer_prediction"] = part_stats

    # 3. Head Outer Accessory Classification
    if "head_outer_accessory_logits" in outputs:
        acc_probs = torch.sigmoid(outputs["head_outer_accessory_logits"].float()).cpu()[0]
        symm_prob = (
            torch.sigmoid(outputs["head_outer_symmetry_logit"].float()).cpu()[0].item()
            if "head_outer_symmetry_logit" in outputs
            else None
        )
        semantic_report["head_accessory"] = {
            "closed_ring_prob": (
                round(float(acc_probs[0].item()), 4)
                if acc_probs.numel() > 0
                else None
            ),
            "open_top_prob": (
                round(float(acc_probs[1].item()), 4)
                if acc_probs.numel() > 1
                else None
            ),
            "symmetry_prob": (
                round(float(symm_prob), 4)
                if symm_prob is not None
                else None
            ),
        }

    # 4. Dedicated projected head-eye occupancy (v5+).  Report this branch
    # separately from the broad pixel-semantic labels: it is the branch that
    # may promote isolated glasses/goggles into the outer layer.
    if "head_eye_face_occupancy_logits" in outputs:
        eye_probability = torch.sigmoid(
            outputs["head_eye_face_occupancy_logits"].float()
        ).detach().cpu()[0]
        eye_band = torch.zeros_like(eye_probability, dtype=torch.bool)
        eye_band[(0, 2, 3), 1:6] = True
        eye_band_probability = eye_probability[eye_band]
        presence_probability = (
            torch.sigmoid(
                outputs["head_eye_accessory_presence_logit"].float()
            ).detach().cpu()[0].item()
            if "head_eye_accessory_presence_logit" in outputs
            else None
        )
        semantic_report["head_eye_accessory"] = {
            "source": "dedicated_projected_uv_head",
            "presence_probability": (
                round(float(presence_probability), 4)
                if presence_probability is not None
                else None
            ),
            "active_texels_at_0_65": int(
                (eye_band_probability >= 0.65).sum().item()
            ),
            "eye_band_texels": int(eye_band_probability.numel()),
            "mean_eye_band_probability": round(
                float(eye_band_probability.mean().item()), 4
            ),
            "max_eye_band_probability": round(
                float(eye_band_probability.max().item()), 4
            ),
        }

    # 5. 3D Outer UV Occupancy Stats
    if "outer_uv_occupancy_logits" in outputs:
        occ_prob = torch.sigmoid(outputs["outer_uv_occupancy_logits"].float()).cpu()[0, 0]
        active_texels = int((occ_prob > 0.5).sum().item())
        mean_prob = float(occ_prob.mean().item())
        semantic_report["outer_uv_occupancy"] = {
            "predicted_active_texels": active_texels,
            "mean_occupancy_prob": round(mean_prob, 4),
        }

    # Human-Readable Console Output
    print("\n" + "━" * 68)
    print(" 🧠 [Dense UV Parser 语义诊断报告 / Semantic Diagnostics]")
    print("━" * 68)

    if "text_prompts" in semantic_report:
        print(" 📌 视觉-语言语义特征识别 (SigLIP2 Text Prompts):")
        for v_item in semantic_report["text_prompts"]:
            print(f"   • 视角 [{v_item['view']}]:")
            for p in v_item["top_prompts"]:
                bar_len = max(0, min(15, int((p["score"] + 1.0) * 7.5)))
                bar = "■" * bar_len + "□" * (15 - bar_len)
                print(f"     - #{p['rank']} [{bar}] {p['score']:+.4f} : {p['prompt']}")

    if "part_outer_prediction" in semantic_report:
        print("\n 📌 各部位外层存在性预测 (Part Outer Presence):")
        for p_name, p_stat in semantic_report["part_outer_prediction"].items():
            cov_str = (
                f", 覆盖率: {p_stat['outer_coverage']*100:5.1f}%"
                if p_stat["outer_coverage"] is not None
                else ""
            )
            prob_pct = p_stat["outer_presence_prob"] * 100
            flag = "✅ [有外层]" if prob_pct >= 50 else "⬜ [无外层]"
            print(f"   • {p_name:10s}: {flag} 存在概率: {prob_pct:5.1f}%{cov_str}")

    if "head_accessory" in semantic_report:
        ha = semantic_report["head_accessory"]
        print("\n 📌 头部饰品几何属性 (Head Accessory Attributes):")
        if ha["open_top_prob"] is not None:
            print(f"   • 镂空顶面/皇冠特征 (Open Top Crown):   {ha['open_top_prob']*100:5.1f}%")
        if ha["closed_ring_prob"] is not None:
            print(f"   • 闭合环/发带特征 (Closed Ring Decor): {ha['closed_ring_prob']*100:5.1f}%")
        if ha["symmetry_prob"] is not None:
            print(f"   • 头部左右对称置信度 (Symmetry):       {ha['symmetry_prob']*100:5.1f}%")

    if "head_eye_accessory" in semantic_report:
        eye = semantic_report["head_eye_accessory"]
        presence = eye["presence_probability"]
        presence_text = (
            f"{presence*100:5.1f}%" if presence is not None else "n/a"
        )
        print("\n 📌 投影式眼部外层分支 (Projected Eye Accessory):")
        print(f"   • 配件存在概率: {presence_text}")
        print(
            "   • 眼部带活跃纹素: "
            f"{eye['active_texels_at_0_65']} / {eye['eye_band_texels']} "
            f"(均值: {eye['mean_eye_band_probability']*100:.1f}%, "
            f"最大值: {eye['max_eye_band_probability']*100:.1f}%)"
        )

    if "outer_uv_occupancy" in semantic_report:
        occ = semantic_report["outer_uv_occupancy"]
        print(
            f"\n 📌 3D UV 外层占有率: 预测活跃外层纹素 = {occ['predicted_active_texels']} / 2048 (均值: {occ['mean_occupancy_prob']*100:.1f}%)"
        )

    print("━" * 68 + "\n")

    if output_json_path is not None:
        try:
            output_json_path = Path(output_json_path)
            output_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(semantic_report, f, ensure_ascii=False, indent=2)
            print(f"Saved semantic_output={output_json_path}")
        except Exception as e:
            print(f"Warning: Failed to save semantic summary to {output_json_path}: {e}")

    return semantic_report


def save_semantic_pixel_labels(
    outputs,
    rendered,
    observed_foreground=None,
    output_path=None,
):
    """Save 2D pixel-level semantic concept prediction map."""
    if output_path is None:
        return
    sim = outputs.get("spatial_prompt_similarity")
    dense_logits = outputs.get("dense_semantic_logits")
    if sim is None and dense_logits is None:
        return

    evidence = dense_logits if dense_logits is not None else sim
    N, P, H, W = evidence.shape
    predicted_classes = evidence.argmax(dim=1)

    if P == 4:
        palette_values = [
            [255, 215, 0],    # 0: head_top_accessory (gold)
            [220, 20, 60],    # 1: other_outer (crimson)
            [30, 144, 255],   # 2: inner (dodger blue)
            [30, 30, 30],     # 3: background (dark gray)
        ]
        background_index = 3
    elif P == 5:
        palette_values = [
            [255, 215, 0],    # 0: head_top_accessory (gold)
            [0, 255, 200],    # 1: head_eye_accessory (bright cyan)
            [220, 20, 60],    # 2: other_outer (crimson)
            [30, 144, 255],   # 3: inner (dodger blue)
            [30, 30, 30],     # 4: background (dark gray)
        ]
        background_index = 4
    else:
        palette_values = [
            [0, 255, 200],    # 0: outer_glasses (bright cyan)
            [255, 215, 0],    # 1: outer_crown_hat (gold)
            [255, 105, 180],  # 2: outer_ears (hot pink)
            [147, 112, 219],  # 3: outer_jacket (medium purple)
            [220, 20, 60],    # 4: outer_limbs (crimson)
            [255, 140, 0],    # 5: outer_hair (dark orange)
            [0, 191, 255],    # 6: outer_mask (deep sky blue)
            [178, 34, 34],    # 7: outer_back (firebrick)
            [255, 200, 160],  # 8: inner_face (peach / flesh)
            [139, 69, 19],    # 9: inner_hair (saddle brown)
            [245, 222, 179],  # 10: inner_skin (wheat)
            [30, 144, 255],   # 11: inner_clothes (dodger blue)
            [50, 205, 50],    # 12: inner_logo (lime green)
            [255, 255, 255],  # 13: inner_pattern (white)
            [30, 30, 30],     # 14: background (dark gray)
        ]
        background_index = 14
    palette = torch.tensor(
        palette_values,
        dtype=torch.float32,
        device=evidence.device,
    ) / 255.0

    color_map = palette[predicted_classes.clamp(0, palette.shape[0] - 1)]
    color_map = color_map.permute(0, 3, 1, 2)

    if observed_foreground is not None:
        fg = observed_foreground.unsqueeze(1) if observed_foreground.dim() == 3 else observed_foreground
        bg = palette[background_index].view(1, 3, 1, 1)
        color_map = torch.where(fg.bool(), color_map, bg)

    raw_rgb = rendered[:, :3].to(device=evidence.device, dtype=torch.float32)
    blended = raw_rgb * 0.50 + color_map * 0.50
    grid = torch.cat([raw_rgb, color_map, blended], dim=0)

    try:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        save_image(grid.clamp(0.0, 1.0).detach().cpu(), str(out_p), nrow=N)
        print(f"Saved semantic_pixel_output={out_p}")
    except Exception as e:
        print(f"Warning: Failed to save semantic pixel map to {output_path}: {e}")


def _raw_debug_foreground(
    outputs,
    routing,
    fg_threshold,
    observed_foreground=None,
):
    """Mask raw semantic heads with observed input foreground when available.

    The routed foreground is deliberately stricter: it excludes pixels rejected
    by route confidence, margin, and UV coverage filters.  Using it for a raw
    head preview makes those routing rejections look like holes in the semantic
    prediction, even though the head produced a class at every foreground pixel.
    """
    model_foreground = (
        torch.sigmoid(outputs["foreground"])[:, 0] > fg_threshold
    )
    if observed_foreground is not None:
        if observed_foreground.shape != model_foreground.shape:
            raise ValueError(
                "observed_foreground must have shape "
                f"{tuple(model_foreground.shape)}, got "
                f"{tuple(observed_foreground.shape)}."
            )
        return observed_foreground.to(
            device=model_foreground.device,
            dtype=torch.bool,
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
    raw_outputs=None,
    raw_observed_foreground=None,
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
    raw_head_outputs = outputs if raw_outputs is None else raw_outputs
    raw_layer_values = raw_head_outputs["layer"].argmax(dim=1)
    raw_fg = _raw_debug_foreground(
        raw_head_outputs,
        None if raw_outputs is not None else routing,
        fg_threshold,
        observed_foreground=raw_observed_foreground,
    )
    pred_layer_values = routing["layer"] if routing is not None else raw_layer_values
    pred_layer = torch.where(
        pred_fg,
        pred_layer_values,
        torch.full_like(outputs["layer"].argmax(dim=1), IGNORE_INDEX),
    )
    raw_face_values = (
        raw_head_outputs["face"].argmax(dim=1)
        if "face" in raw_head_outputs
        else routing["face"]
    )
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
    if "layer_face" in raw_head_outputs:
        raw_layer_face_values = raw_head_outputs["layer_face"].argmax(dim=1)
        raw_layer_face = torch.where(
            raw_fg,
            raw_layer_face_values,
            torch.full_like(raw_layer_face_values, IGNORE_INDEX),
        )
    else:
        raw_layer_values_for_debug = raw_layer_values
        if raw_head_outputs["layer"].shape[1] == 3:
            # Geometry-fit class 2 is a secondary/backface role, not a skin
            # layer. It has no raw inner/outer value until geometry routing,
            # so leave it unlabelled in the pre-affine joint preview.
            raw_layer_values_for_debug = raw_layer_values.masked_fill(
                raw_layer_values == 2,
                IGNORE_INDEX,
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
    preprocessing_defaults = PRODUCTION_PREPROCESSING_DEFAULTS
    splat_defaults = PRODUCTION_SPLAT_DEFAULTS
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
        default=preprocessing_defaults["foreground_flood_tolerance"],
        help="Maximum per-channel RGB distance from the top-left flood seed.",
    )
    parser.add_argument(
        "--foreground_flood_gradient_tolerance",
        type=float,
        default=preprocessing_defaults[
            "foreground_flood_gradient_tolerance"
        ],
        help=(
            "Maximum colour step for following a smooth background gradient "
            "away from the top-left seed."
        ),
    )
    parser.add_argument(
        "--foreground_flood_max_seed_tolerance",
        type=float,
        default=preprocessing_defaults[
            "foreground_flood_max_seed_tolerance"
        ],
        help=(
            "Maximum total drift from the top-left colour while following a "
            "smooth background gradient."
        ),
    )
    parser.add_argument(
        "--foreground_parser_background",
        choices=["adaptive", "neutral"],
        default=preprocessing_defaults["foreground_parser_background"],
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
    parser.add_argument(
        "--output",
        default="outputs/pred_uv.png",
        help=(
            "Final deterministic RGBA UV PNG. Unknown inner-layer texels are "
            "completed by the topology-aware simple inpainting algorithm; "
            "the predicted outer layer is preserved unchanged."
        ),
    )
    parser.add_argument("--conditioning_output", default=None, help="Optional preview image for parser-splatted conditioning.")
    parser.add_argument(
        "--parser_uv_output",
        default="outputs/parser_pred_uv.png",
        help="Optional preliminary RGBA skin merged directly from parser conditioning.",
    )
    parser.add_argument(
        "--simple_inpaint_output",
        default="outputs/parser_pred_uv_simple_inpainting.png",
        help=(
            "Optional copy of the deterministic pre-final UV repair. This is "
            "the same completion used by --output."
        ),
    )
    parser.add_argument(
        "--simple_inpaint_render_output",
        default="outputs/simple_inpaint_render.png",
        help=(
            "Combined front+back render of the simple-inpainted skin. "
            "Left half = front_left view, right half = back_left view."
        ),
    )
    parser.add_argument("--debug_output", default=None, help="Optional path to write a debug preview grid of predictions.")
    parser.add_argument("--semantic_output", default=None, help="Optional path to write a JSON summary of semantic predictions.")
    parser.add_argument(
        "--semantic_pixel_output",
        default="outputs/parser_debug_semantic_pixel_labels.png",
        help="Optional path to write a 2D pixel-level semantic concept map.",
    )
    parser.add_argument(
        "--head_eye_semantic_outer_uv_output",
        default="outputs/parser_debug_head_eye_semantic_outer_uv.png",
        help=(
            "Optional grayscale UV diagnostic for projected v3 eye-band "
            "outer semantic evidence."
        ),
    )
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
    parser.add_argument(
        "--canonical_foreground_output",
        default=None,
        help=(
            "Foreground mask after coverage-preserving affine "
            "canonicalization, before route filtering."
        ),
    )
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
    parser.add_argument(
        "--outer_uv_occupancy_output",
        default=None,
        help=(
            "Projected outer-UV occupancy diagnostic. Columns show raw "
            "probability, topology-propagated probability, and component "
            "routing masks (red=rejected, green=accepted, blue=seed)."
        ),
    )
    parser.add_argument(
        "--head_outer_structure_output",
        default=None,
        help=(
            "Head-only semantic occupancy diagnostic. The left atlas shows "
            "probability and the right atlas shows the 0.50 mask."
        ),
    )
    parser.add_argument("--front", default=None)
    parser.add_argument("--back", default=None)
    parser.add_argument("--combined", default=None)
    parser.add_argument("--view_images", nargs="*", default=None)
    parser.add_argument("--mappings_dir", default=None)
    parser.add_argument(
        "--fg_threshold",
        type=float,
        default=splat_defaults["fg_threshold"],
    )
    parser.add_argument(
        "--background_color_tolerance",
        type=float,
        default=splat_defaults["background_color_tolerance"],
        help="RGB distance used to reject solid-background and antialiased edge pixels.",
    )
    parser.add_argument(
        "--color_background_tolerance",
        type=float,
        default=splat_defaults["color_background_tolerance"],
        help=(
            "Reject background-like RGB candidates only on the foreground "
            "boundary before inverse UV color sampling."
        ),
    )
    parser.add_argument(
        "--color_foreground_inset",
        type=int,
        default=splat_defaults["color_foreground_inset"],
        help="Foreground boundary width demoted during texel-center color selection.",
    )
    parser.add_argument(
        "--route_confidence_threshold",
        type=float,
        default=splat_defaults["route_confidence_threshold"],
    )
    parser.add_argument(
        "--route_margin_threshold",
        type=float,
        default=splat_defaults["route_margin_threshold"],
    )
    parser.add_argument(
        "--outer_route_confidence_threshold",
        type=float,
        default=splat_defaults["outer_route_confidence_threshold"],
    )
    parser.add_argument(
        "--outer_route_margin_threshold",
        type=float,
        default=splat_defaults["outer_route_margin_threshold"],
    )
    parser.add_argument(
        "--outer_uv_min_coverage",
        type=float,
        default=splat_defaults["outer_uv_min_coverage"],
        help="Reject outer UV texels supported by less than this fraction of their projected footprint.",
    )
    parser.add_argument(
        "--outer_uv_min_source_pixels",
        type=int,
        default=splat_defaults["outer_uv_min_source_pixels"],
        help="Minimum routed source pixels required to keep an outer UV texel.",
    )
    parser.add_argument(
        "--outer_silhouette_consistency",
        dest="outer_silhouette_consistency",
        action="store_true",
        default=splat_defaults["outer_silhouette_consistency"],
        help=(
            "Reject an outer texel when its protruding projection is absent "
            "from the observed foreground silhouette."
        ),
    )
    parser.add_argument(
        "--no_outer_silhouette_consistency",
        dest="outer_silhouette_consistency",
        action="store_false",
    )
    parser.add_argument(
        "--outer_silhouette_min_coverage",
        type=float,
        default=splat_defaults["outer_silhouette_min_coverage"],
    )
    parser.add_argument(
        "--outer_silhouette_dilation",
        type=int,
        default=splat_defaults["outer_silhouette_dilation"],
    )
    parser.add_argument(
        "--outer_silhouette_min_pixels",
        type=int,
        default=splat_defaults["outer_silhouette_min_pixels"],
    )
    parser.add_argument(
        "--head_outer_topology_rescue",
        dest="head_outer_topology_rescue",
        action="store_true",
        default=splat_defaults["head_outer_topology_rescue"],
        help=(
            "Restore relaxed head-outer candidates only when they connect "
            "multiple strict seeds in the physical head-cube topology."
        ),
    )
    parser.add_argument(
        "--no_head_outer_topology_rescue",
        dest="head_outer_topology_rescue",
        action="store_false",
    )
    parser.add_argument(
        "--head_outer_topology_auto_reliability",
        dest="head_outer_topology_auto_reliability",
        action="store_true",
        default=preprocessing_defaults[
            "head_outer_topology_auto_reliability"
        ],
        help=(
            "Disable topology rescue automatically when checkpoint head "
            "occupancy precision/recall is below the production floor."
        ),
    )
    parser.add_argument(
        "--no_head_outer_topology_auto_reliability",
        dest="head_outer_topology_auto_reliability",
        action="store_false",
    )
    parser.add_argument(
        "--head_outer_topology_min_precision",
        type=float,
        default=preprocessing_defaults[
            "head_outer_topology_min_precision"
        ],
    )
    parser.add_argument(
        "--head_outer_topology_min_recall",
        type=float,
        default=preprocessing_defaults["head_outer_topology_min_recall"],
    )
    parser.add_argument(
        "--head_outer_topology_semantic_threshold",
        type=float,
        default=splat_defaults["head_outer_topology_semantic_threshold"],
    )
    parser.add_argument(
        "--head_outer_topology_relaxed_route_threshold",
        type=float,
        default=splat_defaults[
            "head_outer_topology_relaxed_route_threshold"
        ],
    )
    parser.add_argument(
        "--head_outer_topology_relaxed_semantic_threshold",
        type=float,
        default=splat_defaults[
            "head_outer_topology_relaxed_semantic_threshold"
        ],
    )
    parser.add_argument(
        "--head_outer_topology_semantic_only_threshold",
        type=float,
        default=splat_defaults[
            "head_outer_topology_semantic_only_threshold"
        ],
    )
    parser.add_argument(
        "--head_outer_topology_ring_semantic_threshold",
        type=float,
        default=splat_defaults["head_outer_topology_ring_semantic_threshold"],
        help=(
            "Minimum head-structure probability for filling gaps in an "
            "established horizontal accessory ring."
        ),
    )
    parser.add_argument(
        "--head_outer_topology_min_seed_nodes",
        type=int,
        default=splat_defaults["head_outer_topology_min_seed_nodes"],
    )
    parser.add_argument(
        "--head_outer_topology_color_tolerance",
        type=float,
        default=splat_defaults["head_outer_topology_color_tolerance"],
    )
    parser.add_argument(
        "--head_outer_completion_threshold",
        type=float,
        default=splat_defaults["head_outer_completion_threshold"],
        help=(
            "Minimum projected head-occupancy probability for constrained, "
            "component-anchored outer UV completion."
        ),
    )
    parser.add_argument(
        "--head_outer_completion_min_component_seeds",
        type=int,
        default=splat_defaults[
            "head_outer_completion_min_component_seeds"
        ],
    )
    parser.add_argument(
        "--head_outer_symmetry_completion_threshold",
        type=float,
        default=splat_defaults[
            "head_outer_symmetry_completion_threshold"
        ],
        help=(
            "Minimum v3 predicted accessory symmetry before isolated "
            "mirrored head-outer texels may be completed."
        ),
    )
    parser.add_argument(
        "--head_outer_symmetry_candidate_threshold",
        type=float,
        default=splat_defaults[
            "head_outer_symmetry_candidate_threshold"
        ],
        help=(
            "Minimum v3 head-occupancy support for a directly mirrored "
            "completion candidate."
        ),
    )
    parser.add_argument(
        "--head_eye_semantic_symmetry_rescue",
        dest="head_eye_semantic_symmetry_rescue",
        action="store_true",
        default=splat_defaults["head_eye_semantic_symmetry_rescue"],
        help=(
            "Allow strong v3 head-eye outer semantics to complete only an "
            "exactly mirrored, already observed outer texel."
        ),
    )
    parser.add_argument(
        "--no_head_eye_semantic_symmetry_rescue",
        dest="head_eye_semantic_symmetry_rescue",
        action="store_false",
    )
    parser.add_argument(
        "--head_eye_semantic_symmetry_candidate_threshold",
        type=float,
        default=splat_defaults[
            "head_eye_semantic_symmetry_candidate_threshold"
        ],
        help=(
            "Minimum projected total-outer semantic probability for a "
            "mirrored eye-band completion candidate."
        ),
    )
    parser.add_argument(
        "--head_eye_semantic_prompt_margin_threshold",
        type=float,
        default=splat_defaults[
            "head_eye_semantic_prompt_margin_threshold"
        ],
        help=(
            "Minimum eye-accessory minus inner-layer SigLIP text score in "
            "every input view before semantic eye-band completion is allowed."
        ),
    )
    parser.add_argument(
        "--head_outer_closed_ring_completion_threshold",
        type=float,
        default=splat_defaults[
            "head_outer_closed_ring_completion_threshold"
        ],
        help=(
            "Minimum v4 closed-side-ring confidence before completing a "
            "missing hat-brim face."
        ),
    )
    parser.add_argument(
        "--head_outer_open_top_completion_threshold",
        type=float,
        default=splat_defaults[
            "head_outer_open_top_completion_threshold"
        ],
        help=(
            "Minimum v4 open-top confidence before closing short crown-rim "
            "gaps."
        ),
    )
    parser.add_argument(
        "--head_outer_open_top_max_gap",
        type=int,
        default=splat_defaults["head_outer_open_top_max_gap"],
        help="Largest bounded top-perimeter gap repaired by v4 completion.",
    )
    parser.add_argument(
        "--outer_geometry_rescue",
        dest="outer_geometry_rescue",
        action="store_true",
        default=splat_defaults["outer_geometry_rescue"],
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
        default=splat_defaults["outer_semantic_rescue"],
        help="Relax outer gates only on parts whose global semantic heads predict a substantial outer layer.",
    )
    parser.add_argument(
        "--no_outer_semantic_rescue",
        dest="outer_semantic_rescue",
        action="store_false",
    )
    parser.add_argument(
        "--outer_semantic_presence_threshold",
        type=float,
        default=splat_defaults["outer_semantic_presence_threshold"],
    )
    parser.add_argument(
        "--outer_semantic_coverage_threshold",
        type=float,
        default=splat_defaults["outer_semantic_coverage_threshold"],
    )
    parser.add_argument(
        "--outer_rescue_confidence_threshold",
        type=float,
        default=splat_defaults["outer_rescue_confidence_threshold"],
    )
    parser.add_argument(
        "--outer_rescue_margin_threshold",
        type=float,
        default=splat_defaults["outer_rescue_margin_threshold"],
    )
    parser.add_argument(
        "--outer_rescue_min_coverage",
        type=float,
        default=splat_defaults["outer_rescue_min_coverage"],
    )
    parser.add_argument(
        "--geometry_route_texel_consensus",
        dest="geometry_route_texel_consensus",
        action="store_true",
        default=splat_defaults["geometry_route_texel_consensus"],
        help="Use projected UV-cell voting instead of semantic-first per-pixel routing.",
    )
    parser.add_argument(
        "--no_geometry_route_texel_consensus",
        dest="geometry_route_texel_consensus",
        action="store_false",
    )
    parser.add_argument(
        "--geometry_route_texel_consensus_weight",
        type=float,
        default=splat_defaults["geometry_route_texel_consensus_weight"],
        help="Blend weight for center-weighted texel consensus; 0 keeps local probabilities and 1 uses only the cell aggregate.",
    )
    parser.add_argument(
        "--geometry_route_preserve_outer_confidence",
        type=float,
        default=splat_defaults["geometry_route_preserve_outer_confidence"],
        help=(
            "Minimum local outer confidence required in addition to the fused "
            "texel gate when the raw route predicts outer."
        ),
    )
    parser.add_argument(
        "--geometry_route_preserve_outer_margin",
        type=float,
        default=splat_defaults["geometry_route_preserve_outer_margin"],
        help=(
            "Minimum local outer margin required in addition to the fused "
            "texel gate when the raw route predicts outer."
        ),
    )
    parser.add_argument(
        "--geometry_route_consensus_outer_confidence",
        type=float,
        default=splat_defaults[
            "geometry_route_consensus_outer_confidence"
        ],
        help="Minimum fused confidence required to promote a route to outer.",
    )
    parser.add_argument(
        "--geometry_route_consensus_outer_margin",
        type=float,
        default=splat_defaults["geometry_route_consensus_outer_margin"],
        help="Minimum fused margin ratio required to promote a route to outer.",
    )
    parser.add_argument(
        "--geometry_cross_view_outer_consistency",
        dest="geometry_cross_view_outer_consistency",
        action="store_true",
        default=splat_defaults["geometry_cross_view_outer_consistency"],
        help=(
            "Veto a strong outer route when another view gives strong "
            "background/inner evidence for the same outer UV texel."
        ),
    )
    parser.add_argument(
        "--no_geometry_cross_view_outer_consistency",
        dest="geometry_cross_view_outer_consistency",
        action="store_false",
    )
    parser.add_argument(
        "--geometry_cross_view_outer_weight",
        type=float,
        default=splat_defaults["geometry_cross_view_outer_weight"],
    )
    parser.add_argument(
        "--geometry_cross_view_outer_positive_confidence",
        type=float,
        default=splat_defaults[
            "geometry_cross_view_outer_positive_confidence"
        ],
    )
    parser.add_argument(
        "--geometry_cross_view_outer_positive_margin",
        type=float,
        default=splat_defaults[
            "geometry_cross_view_outer_positive_margin"
        ],
    )
    parser.add_argument(
        "--geometry_cross_view_outer_negative_confidence",
        type=float,
        default=splat_defaults[
            "geometry_cross_view_outer_negative_confidence"
        ],
    )
    parser.add_argument(
        "--geometry_cross_view_outer_negative_margin",
        type=float,
        default=splat_defaults[
            "geometry_cross_view_outer_negative_margin"
        ],
    )
    parser.add_argument(
        "--geometry_cross_view_outer_background_max_coverage",
        type=float,
        default=splat_defaults[
            "geometry_cross_view_outer_background_max_coverage"
        ],
    )
    parser.add_argument(
        "--geometry_cross_view_outer_min_views",
        type=int,
        default=splat_defaults["geometry_cross_view_outer_min_views"],
    )
    parser.add_argument(
        "--outer_uv_occupancy",
        dest="outer_uv_occupancy",
        action="store_true",
        default=splat_defaults["outer_uv_occupancy"],
        help="Use the checkpoint's grouped 64x64 outer-layer occupancy prior.",
    )
    parser.add_argument(
        "--no_outer_uv_occupancy",
        dest="outer_uv_occupancy",
        action="store_false",
    )
    parser.add_argument(
        "--outer_uv_occupancy_blend_weight",
        type=float,
        default=splat_defaults["outer_uv_occupancy_blend_weight"],
    )
    parser.add_argument(
        "--outer_uv_occupancy_gate_threshold",
        type=float,
        default=splat_defaults["outer_uv_occupancy_gate_threshold"],
    )
    parser.add_argument(
        "--outer_uv_occupancy_rescue_threshold",
        type=float,
        default=splat_defaults["outer_uv_occupancy_rescue_threshold"],
    )
    parser.add_argument(
        "--outer_uv_occupancy_rescue_route_threshold",
        type=float,
        default=splat_defaults[
            "outer_uv_occupancy_rescue_route_threshold"
        ],
    )
    parser.add_argument(
        "--outer_uv_occupancy_auto_reliability",
        dest="outer_uv_occupancy_auto_reliability",
        action="store_true",
        default=True,
        help=(
            "Allow occupancy to alter routing only when checkpoint validation "
            "precision and recall clear the configured floors."
        ),
    )
    parser.add_argument(
        "--no_outer_uv_occupancy_auto_reliability",
        dest="outer_uv_occupancy_auto_reliability",
        action="store_false",
    )
    parser.add_argument(
        "--outer_uv_occupancy_min_precision", type=float, default=0.60
    )
    parser.add_argument(
        "--outer_uv_occupancy_min_recall", type=float, default=0.40
    )
    parser.add_argument(
        "--outer_uv_component_routing",
        dest="outer_uv_component_routing",
        action="store_true",
        default=splat_defaults["outer_uv_component_routing"],
    )
    parser.add_argument(
        "--no_outer_uv_component_routing",
        dest="outer_uv_component_routing",
        action="store_false",
    )
    parser.add_argument(
        "--outer_uv_component_seed_threshold",
        type=float,
        default=splat_defaults["outer_uv_component_seed_threshold"],
    )
    parser.add_argument(
        "--outer_uv_component_grow_threshold",
        type=float,
        default=splat_defaults["outer_uv_component_grow_threshold"],
    )
    parser.add_argument(
        "--outer_uv_component_min_size",
        type=int,
        default=splat_defaults["outer_uv_component_min_size"],
    )
    parser.add_argument(
        "--color_aggregation",
        choices=SPLAT_COLOR_AGGREGATIONS,
        default=splat_defaults["color_aggregation"],
        help="How colors inside each fitted layer/UV grid cell are selected.",
    )
    parser.add_argument(
        "--allow_semantic_fallback",
        action="store_true",
        help="Keep pixels whose strict semantic routing had no valid candidate.",
    )
    parser.add_argument("--no_semantic_gate", dest="semantic_gate", action="store_false", default=splat_defaults["semantic_gate"])
    parser.add_argument("--affine_refine", dest="affine_refine", action="store_true", default=splat_defaults["affine_refine"])
    parser.add_argument("--no_affine_refine", dest="affine_refine", action="store_false")
    parser.add_argument("--affine_refine_translation_px", type=float, default=splat_defaults["affine_refine_translation_px"])
    parser.add_argument("--affine_refine_scale", type=float, default=splat_defaults["affine_refine_scale"])
    parser.add_argument("--alpha_threshold", type=float, default=preprocessing_defaults["alpha_threshold"])
    parser.add_argument(
        "--hypothesis_render_refine",
        dest="hypothesis_render_refine",
        action="store_true",
        default=splat_defaults.get("hypothesis_render_refine", True),
        help="Arbitrate outer vs inner skin hypotheses via multi-view differentiable re-rendering.",
    )
    parser.add_argument(
        "--no_hypothesis_render_refine",
        dest="hypothesis_render_refine",
        action="store_false",
    )
    parser.add_argument(
        "--protect_chin_occlusion",
        dest="protect_chin_occlusion",
        action="store_true",
        default=splat_defaults.get("protect_chin_occlusion", True),
        help="Veto outer head patches that occlude the face/chin unless confirmed by 3D render loss.",
    )
    parser.add_argument(
        "--no_protect_chin_occlusion",
        dest="protect_chin_occlusion",
        action="store_false",
    )
    parser.add_argument("--device", default="auto")
    return parser


def main():
    args = build_arg_parser().parse_args()
    if not 0.0 <= args.background_color_tolerance <= 1.0:
        raise ValueError("--background_color_tolerance must be in [0, 1].")
    if not 0.0 <= args.color_background_tolerance <= 1.0:
        raise ValueError("--color_background_tolerance must be in [0, 1].")
    if args.color_foreground_inset < 0:
        raise ValueError("--color_foreground_inset must be non-negative.")
    if args.outer_uv_min_source_pixels < 1:
        raise ValueError("--outer_uv_min_source_pixels must be positive.")
    if not 0.0 <= args.outer_silhouette_min_coverage <= 1.0:
        raise ValueError(
            "--outer_silhouette_min_coverage must be in [0, 1]."
        )
    if args.outer_silhouette_dilation < 0:
        raise ValueError("--outer_silhouette_dilation must be non-negative.")
    if args.outer_silhouette_min_pixels < 1:
        raise ValueError("--outer_silhouette_min_pixels must be positive.")
    if not 0.0 <= args.head_outer_topology_semantic_threshold <= 1.0:
        raise ValueError(
            "--head_outer_topology_semantic_threshold must be in [0, 1]."
        )
    if not 0.0 <= args.head_outer_topology_relaxed_route_threshold <= 1.0:
        raise ValueError(
            "--head_outer_topology_relaxed_route_threshold must be in [0, 1]."
        )
    if not 0.0 <= args.head_outer_topology_relaxed_semantic_threshold <= 1.0:
        raise ValueError(
            "--head_outer_topology_relaxed_semantic_threshold must be in [0, 1]."
        )
    if not 0.0 <= args.head_outer_topology_semantic_only_threshold <= 1.0:
        raise ValueError(
            "--head_outer_topology_semantic_only_threshold must be in [0, 1]."
        )
    if not 0.0 <= args.head_outer_topology_ring_semantic_threshold <= 1.0:
        raise ValueError(
            "--head_outer_topology_ring_semantic_threshold must be in [0, 1]."
        )
    if args.head_outer_topology_min_seed_nodes < 2:
        raise ValueError(
            "--head_outer_topology_min_seed_nodes must be at least 2."
        )
    if args.head_outer_topology_color_tolerance < 0.0:
        raise ValueError(
            "--head_outer_topology_color_tolerance must be non-negative."
        )
    if not 0.0 <= args.head_outer_completion_threshold <= 1.0:
        raise ValueError(
            "--head_outer_completion_threshold must be in [0, 1]."
        )
    if args.head_outer_completion_min_component_seeds < 1:
        raise ValueError(
            "--head_outer_completion_min_component_seeds must be positive."
        )
    if not 0.0 <= args.head_outer_symmetry_completion_threshold <= 1.0:
        raise ValueError(
            "--head_outer_symmetry_completion_threshold must be in [0, 1]."
        )
    if not 0.0 <= args.head_outer_symmetry_candidate_threshold <= 1.0:
        raise ValueError(
            "--head_outer_symmetry_candidate_threshold must be in [0, 1]."
        )
    if not 0.0 <= args.head_eye_semantic_symmetry_candidate_threshold <= 1.0:
        raise ValueError(
            "--head_eye_semantic_symmetry_candidate_threshold must be in [0, 1]."
        )
    if args.head_eye_semantic_prompt_margin_threshold < 0.0:
        raise ValueError(
            "--head_eye_semantic_prompt_margin_threshold must be non-negative."
        )
    if not 0.0 <= args.head_outer_closed_ring_completion_threshold <= 1.0:
        raise ValueError(
            "--head_outer_closed_ring_completion_threshold must be in [0, 1]."
        )
    if not 0.0 <= args.head_outer_open_top_completion_threshold <= 1.0:
        raise ValueError(
            "--head_outer_open_top_completion_threshold must be in [0, 1]."
        )
    if args.head_outer_open_top_max_gap < 0:
        raise ValueError("--head_outer_open_top_max_gap must be non-negative.")
    if not 0.0 <= args.head_outer_topology_min_precision <= 1.0:
        raise ValueError(
            "--head_outer_topology_min_precision must be in [0, 1]."
        )
    if not 0.0 <= args.head_outer_topology_min_recall <= 1.0:
        raise ValueError(
            "--head_outer_topology_min_recall must be in [0, 1]."
        )
    if not 0.0 <= args.foreground_flood_tolerance <= 1.0:
        raise ValueError("--foreground_flood_tolerance must be in [0, 1].")
    if not 0.0 <= args.foreground_flood_gradient_tolerance <= 1.0:
        raise ValueError(
            "--foreground_flood_gradient_tolerance must be in [0, 1]."
        )
    if not 0.0 <= args.foreground_flood_max_seed_tolerance <= 1.0:
        raise ValueError(
            "--foreground_flood_max_seed_tolerance must be in [0, 1]."
        )
    if not 0.0 <= args.outer_uv_occupancy_min_precision <= 1.0:
        raise ValueError(
            "--outer_uv_occupancy_min_precision must be in [0, 1]."
        )
    if not 0.0 <= args.outer_uv_occupancy_min_recall <= 1.0:
        raise ValueError(
            "--outer_uv_occupancy_min_recall must be in [0, 1]."
        )
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
            args.outer_uv_occupancy_output,
            args.head_outer_structure_output,
            args.simple_inpaint_render_output,
        )
    ):
        raise ValueError(
            "Provide --output, --conditioning_output, --parser_uv_output, --simple_inpaint_output, "
            "--simple_inpaint_render_output, "
            "--debug_output, --overlay_output, "
            "--inner_cutout_output, --outer_cutout_output, --secondary_cutout_output, "
            "and/or --color_source_output."
        )

    device = get_device(args.device)
    parser_model, parser_args = load_parser(args.parser_checkpoint, device)
    views = parse_views(parser_args.get("views", "walk_front_both_layer_ortho,walk_back_both_layer_ortho"))
    if parser_model.view_classes not in (0, len(views)):
        raise ValueError(
            f"Parser checkpoint expects {parser_model.view_classes} views, but its metadata lists {len(views)}: {views}"
        )
    mappings_dir = args.mappings_dir or parser_args.get("mappings_dir")
    renderer = DifferentiableRenderer(mappings_dir=mappings_dir).to(device)
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
    geometry_route_texel_consensus_weight = (
        parser_args.get("geometry_route_texel_consensus_weight", 0.60)
        if args.geometry_route_texel_consensus_weight is None
        else args.geometry_route_texel_consensus_weight
    )
    geometry_route_preserve_outer_confidence = (
        parser_args.get("geometry_route_preserve_outer_confidence", 0.80)
        if args.geometry_route_preserve_outer_confidence is None
        else args.geometry_route_preserve_outer_confidence
    )
    geometry_route_preserve_outer_margin = (
        parser_args.get("geometry_route_preserve_outer_margin", 0.35)
        if args.geometry_route_preserve_outer_margin is None
        else args.geometry_route_preserve_outer_margin
    )
    geometry_route_consensus_outer_confidence = (
        parser_args.get("geometry_route_consensus_outer_confidence", 0.70)
        if args.geometry_route_consensus_outer_confidence is None
        else args.geometry_route_consensus_outer_confidence
    )
    geometry_route_consensus_outer_margin = (
        parser_args.get("geometry_route_consensus_outer_margin", 0.20)
        if args.geometry_route_consensus_outer_margin is None
        else args.geometry_route_consensus_outer_margin
    )
    geometry_cross_view_outer_consistency = (
        parser_args.get("geometry_cross_view_outer_consistency", False)
        if args.geometry_cross_view_outer_consistency is None
        else args.geometry_cross_view_outer_consistency
    )
    geometry_cross_view_outer_weight = (
        parser_args.get("geometry_cross_view_outer_weight", 0.50)
        if args.geometry_cross_view_outer_weight is None
        else args.geometry_cross_view_outer_weight
    )
    geometry_cross_view_outer_positive_confidence = (
        parser_args.get(
            "geometry_cross_view_outer_positive_confidence", 0.70
        )
        if args.geometry_cross_view_outer_positive_confidence is None
        else args.geometry_cross_view_outer_positive_confidence
    )
    geometry_cross_view_outer_positive_margin = (
        parser_args.get("geometry_cross_view_outer_positive_margin", 0.20)
        if args.geometry_cross_view_outer_positive_margin is None
        else args.geometry_cross_view_outer_positive_margin
    )
    geometry_cross_view_outer_negative_confidence = (
        parser_args.get(
            "geometry_cross_view_outer_negative_confidence", 0.70
        )
        if args.geometry_cross_view_outer_negative_confidence is None
        else args.geometry_cross_view_outer_negative_confidence
    )
    geometry_cross_view_outer_negative_margin = (
        parser_args.get("geometry_cross_view_outer_negative_margin", 0.20)
        if args.geometry_cross_view_outer_negative_margin is None
        else args.geometry_cross_view_outer_negative_margin
    )
    geometry_cross_view_outer_background_max_coverage = (
        parser_args.get(
            "geometry_cross_view_outer_background_max_coverage", 0.25
        )
        if args.geometry_cross_view_outer_background_max_coverage is None
        else args.geometry_cross_view_outer_background_max_coverage
    )
    geometry_cross_view_outer_min_views = (
        parser_args.get("geometry_cross_view_outer_min_views", 2)
        if args.geometry_cross_view_outer_min_views is None
        else args.geometry_cross_view_outer_min_views
    )
    head_checkpoint_precision = parser_args.get(
        "_checkpoint_head_outer_occupancy_precision"
    )
    head_checkpoint_recall = parser_args.get(
        "_checkpoint_head_outer_occupancy_recall"
    )
    head_topology_reliable = (
        head_checkpoint_precision is not None
        and head_checkpoint_recall is not None
        and head_checkpoint_precision
        >= float(args.head_outer_topology_min_precision)
        and head_checkpoint_recall
        >= float(args.head_outer_topology_min_recall)
    )
    head_outer_topology_rescue = bool(
        args.head_outer_topology_rescue
    ) and (
        head_topology_reliable
        or not args.head_outer_topology_auto_reliability
    )
    print(
        "head_outer_topology_reliability="
        + json.dumps(
            {
                "auto": bool(args.head_outer_topology_auto_reliability),
                "enabled": bool(head_outer_topology_rescue),
                "requested": bool(args.head_outer_topology_rescue),
                "checkpoint_precision": head_checkpoint_precision,
                "checkpoint_recall": head_checkpoint_recall,
                "min_precision": float(
                    args.head_outer_topology_min_precision
                ),
                "min_recall": float(args.head_outer_topology_min_recall),
            },
            sort_keys=True,
        )
    )
    outer_uv_occupancy_requested = (
        parser_args.get(
            "outer_uv_occupancy_routing",
            False,
        )
        if args.outer_uv_occupancy is None
        else args.outer_uv_occupancy
    )
    occupancy_checkpoint_precision = parser_args.get(
        "_checkpoint_outer_uv_occupancy_precision"
    )
    occupancy_checkpoint_recall = parser_args.get(
        "_checkpoint_outer_uv_occupancy_recall"
    )
    occupancy_reliable = (
        occupancy_checkpoint_precision is not None
        and occupancy_checkpoint_recall is not None
        and occupancy_checkpoint_precision
        >= float(args.outer_uv_occupancy_min_precision)
        and occupancy_checkpoint_recall
        >= float(args.outer_uv_occupancy_min_recall)
    )
    outer_uv_occupancy = bool(outer_uv_occupancy_requested) and (
        occupancy_reliable
        or not args.outer_uv_occupancy_auto_reliability
    )
    print(
        "outer_uv_occupancy_reliability="
        + json.dumps(
            {
                "auto": bool(args.outer_uv_occupancy_auto_reliability),
                "enabled": bool(outer_uv_occupancy),
                "requested": bool(outer_uv_occupancy_requested),
                "checkpoint_precision": occupancy_checkpoint_precision,
                "checkpoint_recall": occupancy_checkpoint_recall,
                "min_precision": float(
                    args.outer_uv_occupancy_min_precision
                ),
                "min_recall": float(args.outer_uv_occupancy_min_recall),
            },
            sort_keys=True,
        )
    )
    outer_uv_occupancy_blend_weight = (
        parser_args.get("outer_uv_occupancy_blend_weight", 0.0)
        if args.outer_uv_occupancy_blend_weight is None
        else args.outer_uv_occupancy_blend_weight
    )
    outer_uv_occupancy_gate_threshold = (
        parser_args.get("outer_uv_occupancy_gate_threshold", 0.15)
        if args.outer_uv_occupancy_gate_threshold is None
        else args.outer_uv_occupancy_gate_threshold
    )
    outer_uv_occupancy_rescue_threshold = (
        parser_args.get("outer_uv_occupancy_rescue_threshold", 0.70)
        if args.outer_uv_occupancy_rescue_threshold is None
        else args.outer_uv_occupancy_rescue_threshold
    )
    outer_uv_occupancy_rescue_route_threshold = (
        parser_args.get(
            "outer_uv_occupancy_rescue_route_threshold", 0.30
        )
        if args.outer_uv_occupancy_rescue_route_threshold is None
        else args.outer_uv_occupancy_rescue_route_threshold
    )
    outer_uv_component_routing = (
        parser_args.get("outer_uv_component_routing", False)
        if args.outer_uv_component_routing is None
        else args.outer_uv_component_routing
    )
    outer_uv_component_seed_threshold = (
        parser_args.get("outer_uv_component_seed_threshold", 0.80)
        if args.outer_uv_component_seed_threshold is None
        else args.outer_uv_component_seed_threshold
    )
    outer_uv_component_grow_threshold = (
        parser_args.get("outer_uv_component_grow_threshold", 0.50)
        if args.outer_uv_component_grow_threshold is None
        else args.outer_uv_component_grow_threshold
    )
    outer_uv_component_min_size = (
        parser_args.get("outer_uv_component_min_size", 2)
        if args.outer_uv_component_min_size is None
        else args.outer_uv_component_min_size
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
                gradient_tolerance=(
                    args.foreground_flood_gradient_tolerance
                ),
                max_seed_tolerance=(
                    args.foreground_flood_max_seed_tolerance
                ),
            )
            foreground_log = {
                "method": "top_left_flood",
                "seed_rgb": [
                    [int(round(channel * 255.0)) for channel in color]
                    for color in rendered[:, :3, 0, 0].detach().cpu().tolist()
                ],
                "tolerance": round(float(args.foreground_flood_tolerance), 6),
                "gradient_tolerance": round(
                    float(args.foreground_flood_gradient_tolerance), 6
                ),
                "max_seed_tolerance": round(
                    float(args.foreground_flood_max_seed_tolerance), 6
                ),
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
        static_mappings = (
            [
                build_static_surface_routing(renderer, view, device)
                for view in views
            ]
            if renderer is not None
            else None
        )
        outputs = parser_model(
            parser_rendered,
            view_ids=view_ids,
            semantic_foreground=observed_foreground,
            static_mappings=static_mappings,
        )
        if args.semantic_pixel_output:
            save_semantic_pixel_labels(
                outputs,
                parser_rendered,
                observed_foreground=observed_foreground,
                output_path=args.semantic_pixel_output,
            )
        outputs = attach_projected_outer_uv_occupancy(
            parser_model,
            outputs,
            renderer,
            views,
            observed_foreground=observed_foreground,
            center_power=float(
                parser_args.get("route_texel_center_power", 2.0)
            ),
        )
        outputs = attach_projected_head_outer_structure(
            parser_model,
            outputs,
            renderer,
            views,
            observed_foreground=observed_foreground,
            source_images=parser_rendered,
            center_power=float(
                parser_args.get("route_texel_center_power", 2.0)
            ),
        )
        log_and_save_semantic_diagnostics(
            outputs,
            views,
            parser_model,
            output_json_path=args.semantic_output,
        )
        if (
            args.outer_uv_occupancy_output
            and "outer_uv_occupancy_logits" in outputs
        ):
            occupancy_probability = torch.sigmoid(
                outputs["outer_uv_occupancy_logits"]
            )
            occupancy_components = outer_uv_topology_hysteresis(
                occupancy_probability,
                seed_threshold=outer_uv_component_seed_threshold,
                grow_threshold=outer_uv_component_grow_threshold,
                min_component_size=outer_uv_component_min_size,
            )
            raw_preview = occupancy_probability.expand(-1, 3, -1, -1)
            structured_preview = occupancy_components[
                "probability"
            ].expand(-1, 3, -1, -1)
            component_preview = torch.cat(
                (
                    occupancy_components["rejected_candidate"].float(),
                    occupancy_components["accepted"].float(),
                    occupancy_components["seed"].float(),
                ),
                dim=1,
            )
            occupancy_preview = torch.cat(
                (raw_preview, structured_preview, component_preview),
                dim=-1,
            )
            occupancy_path = Path(args.outer_uv_occupancy_output)
            occupancy_path.parent.mkdir(parents=True, exist_ok=True)
            save_image(occupancy_preview.detach().cpu(), occupancy_path)
            print(f"Saved outer_uv_occupancy={occupancy_path}")
        if (
            args.head_outer_structure_output
            and "head_outer_face_occupancy_logits" in outputs
        ):
            head_probability = torch.sigmoid(
                outputs["head_outer_face_occupancy_logits"].float()
            )
            head_atlas = head_outer_face_values_to_uv(head_probability)
            head_preview = torch.cat(
                (
                    head_atlas.expand(-1, 3, -1, -1),
                    (head_atlas >= 0.50).float().expand(-1, 3, -1, -1),
                ),
                dim=-1,
            )
            head_path = Path(args.head_outer_structure_output)
            head_path.parent.mkdir(parents=True, exist_ok=True)
            save_image(head_preview.detach().cpu(), head_path)
            print(f"Saved head_outer_structure={head_path}")
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
            outer_silhouette_consistency=args.outer_silhouette_consistency,
            outer_silhouette_min_coverage=(
                args.outer_silhouette_min_coverage
            ),
            outer_silhouette_dilation=args.outer_silhouette_dilation,
            outer_silhouette_min_pixels=args.outer_silhouette_min_pixels,
            head_outer_topology_rescue=head_outer_topology_rescue,
            head_outer_topology_semantic_threshold=(
                args.head_outer_topology_semantic_threshold
            ),
            head_outer_topology_relaxed_route_threshold=(
                args.head_outer_topology_relaxed_route_threshold
            ),
            head_outer_topology_relaxed_semantic_threshold=(
                args.head_outer_topology_relaxed_semantic_threshold
            ),
            head_outer_topology_semantic_only_threshold=(
                args.head_outer_topology_semantic_only_threshold
            ),
            head_outer_topology_ring_semantic_threshold=(
                args.head_outer_topology_ring_semantic_threshold
            ),
            head_outer_topology_min_seed_nodes=(
                args.head_outer_topology_min_seed_nodes
            ),
            head_outer_topology_color_tolerance=(
                args.head_outer_topology_color_tolerance
            ),
            outer_geometry_rescue=args.outer_geometry_rescue,
            outer_semantic_rescue=args.outer_semantic_rescue,
            outer_semantic_presence_threshold=args.outer_semantic_presence_threshold,
            outer_semantic_coverage_threshold=args.outer_semantic_coverage_threshold,
            outer_rescue_confidence_threshold=args.outer_rescue_confidence_threshold,
            outer_rescue_margin_threshold=args.outer_rescue_margin_threshold,
            outer_rescue_min_coverage=args.outer_rescue_min_coverage,
            color_aggregation=args.color_aggregation,
            geometry_route_texel_consensus=geometry_route_texel_consensus,
            geometry_route_texel_consensus_weight=(
                geometry_route_texel_consensus_weight
            ),
            geometry_route_preserve_outer_confidence=(
                geometry_route_preserve_outer_confidence
            ),
            geometry_route_preserve_outer_margin=(
                geometry_route_preserve_outer_margin
            ),
            geometry_route_consensus_outer_confidence=(
                geometry_route_consensus_outer_confidence
            ),
            geometry_route_consensus_outer_margin=(
                geometry_route_consensus_outer_margin
            ),
            geometry_cross_view_outer_consistency=(
                geometry_cross_view_outer_consistency
            ),
            geometry_cross_view_outer_weight=(
                geometry_cross_view_outer_weight
            ),
            geometry_cross_view_outer_positive_confidence=(
                geometry_cross_view_outer_positive_confidence
            ),
            geometry_cross_view_outer_positive_margin=(
                geometry_cross_view_outer_positive_margin
            ),
            geometry_cross_view_outer_negative_confidence=(
                geometry_cross_view_outer_negative_confidence
            ),
            geometry_cross_view_outer_negative_margin=(
                geometry_cross_view_outer_negative_margin
            ),
            geometry_cross_view_outer_background_max_coverage=(
                geometry_cross_view_outer_background_max_coverage
            ),
            geometry_cross_view_outer_min_views=(
                geometry_cross_view_outer_min_views
            ),
            outer_uv_occupancy=outer_uv_occupancy,
            outer_uv_occupancy_blend_weight=(
                outer_uv_occupancy_blend_weight
            ),
            outer_uv_occupancy_gate_threshold=(
                outer_uv_occupancy_gate_threshold
            ),
            outer_uv_occupancy_rescue_threshold=(
                outer_uv_occupancy_rescue_threshold
            ),
            outer_uv_occupancy_rescue_route_threshold=(
                outer_uv_occupancy_rescue_route_threshold
            ),
            outer_uv_component_routing=outer_uv_component_routing,
            outer_uv_component_seed_threshold=(
                outer_uv_component_seed_threshold
            ),
            outer_uv_component_grow_threshold=(
                outer_uv_component_grow_threshold
            ),
            outer_uv_component_min_size=outer_uv_component_min_size,
            observed_foreground=observed_foreground,
            background_color_tolerance=args.background_color_tolerance,
            color_background_tolerance=args.color_background_tolerance,
            color_foreground_inset=args.color_foreground_inset,
            reject_semantic_fallback=not args.allow_semantic_fallback,
            include_rejected_context=False,
            include_confidence=False,
            return_details=True,
        )

    routing = routing_details.get("routing")
    if routing is not None:
        if args.canonical_foreground_output:
            canonical_foreground_path = Path(
                args.canonical_foreground_output
            )
            canonical_foreground_path.parent.mkdir(
                parents=True, exist_ok=True
            )
            save_image(
                routing["observed_foreground"]
                .unsqueeze(1)
                .to(dtype=rendered.dtype)
                .detach()
                .cpu(),
                canonical_foreground_path,
                nrow=len(views),
            )
        observed_routing_foreground = routing.get(
            "observed_foreground", routing["raw_foreground"]
        )
        observed_count = int(observed_routing_foreground.sum().item())
        canonical_foreground_coverage_rescued_count = int(
            routing.get(
                "canonical_foreground_coverage_rescued",
                torch.zeros_like(observed_routing_foreground),
            ).sum().item()
        )
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
        outer_silhouette_assessed_count = int(
            routing.get(
                "outer_silhouette_assessed",
                torch.zeros_like(raw_outer),
            ).sum().item()
        )
        outer_silhouette_rejected_count = int(
            routing.get(
                "outer_silhouette_rejected",
                torch.zeros_like(raw_outer),
            ).sum().item()
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
        consensus_changed_count = int(
            routing.get(
                "consensus_changed", torch.zeros_like(raw_outer)
            ).sum().item()
        )
        consensus_inner_to_outer_count = int(
            routing.get(
                "consensus_inner_to_outer", torch.zeros_like(raw_outer)
            ).sum().item()
        )
        consensus_outer_to_inner_count = int(
            routing.get(
                "consensus_outer_to_inner", torch.zeros_like(raw_outer)
            ).sum().item()
        )
        consensus_preserved_outer_count = int(
            routing.get(
                "consensus_preserved_outer", torch.zeros_like(raw_outer)
            ).sum().item()
        )
        consensus_outer_gate_rejected_count = int(
            routing.get(
                "consensus_outer_gate_rejected",
                torch.zeros_like(raw_outer),
            ).sum().item()
        )
        consensus_outer_gate_deferred_count = int(
            (
                routing.get(
                    "consensus_outer_gate_rejected",
                    torch.zeros_like(raw_outer),
                )
                & (routing.get("route_role") == 1)
            ).sum().item()
        )
        cross_view_outer_shared_count = int(
            routing.get(
                "cross_view_outer_shared", torch.zeros_like(raw_outer)
            ).sum().item()
        )
        cross_view_outer_conflict_count = int(
            routing.get(
                "cross_view_outer_conflict", torch.zeros_like(raw_outer)
            ).sum().item()
        )
        cross_view_outer_vetoed_count = int(
            routing.get(
                "cross_view_outer_vetoed", torch.zeros_like(raw_outer)
            ).sum().item()
        )
        silhouette_outer_rescued_count = int(
            routing.get(
                "silhouette_outer_rescued", torch.zeros_like(raw_outer)
            ).sum().item()
        )
        silhouette_outer_only_count = int(
            routing.get(
                "silhouette_outer_only", torch.zeros_like(raw_outer)
            ).sum().item()
        )
        occupancy_promoted_outer_count = int(
            routing.get(
                "occupancy_rescued_outer", torch.zeros_like(raw_outer)
            ).sum().item()
        )
        occupancy_rejected_outer_count = int(
            routing.get(
                "occupancy_rejected_outer", torch.zeros_like(raw_outer)
            ).sum().item()
        )
        occupancy_supported_outer_count = int(
            routing.get(
                "outer_occupancy_supported",
                torch.zeros_like(raw_outer),
            ).sum().item()
        )
        occupancy_trusted_outer_count = int(
            routing.get(
                "outer_occupancy_rescued",
                torch.zeros_like(raw_outer),
            ).sum().item()
        )
        occupancy_component_seed_count = int(
            routing.get(
                "projected_outer_component_seed",
                torch.zeros_like(raw_outer),
            ).sum().item()
        )
        occupancy_component_grown_count = int(
            routing.get(
                "projected_outer_component_grown",
                torch.zeros_like(raw_outer),
            ).sum().item()
        )
        occupancy_component_rejected_count = int(
            routing.get(
                "projected_outer_component_rejected",
                torch.zeros_like(raw_outer),
            ).sum().item()
        )
        head_topology_details = routing.get(
            "head_outer_topology_details"
        ) or {}
        print(
            "routing_filter="
            + json.dumps(
                {
                    "observed_foreground_pixels": observed_count,
                    "canonical_foreground_coverage_rescued_pixels": (
                        canonical_foreground_coverage_rescued_count
                    ),
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
                    "outer_silhouette_consistency": bool(
                        args.outer_silhouette_consistency
                    ),
                    "outer_silhouette_min_coverage": float(
                        args.outer_silhouette_min_coverage
                    ),
                    "outer_silhouette_view_reduction": "minimum",
                    "outer_silhouette_dilation": int(
                        args.outer_silhouette_dilation
                    ),
                    "outer_silhouette_evidence_inset": 0,
                    "outer_silhouette_evidence": "color_safe_foreground",
                    "outer_silhouette_assessed_pixels": (
                        outer_silhouette_assessed_count
                    ),
                    "outer_silhouette_rejected_pixels": (
                        outer_silhouette_rejected_count
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
                    "consensus_changed_pixels": consensus_changed_count,
                    "consensus_inner_to_outer_pixels": (
                        consensus_inner_to_outer_count
                    ),
                    "consensus_outer_to_inner_pixels": (
                        consensus_outer_to_inner_count
                    ),
                    "consensus_preserved_outer_pixels": (
                        consensus_preserved_outer_count
                    ),
                    "consensus_outer_gate_rejected_pixels": (
                        consensus_outer_gate_rejected_count
                    ),
                    "consensus_outer_gate_deferred_pixels": (
                        consensus_outer_gate_deferred_count
                    ),
                    "cross_view_outer_consistency": bool(
                        geometry_cross_view_outer_consistency
                    ),
                    "cross_view_outer_shared_pixels": (
                        cross_view_outer_shared_count
                    ),
                    "cross_view_outer_conflict_pixels": (
                        cross_view_outer_conflict_count
                    ),
                    "cross_view_outer_vetoed_pixels": (
                        cross_view_outer_vetoed_count
                    ),
                    "cross_view_outer_weight": round(
                        float(geometry_cross_view_outer_weight), 6
                    ),
                    "silhouette_outer_rescued_pixels": (
                        silhouette_outer_rescued_count
                    ),
                    "silhouette_outer_only_pixels": silhouette_outer_only_count,
                    "consensus_weight": round(
                        float(geometry_route_texel_consensus_weight), 6
                    ),
                    "outer_uv_occupancy_available": bool(
                        routing.get(
                            "outer_uv_occupancy_available",
                            torch.zeros_like(raw_outer),
                        ).any().item()
                    ),
                    "outer_uv_occupancy_promoted_pixels": (
                        occupancy_promoted_outer_count
                    ),
                    "outer_uv_occupancy_rejected_pixels": (
                        occupancy_rejected_outer_count
                    ),
                    "outer_uv_occupancy_supported_pixels": (
                        occupancy_supported_outer_count
                    ),
                    "outer_uv_occupancy_trusted_rescue_pixels": (
                        occupancy_trusted_outer_count
                    ),
                    "outer_uv_occupancy_blend_weight": round(
                        float(outer_uv_occupancy_blend_weight), 6
                    ),
                    "outer_uv_component_routing": bool(
                        outer_uv_component_routing
                    ),
                    "outer_uv_component_seed_pixels": (
                        occupancy_component_seed_count
                    ),
                    "outer_uv_component_grown_pixels": (
                        occupancy_component_grown_count
                    ),
                    "outer_uv_component_rejected_pixels": (
                        occupancy_component_rejected_count
                    ),
                    "outer_uv_component_seed_threshold": round(
                        float(outer_uv_component_seed_threshold), 6
                    ),
                    "outer_uv_component_grow_threshold": round(
                        float(outer_uv_component_grow_threshold), 6
                    ),
                    "head_outer_topology_rescue": bool(
                        head_outer_topology_rescue
                    ),
                    "head_outer_topology_candidate_source_texels": int(
                        head_topology_details.get(
                            "candidate_source_texels", 0
                        )
                    ),
                    "head_outer_topology_seed_texels": int(
                        head_topology_details.get("seed_texels", 0)
                    ),
                    "head_outer_topology_rescued_texels": int(
                        head_topology_details.get("rescued_texels", 0)
                    ),
                    "head_outer_topology_rescued_pixels": int(
                        head_topology_details.get("rescued_pixels", 0)
                    ),
                    "head_outer_topology_anchored_components": int(
                        head_topology_details.get(
                            "anchored_components", 0
                        )
                    ),
                    "head_outer_topology_color_rejected_texels": int(
                        head_topology_details.get(
                            "color_rejected_texels", 0
                        )
                    ),
                    "head_outer_topology_ring_candidate_texels": int(
                        head_topology_details.get(
                            "ring_candidate_texels", 0
                        )
                    ),
                    "head_outer_topology_ring_rescued_texels": int(
                        head_topology_details.get(
                            "ring_rescued_texels", 0
                        )
                    ),
                    "head_outer_topology_pruned_isolated_texels": int(
                        head_topology_details.get(
                            "pruned_isolated_texels", 0
                        )
                    ),
                    "head_outer_topology_pruned_isolated_pixels": int(
                        head_topology_details.get(
                            "pruned_isolated_pixels", 0
                        )
                    ),
                    "head_outer_protrusion_color_assessed_pixels": int(
                        head_topology_details.get(
                            "protrusion_color_assessed_pixels", 0
                        )
                    ),
                    "head_outer_protrusion_color_rejected_pixels": int(
                        head_topology_details.get(
                            "protrusion_color_rejected_pixels", 0
                        )
                    ),
                    "head_outer_protrusion_color_rejected_texels": int(
                        head_topology_details.get(
                            "protrusion_color_rejected_texels", 0
                        )
                    ),
                    "head_outer_visible_color_assessed_pixels": int(
                        head_topology_details.get(
                            "visible_color_assessed_pixels", 0
                        )
                    ),
                    "head_outer_visible_color_rejected_pixels": int(
                        head_topology_details.get(
                            "visible_color_rejected_pixels", 0
                        )
                    ),
                    "head_outer_visible_color_rejected_texels": int(
                        head_topology_details.get(
                            "visible_color_rejected_texels", 0
                        )
                    ),
                    "head_outer_topology_semantic_threshold": round(
                        float(
                            args.head_outer_topology_semantic_threshold
                        ),
                        6,
                    ),
                    "head_outer_topology_color_tolerance": round(
                        float(args.head_outer_topology_color_tolerance),
                        6,
                    ),
                    "head_outer_topology_relaxed_route_threshold": round(
                        float(
                            args.head_outer_topology_relaxed_route_threshold
                        ),
                        6,
                    ),
                    "head_outer_topology_relaxed_semantic_threshold": round(
                        float(
                            args.head_outer_topology_relaxed_semantic_threshold
                        ),
                        6,
                    ),
                    "head_outer_topology_semantic_only_threshold": round(
                        float(
                            args.head_outer_topology_semantic_only_threshold
                        ),
                        6,
                    ),
                    "head_outer_topology_ring_semantic_threshold": round(
                        float(
                            args.head_outer_topology_ring_semantic_threshold
                        ),
                        6,
                    ),
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
            args.outer_uv_occupancy_output,
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
            raw_outputs=outputs,
            raw_observed_foreground=observed_foreground,
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

    needs_repair = bool(
        args.simple_inpaint_output
        or args.output
        or args.simple_inpaint_render_output
    )
    repair_outputs = []
    if args.simple_inpaint_output:
        repair_outputs.append(
            ("parser_simple_inpaint_uv", Path(args.simple_inpaint_output))
        )
    if args.output:
        repair_outputs.append(("completed_uv", Path(args.output)))
    if needs_repair:
        head_outer_completion_probability = None
        head_outer_symmetric_candidate_probability = None
        head_outer_symmetry_probability = None
        head_outer_closed_ring_probability = None
        head_outer_open_top_probability = None
        if "head_outer_symmetry_logit" in outputs:
            head_outer_symmetry_probability = torch.sigmoid(
                outputs["head_outer_symmetry_logit"].float()
            )[0].detach().cpu()
        if (
            int(
                getattr(
                    parser_model,
                    "head_outer_projected_input_version",
                    1,
                )
            )
            >= 2
            and head_outer_topology_rescue
            and "head_outer_face_occupancy_logits" in outputs
        ):
            head_outer_completion_probability = head_outer_face_values_to_uv(
                torch.sigmoid(
                    outputs["head_outer_face_occupancy_logits"].float()
                )
            )[0, 0].detach().cpu()
            if "head_outer_accessory_logits" in outputs:
                accessory_probability = torch.sigmoid(
                    outputs["head_outer_accessory_logits"].float()
                )[0].detach().cpu()
                head_outer_closed_ring_probability = accessory_probability[0]
                head_outer_open_top_probability = accessory_probability[1]
        learned_head_eye_available = (
            "head_eye_face_occupancy_logits" in outputs
            and "head_eye_accessory_presence_logit" in outputs
        )
        if (
            args.head_eye_semantic_symmetry_rescue
            and head_outer_topology_rescue
            and learned_head_eye_available
        ):
            learned_eye_faces = torch.sigmoid(
                outputs["head_eye_face_occupancy_logits"].float()
            )
            learned_eye_presence = torch.sigmoid(
                outputs["head_eye_accessory_presence_logit"].float()
            )
            eye_band = torch.zeros_like(learned_eye_faces)
            eye_band[:, (0, 2, 3), 1:6] = 1.0
            learned_eye_faces = (
                learned_eye_faces
                * eye_band
                * (learned_eye_presence >= 0.5)
                .to(dtype=learned_eye_faces.dtype)
                .view(-1, 1, 1, 1)
            )
            learned_eye_uv = head_outer_face_values_to_uv(
                learned_eye_faces
            )[:, 0].detach().cpu()
            head_outer_symmetric_candidate_probability = learned_eye_uv[0]
            learned_eye_stats = {
                "available": True,
                "source": "dedicated_projected_uv_head",
                "presence_probability": round(
                    float(learned_eye_presence[0].item()), 6
                ),
                "presence_threshold": 0.5,
                "rescue_enabled_by_presence": bool(
                    learned_eye_presence[0].item() >= 0.5
                ),
                "candidate_threshold": float(
                    args.head_eye_semantic_symmetry_candidate_threshold
                ),
                "candidate_texels": int(
                    (
                        learned_eye_uv[0]
                        >= float(
                            args.head_eye_semantic_symmetry_candidate_threshold
                        )
                    ).sum().item()
                ),
            }
            if args.head_eye_semantic_outer_uv_output:
                learned_eye_path = Path(
                    args.head_eye_semantic_outer_uv_output
                )
                learned_eye_path.parent.mkdir(parents=True, exist_ok=True)
                save_image(
                    learned_eye_uv.unsqueeze(1),
                    learned_eye_path,
                    nrow=learned_eye_uv.shape[0],
                )
                print(f"Saved head_eye_learned_outer_uv={learned_eye_path}")
            print(
                "head_eye_learned_outer_uv="
                + json.dumps(learned_eye_stats, sort_keys=True)
            )
        if (
            args.head_eye_semantic_symmetry_rescue
            and head_outer_topology_rescue
            and not learned_head_eye_available
            and int(
                getattr(parser_model, "dense_semantic_target_version", 1)
            )
            == 3
        ):
            (
                projected_head_eye_outer,
                projected_head_eye_stats,
            ) = project_head_eye_semantic_outer_probability(
                outputs,
                renderer,
                views,
                center_power=float(
                    parser_args.get("route_texel_center_power", 2.0)
                ),
            )
            if projected_head_eye_outer is not None:
                projected_candidate_probability = (
                    projected_head_eye_outer[0].detach().cpu()
                )
                threshold = float(
                    args.head_eye_semantic_symmetry_candidate_threshold
                )
                projected_head_eye_stats["candidate_threshold"] = threshold
                projected_head_eye_stats["candidate_texels"] = int(
                    (
                        projected_candidate_probability
                        >= threshold
                    ).sum().item()
                )
                prompt_margin = projected_head_eye_stats.get(
                    "global_eye_over_inner_margin_min"
                )
                prompt_margin_threshold = float(
                    args.head_eye_semantic_prompt_margin_threshold
                )
                prompt_gate = (
                    prompt_margin is not None
                    and float(prompt_margin) >= prompt_margin_threshold
                )
                projected_head_eye_stats[
                    "prompt_margin_threshold"
                ] = prompt_margin_threshold
                projected_head_eye_stats[
                    "rescue_enabled_by_prompt_margin"
                ] = bool(prompt_gate)
                if prompt_gate:
                    head_outer_symmetric_candidate_probability = (
                        projected_candidate_probability
                    )
                if args.head_eye_semantic_outer_uv_output:
                    semantic_uv_path = Path(
                        args.head_eye_semantic_outer_uv_output
                    )
                    semantic_uv_path.parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    save_image(
                        projected_head_eye_outer.detach().cpu().unsqueeze(1),
                        semantic_uv_path,
                        nrow=projected_head_eye_outer.shape[0],
                    )
                    print(
                        "Saved head_eye_semantic_outer_uv="
                        f"{semantic_uv_path}"
                    )
            print(
                "head_eye_semantic_outer_uv="
                + json.dumps(projected_head_eye_stats, sort_keys=True)
            )
        repaired, stats = simple_inpaint_uv(
            conditioning.detach().cpu(),
            alpha_threshold=args.alpha_threshold,
            head_outer_probability=head_outer_completion_probability,
            head_outer_threshold=args.head_outer_completion_threshold,
            head_outer_min_component_seeds=(
                args.head_outer_completion_min_component_seeds
            ),
            head_outer_symmetric_candidate_probability=(
                head_outer_symmetric_candidate_probability
            ),
            head_outer_symmetry_probability=(
                head_outer_symmetry_probability
            ),
            head_outer_symmetry_threshold=(
                args.head_outer_symmetry_completion_threshold
            ),
            head_outer_symmetry_candidate_threshold=(
                args.head_eye_semantic_symmetry_candidate_threshold
            ),
            head_outer_closed_ring_probability=(
                head_outer_closed_ring_probability
            ),
            head_outer_closed_ring_threshold=(
                args.head_outer_closed_ring_completion_threshold
            ),
            head_outer_open_top_probability=(
                head_outer_open_top_probability
            ),
            head_outer_open_top_threshold=(
                args.head_outer_open_top_completion_threshold
            ),
        )
        if getattr(args, "hypothesis_render_refine", True):
            refiner_target = rendered.to(device)
            if observed_foreground is not None and refiner_target.shape[1] == 3:
                fg_alpha = observed_foreground.to(
                    device=device, dtype=refiner_target.dtype
                )
                if fg_alpha.dim() == 3:
                    fg_alpha = fg_alpha.unsqueeze(1)
                refiner_target = torch.cat([refiner_target, fg_alpha], dim=1)
            repaired, refiner_stats = refine_uv_by_analysis_by_synthesis(
                repaired.to(device),
                refiner_target,
                renderer,
                views,
                alpha_threshold=args.alpha_threshold,
                protect_chin_occlusion=getattr(
                    args, "protect_chin_occlusion", True
                ),
            )
            repaired = repaired.detach().cpu()
            stats["hypothesis_refiner"] = refiner_stats
            print(
                "hypothesis_refiner_stats="
                + json.dumps(refiner_stats, sort_keys=True)
            )
        print("simple_inpaint_stats=" + json.dumps(stats, sort_keys=True))
        written_paths = set()
        for output_label, output_path in repair_outputs:
            resolved_path = output_path.resolve()
            if resolved_path not in written_paths:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                tensor_to_rgba_image(repaired).save(output_path)
                written_paths.add(resolved_path)
            print(f"Saved {output_label}={output_path}")

        if args.simple_inpaint_render_output:
            repaired_skin = repaired.unsqueeze(0).to(device)
            rendered_views = renderer.forward(repaired_skin)
            expected_views = ("front_left", "back_left")
            missing = [v for v in expected_views if v not in rendered_views]
            if missing:
                print(
                    "WARNING: simple_inpaint_render_output requires views "
                    f"{expected_views}, but renderer only has "
                    f"{list(rendered_views.keys())}. "
                    f"Missing views: {missing}. "
                    "Skipping combined render."
                )
            else:
                bg = (
                    torch.tensor(bg_color, device=device, dtype=repaired_skin.dtype)
                    .view(1, 3, 1, 1)
                    / 255.0
                )
                front_rgba = rendered_views["front_left"]
                back_rgba = rendered_views["back_left"]
                front_rgb = front_rgba[:, :3] * front_rgba[:, 3:4] + bg * (
                    1.0 - front_rgba[:, 3:4]
                )
                back_rgb = back_rgba[:, :3] * back_rgba[:, 3:4] + bg * (
                    1.0 - back_rgba[:, 3:4]
                )
                combined = torch.cat([front_rgb, back_rgb], dim=3)
                render_path = Path(args.simple_inpaint_render_output)
                render_path.parent.mkdir(parents=True, exist_ok=True)
                save_image(
                    combined[0].clamp(0.0, 1.0).cpu(),
                    render_path,
                )
                print(f"Saved simple_inpaint_render={render_path}")


if __name__ == "__main__":
    main()
