"""Shared production defaults for Dense UV inference.

The CLI and the persistent GPU worker must use the same routing policy.  Keep
the values in this module aligned with ``run_infer.sh``; callers may copy and
override the returned dictionaries without mutating the global defaults.
"""

from types import MappingProxyType


PRODUCTION_PREPROCESSING_DEFAULTS = MappingProxyType(
    {
        "foreground_flood_tolerance": 0.03,
        "foreground_flood_gradient_tolerance": 0.05,
        "foreground_flood_max_seed_tolerance": 0.20,
        "foreground_parser_background": "adaptive",
        "alpha_threshold": 0.5,
        "head_outer_topology_auto_reliability": True,
        "head_outer_topology_min_precision": 0.75,
        "head_outer_topology_min_recall": 0.55,
    }
)


PRODUCTION_SPLAT_DEFAULTS = MappingProxyType(
    {
        "fg_threshold": 0.5,
        "semantic_gate": True,
        "affine_refine": False,
        "affine_refine_translation_px": 0.0,
        "affine_refine_scale": 0.0,
        "route_confidence_threshold": 0.0,
        "route_margin_threshold": 0.0,
        "outer_route_confidence_threshold": 0.80,
        "outer_route_margin_threshold": 0.55,
        "outer_uv_min_coverage": 0.25,
        "outer_uv_min_source_pixels": 4,
        "outer_silhouette_consistency": True,
        "outer_silhouette_min_coverage": 0.50,
        "outer_silhouette_dilation": 0,
        "outer_silhouette_min_pixels": 1,
        "head_outer_topology_rescue": True,
        "head_outer_topology_semantic_threshold": 0.25,
        "head_outer_topology_relaxed_route_threshold": 0.50,
        "head_outer_topology_relaxed_semantic_threshold": 0.80,
        "head_outer_topology_semantic_only_threshold": 0.92,
        "head_outer_topology_ring_semantic_threshold": 0.65,
        "head_outer_topology_min_seed_nodes": 2,
        "head_outer_topology_color_tolerance": 0.20,
        "head_outer_completion_threshold": 0.65,
        "head_outer_completion_min_component_seeds": 2,
        "head_outer_symmetry_completion_threshold": 0.80,
        "head_outer_symmetry_candidate_threshold": 0.20,
        "head_outer_closed_ring_completion_threshold": 0.70,
        "head_outer_open_top_completion_threshold": 0.70,
        "head_outer_open_top_max_gap": 3,
        "outer_geometry_rescue": True,
        "outer_semantic_rescue": True,
        "outer_semantic_presence_threshold": 0.80,
        "outer_semantic_coverage_threshold": 0.20,
        "outer_rescue_confidence_threshold": 0.60,
        "outer_rescue_margin_threshold": 0.25,
        "outer_rescue_min_coverage": 0.10,
        "color_aggregation": "grid_mode",
        "geometry_route_texel_consensus": True,
        "geometry_route_texel_consensus_weight": 0.60,
        "geometry_route_preserve_outer_confidence": 0.80,
        "geometry_route_preserve_outer_margin": 0.35,
        "geometry_route_consensus_outer_confidence": 0.70,
        "geometry_route_consensus_outer_margin": 0.20,
        "geometry_cross_view_outer_consistency": True,
        "geometry_cross_view_outer_weight": 0.50,
        "geometry_cross_view_outer_positive_confidence": 0.70,
        "geometry_cross_view_outer_positive_margin": 0.20,
        "geometry_cross_view_outer_negative_confidence": 0.70,
        "geometry_cross_view_outer_negative_margin": 0.20,
        "geometry_cross_view_outer_background_max_coverage": 0.25,
        "geometry_cross_view_outer_min_views": 2,
        "outer_uv_occupancy": False,
        "outer_uv_occupancy_blend_weight": 0.0,
        "outer_uv_occupancy_gate_threshold": 0.15,
        "outer_uv_occupancy_rescue_threshold": 0.70,
        "outer_uv_occupancy_rescue_route_threshold": 0.30,
        "outer_uv_component_routing": False,
        "outer_uv_component_seed_threshold": 0.80,
        "outer_uv_component_grow_threshold": 0.50,
        "outer_uv_component_min_size": 2,
        "background_color_tolerance": 0.25,
        # This is only applied to silhouette-boundary color pickup. A wider
        # threshold removes antialiased solid-background fringes without
        # deleting matching colors enclosed inside the character.
        "color_background_tolerance": 0.20,
        "color_foreground_inset": 1,
        "reject_semantic_fallback": True,
        "include_rejected_context": False,
        "include_confidence": False,
    }
)


def production_preprocessing_defaults():
    return dict(PRODUCTION_PREPROCESSING_DEFAULTS)


def production_splat_defaults():
    return dict(PRODUCTION_SPLAT_DEFAULTS)


__all__ = [
    "PRODUCTION_PREPROCESSING_DEFAULTS",
    "PRODUCTION_SPLAT_DEFAULTS",
    "production_preprocessing_defaults",
    "production_splat_defaults",
]
