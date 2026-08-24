#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export SEMANTIC_ONLY="true"
export PARSER_RUN_PREFIX="${PARSER_RUN_PREFIX:-dense_uv_semantic_v}"
export PARSER_CHECKPOINT_FALLBACK="false"

export OUTPUT=""
export CONDITIONING_OUTPUT=""
export PARSER_UV_OUTPUT=""
export SIMPLE_INPAINT_OUTPUT=""
export SIMPLE_INPAINT_RENDER_OUTPUT=""
export DEBUG_OUTPUT=""
export OVERLAY_OUTPUT=""
export INNER_CUTOUT_OUTPUT=""
export OUTER_CUTOUT_OUTPUT=""
export SECONDARY_CUTOUT_OUTPUT=""
export COLOR_SOURCE_OUTPUT=""
export FACE_OUTPUT=""
export LAYER_FACE_OUTPUT=""
export RAW_FACE_OUTPUT=""
export RAW_LAYER_FACE_OUTPUT=""
export CANONICAL_FOREGROUND_OUTPUT=""
export GEOMETRY_GRID_OUTPUT=""
export GEOMETRY_OVERLAY_OUTPUT=""
export GEOMETRY_ROUTED_OVERLAY_OUTPUT=""
export GEOMETRY_FILL_OUTPUT=""
export OUTER_UV_OCCUPANCY_OUTPUT=""
export HEAD_OUTER_STRUCTURE_OUTPUT=""
export HEAD_EYE_SEMANTIC_OUTER_UV_OUTPUT=""

export SEMANTIC_OUTPUT="${SEMANTIC_OUTPUT:-outputs/semantic_summary.json}"
export SEMANTIC_PIXEL_OUTPUT="${SEMANTIC_PIXEL_OUTPUT:-outputs/parser_debug_semantic_pixel_labels.png}"

exec ./run_infer.sh "$@"
