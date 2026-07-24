#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

find_latest_checkpoint() {
  local root="$1"
  local prefix="$2"
  local checkpoint_name="$3"
  local best_v=-1
  local best_checkpoint=""
  local dir base suffix v

  shopt -s nullglob
  for dir in "$root"/"${prefix}"*; do
    [[ -d "$dir" && -f "$dir/$checkpoint_name" ]] || continue
    base="$(basename "$dir")"
    suffix="${base#"$prefix"}"
    [[ "$suffix" =~ ^[0-9]+$ ]] || continue
    v=$((10#$suffix))
    if (( v > best_v )); then
      best_v="$v"
      best_checkpoint="$dir/$checkpoint_name"
    fi
  done
  shopt -u nullglob

  printf '%s\n' "$best_checkpoint"
}

find_latest_family_checkpoint() {
  local root="$1"
  local family="$2"
  local checkpoint_name="$3"
  local best_v=-1
  local best_checkpoint=""
  local dir base v

  shopt -s nullglob
  for dir in "$root"/"${family}"*; do
    [[ -d "$dir" && -f "$dir/$checkpoint_name" ]] || continue
    base="$(basename "$dir")"
    if [[ "$base" =~ _v([0-9]+)$ ]]; then
      v=$((10#${BASH_REMATCH[1]}))
      if (( v > best_v )); then
        best_v="$v"
        best_checkpoint="$dir/$checkpoint_name"
      fi
    elif [[ -z "$best_checkpoint" ]]; then
      best_checkpoint="$dir/$checkpoint_name"
    elif [[ "$dir/$checkpoint_name" -nt "$best_checkpoint" && $best_v -lt 0 ]]; then
      best_checkpoint="$dir/$checkpoint_name"
    fi
  done
  shopt -u nullglob

  printf '%s\n' "$best_checkpoint"
}

PYTHON_BIN="${PYTHON_BIN:-python}"
PARSER_ONLY="${PARSER_ONLY:-false}"

PARSER_RUNS_DIR="${PARSER_RUNS_DIR:-runs}"
PARSER_RUN_PREFIX="${PARSER_RUN_PREFIX:-dense_uv_parser_v}"
PARSER_CHECKPOINT_NAME="${PARSER_CHECKPOINT_NAME:-best.pt}"
PARSER_CHECKPOINT="${PARSER_CHECKPOINT:-${CHECKPOINT:-}}"

if [[ -z "$PARSER_CHECKPOINT" ]]; then
  PARSER_CHECKPOINT="$(find_latest_checkpoint "$PARSER_RUNS_DIR" "$PARSER_RUN_PREFIX" "$PARSER_CHECKPOINT_NAME")"
fi
if [[ -z "$PARSER_CHECKPOINT" ]]; then
  PARSER_CHECKPOINT="$(find_latest_family_checkpoint "$PARSER_RUNS_DIR" "dense_uv_parser" "$PARSER_CHECKPOINT_NAME")"
fi
if [[ -z "$PARSER_CHECKPOINT" && "$PARSER_CHECKPOINT_NAME" != "latest.pt" ]]; then
  PARSER_CHECKPOINT="$(find_latest_family_checkpoint "$PARSER_RUNS_DIR" "dense_uv_parser" "latest.pt")"
fi
if [[ -z "$PARSER_CHECKPOINT" ]]; then
  echo "No parser checkpoint found in $PARSER_RUNS_DIR." >&2
  echo "Train one first with ./run_dense_uv_parser_training.sh or set PARSER_CHECKPOINT=/path/to/best.pt." >&2
  exit 1
fi
if [[ ! -f "$PARSER_CHECKPOINT" ]]; then
  echo "Parser checkpoint not found: $PARSER_CHECKPOINT" >&2
  exit 1
fi

FOREGROUND_METHOD="${FOREGROUND_METHOD:-flood}"
case "$FOREGROUND_METHOD" in
  flood|legacy)
    ;;
  *)
    echo "Unknown FOREGROUND_METHOD=$FOREGROUND_METHOD; expected flood or legacy." >&2
    exit 1
    ;;
esac

echo "Using parser checkpoint: $PARSER_CHECKPOINT"
echo "Using foreground method: $FOREGROUND_METHOD"
if [[ "$FOREGROUND_METHOD" == "flood" ]]; then
  echo "Using top-left pixel color flood-fill background removal."
else
  echo "Using legacy background removal inside UV routing."
fi
OUTPUT="${OUTPUT-outputs/pred_uv.png}"
if [[ "$PARSER_ONLY" == "true" ]]; then
  OUTPUT=""
fi
CONDITIONING_OUTPUT="${CONDITIONING_OUTPUT-outputs/parser_conditioning.png}"
PARSER_UV_OUTPUT_WAS_SET=false
if [[ ${PARSER_UV_OUTPUT+x} ]]; then
  PARSER_UV_OUTPUT_WAS_SET=true
fi
PARSER_UV_OUTPUT="${PARSER_UV_OUTPUT-outputs/parser_pred_uv.png}"
if [[ "$PARSER_ONLY" == "true" && "$PARSER_UV_OUTPUT_WAS_SET" == "false" ]]; then
  PARSER_UV_OUTPUT="outputs/parser_only_uv.png"
fi
SIMPLE_INPAINT_OUTPUT="${SIMPLE_INPAINT_OUTPUT-outputs/parser_pred_uv_simple_inpainting.png}"
if [[ "$PARSER_ONLY" == "true" ]]; then
  SIMPLE_INPAINT_OUTPUT=""
fi
DEBUG_OUTPUT="${DEBUG_OUTPUT-outputs/parser_debug.png}"
OVERLAY_OUTPUT="${OVERLAY_OUTPUT-outputs/parser_debug_overlay.png}"
INNER_CUTOUT_OUTPUT="${INNER_CUTOUT_OUTPUT-outputs/parser_debug_inner.png}"
OUTER_CUTOUT_OUTPUT="${OUTER_CUTOUT_OUTPUT-outputs/parser_debug_outer.png}"
SECONDARY_CUTOUT_OUTPUT="${SECONDARY_CUTOUT_OUTPUT-outputs/parser_debug_secondary.png}"
COLOR_SOURCE_OUTPUT="${COLOR_SOURCE_OUTPUT-outputs/parser_debug_color_source.png}"
FACE_OUTPUT="${FACE_OUTPUT-outputs/parser_debug_face.png}"
LAYER_FACE_OUTPUT="${LAYER_FACE_OUTPUT-outputs/parser_debug_layer_face.png}"
RAW_FACE_OUTPUT="${RAW_FACE_OUTPUT-outputs/parser_debug_face_raw.png}"
RAW_LAYER_FACE_OUTPUT="${RAW_LAYER_FACE_OUTPUT-outputs/parser_debug_layer_face_raw.png}"
GEOMETRY_GRID_OUTPUT="${GEOMETRY_GRID_OUTPUT-outputs/parser_debug_geometry_grid.png}"
GEOMETRY_OVERLAY_OUTPUT="${GEOMETRY_OVERLAY_OUTPUT-outputs/parser_debug_geometry_overlay.png}"
GEOMETRY_ROUTED_OVERLAY_OUTPUT="${GEOMETRY_ROUTED_OVERLAY_OUTPUT-outputs/parser_debug_geometry_routed_overlay.png}"
GEOMETRY_FILL_OUTPUT="${GEOMETRY_FILL_OUTPUT-outputs/parser_debug_geometry_fill.png}"
FOREGROUND_PROBABILITY_OUTPUT="${FOREGROUND_PROBABILITY_OUTPUT-outputs/foreground_probability.png}"
FOREGROUND_MASK_OUTPUT="${FOREGROUND_MASK_OUTPUT-outputs/foreground_mask.png}"
FOREGROUND_RAW_MASK_OUTPUT="${FOREGROUND_RAW_MASK_OUTPUT-outputs/foreground_mask_raw.png}"
FOREGROUND_CUTOUT_OUTPUT="${FOREGROUND_CUTOUT_OUTPUT-outputs/foreground_cutout.png}"
FOREGROUND_PARSER_INPUT_OUTPUT="${FOREGROUND_PARSER_INPUT_OUTPUT-outputs/foreground_parser_input.png}"

COMBINED="${COMBINED:-}"
VIEW_IMAGES="${VIEW_IMAGES:-}"
FRONT="${FRONT:-../test_imgs/front_rgba.png}"
BACK="${BACK:-../test_imgs/back_rgba.png}"
MAPPINGS_DIR="${MAPPINGS_DIR:-}"
ROUTING_PROFILE="${ROUTING_PROFILE:-conservative}"
case "$ROUTING_PROFILE" in
  balanced)
    DEFAULT_BACKGROUND_COLOR_TOLERANCE="0.1882352941"
    DEFAULT_ROUTE_CONFIDENCE_THRESHOLD="0.0"
    DEFAULT_ROUTE_MARGIN_THRESHOLD="0.0"
    DEFAULT_OUTER_ROUTE_CONFIDENCE_THRESHOLD="0.55"
    DEFAULT_OUTER_ROUTE_MARGIN_THRESHOLD="0.35"
    DEFAULT_OUTER_UV_MIN_COVERAGE="0.0"
    DEFAULT_OUTER_UV_MIN_SOURCE_PIXELS="15"
    DEFAULT_OUTER_GEOMETRY_RESCUE="false"
    DEFAULT_OUTER_SEMANTIC_RESCUE="true"
    DEFAULT_GEOMETRY_ROUTE_TEXEL_CONSENSUS="false"
    ;;
  conservative)
    # Prefer missing evidence over persistent wrong-color parser evidence.
    # Deterministic repair fills inner-layer holes only; rejected outer texels
    # intentionally remain transparent instead of being hallucinated.
    DEFAULT_BACKGROUND_COLOR_TOLERANCE="0.25"
    DEFAULT_ROUTE_CONFIDENCE_THRESHOLD="0.0"
    DEFAULT_ROUTE_MARGIN_THRESHOLD="0.0"
    DEFAULT_OUTER_ROUTE_CONFIDENCE_THRESHOLD="0.80"
    DEFAULT_OUTER_ROUTE_MARGIN_THRESHOLD="0.55"
    DEFAULT_OUTER_UV_MIN_COVERAGE="0.25"
    DEFAULT_OUTER_UV_MIN_SOURCE_PIXELS="15"
    DEFAULT_OUTER_GEOMETRY_RESCUE="true"
    DEFAULT_OUTER_SEMANTIC_RESCUE="true"
    DEFAULT_GEOMETRY_ROUTE_TEXEL_CONSENSUS="true"
    ;;
  *)
    echo "Unknown ROUTING_PROFILE=$ROUTING_PROFILE; expected balanced or conservative." >&2
    exit 1
    ;;
esac
FG_THRESHOLD="${FG_THRESHOLD:-0.5}"
FOREGROUND_FLOOD_TOLERANCE="${FOREGROUND_FLOOD_TOLERANCE:-0.03}"
COLOR_BACKGROUND_TOLERANCE="${COLOR_BACKGROUND_TOLERANCE:-0.031372549}"
COLOR_FOREGROUND_INSET="${COLOR_FOREGROUND_INSET:-1}"
FOREGROUND_PARSER_BACKGROUND="${FOREGROUND_PARSER_BACKGROUND:-adaptive}"
BACKGROUND_COLOR_TOLERANCE="${BACKGROUND_COLOR_TOLERANCE:-$DEFAULT_BACKGROUND_COLOR_TOLERANCE}"
ROUTE_CONFIDENCE_THRESHOLD="${ROUTE_CONFIDENCE_THRESHOLD:-$DEFAULT_ROUTE_CONFIDENCE_THRESHOLD}"
ROUTE_MARGIN_THRESHOLD="${ROUTE_MARGIN_THRESHOLD:-$DEFAULT_ROUTE_MARGIN_THRESHOLD}"
OUTER_ROUTE_CONFIDENCE_THRESHOLD="${OUTER_ROUTE_CONFIDENCE_THRESHOLD:-$DEFAULT_OUTER_ROUTE_CONFIDENCE_THRESHOLD}"
OUTER_ROUTE_MARGIN_THRESHOLD="${OUTER_ROUTE_MARGIN_THRESHOLD:-$DEFAULT_OUTER_ROUTE_MARGIN_THRESHOLD}"
OUTER_UV_MIN_COVERAGE="${OUTER_UV_MIN_COVERAGE:-$DEFAULT_OUTER_UV_MIN_COVERAGE}"
OUTER_UV_MIN_SOURCE_PIXELS="${OUTER_UV_MIN_SOURCE_PIXELS:-$DEFAULT_OUTER_UV_MIN_SOURCE_PIXELS}"
OUTER_GEOMETRY_RESCUE="${OUTER_GEOMETRY_RESCUE:-$DEFAULT_OUTER_GEOMETRY_RESCUE}"
OUTER_SEMANTIC_RESCUE="${OUTER_SEMANTIC_RESCUE:-$DEFAULT_OUTER_SEMANTIC_RESCUE}"
OUTER_SEMANTIC_PRESENCE_THRESHOLD="${OUTER_SEMANTIC_PRESENCE_THRESHOLD:-0.80}"
OUTER_SEMANTIC_COVERAGE_THRESHOLD="${OUTER_SEMANTIC_COVERAGE_THRESHOLD:-0.20}"
OUTER_RESCUE_CONFIDENCE_THRESHOLD="${OUTER_RESCUE_CONFIDENCE_THRESHOLD:-0.60}"
OUTER_RESCUE_MARGIN_THRESHOLD="${OUTER_RESCUE_MARGIN_THRESHOLD:-0.25}"
OUTER_RESCUE_MIN_COVERAGE="${OUTER_RESCUE_MIN_COVERAGE:-0.10}"
GEOMETRY_ROUTE_TEXEL_CONSENSUS="${GEOMETRY_ROUTE_TEXEL_CONSENSUS:-$DEFAULT_GEOMETRY_ROUTE_TEXEL_CONSENSUS}"
GEOMETRY_ROUTE_TEXEL_CONSENSUS_WEIGHT="${GEOMETRY_ROUTE_TEXEL_CONSENSUS_WEIGHT:-0.60}"
GEOMETRY_ROUTE_PRESERVE_OUTER_CONFIDENCE="${GEOMETRY_ROUTE_PRESERVE_OUTER_CONFIDENCE:-0.80}"
GEOMETRY_ROUTE_PRESERVE_OUTER_MARGIN="${GEOMETRY_ROUTE_PRESERVE_OUTER_MARGIN:-0.35}"
GEOMETRY_ROUTE_CONSENSUS_OUTER_CONFIDENCE="${GEOMETRY_ROUTE_CONSENSUS_OUTER_CONFIDENCE:-0.70}"
GEOMETRY_ROUTE_CONSENSUS_OUTER_MARGIN="${GEOMETRY_ROUTE_CONSENSUS_OUTER_MARGIN:-0.20}"
OUTER_UV_OCCUPANCY="${OUTER_UV_OCCUPANCY:-false}"
OUTER_UV_OCCUPANCY_BLEND_WEIGHT="${OUTER_UV_OCCUPANCY_BLEND_WEIGHT:-0.30}"
OUTER_UV_OCCUPANCY_GATE_THRESHOLD="${OUTER_UV_OCCUPANCY_GATE_THRESHOLD:-0.10}"
OUTER_UV_OCCUPANCY_RESCUE_THRESHOLD="${OUTER_UV_OCCUPANCY_RESCUE_THRESHOLD:-0.70}"
OUTER_UV_OCCUPANCY_RESCUE_ROUTE_THRESHOLD="${OUTER_UV_OCCUPANCY_RESCUE_ROUTE_THRESHOLD:-0.30}"
COLOR_AGGREGATION="${COLOR_AGGREGATION:-grid_mode}"
ALLOW_SEMANTIC_FALLBACK="${ALLOW_SEMANTIC_FALLBACK:-false}"
SEMANTIC_GATE="${SEMANTIC_GATE:-true}"
AFFINE_REFINE="${AFFINE_REFINE:-false}"
AFFINE_REFINE_TRANSLATION_PX="${AFFINE_REFINE_TRANSLATION_PX:-0.0}"
AFFINE_REFINE_SCALE="${AFFINE_REFINE_SCALE:-0.0}"
ALPHA_THRESHOLD="${ALPHA_THRESHOLD:-0.5}"
DEVICE="${DEVICE:-auto}"
OVERLAY_ALPHA="${OVERLAY_ALPHA:-0.45}"

echo "Using routing profile: $ROUTING_PROFILE"
echo "Using grid color aggregation: $COLOR_AGGREGATION"
if [[ "$PARSER_ONLY" == "true" ]]; then
  echo "Parser-only mode: deterministic UV repair is disabled."
fi

args=(
  infer.py
  --parser_checkpoint "$PARSER_CHECKPOINT"
  --foreground_method "$FOREGROUND_METHOD"
  --foreground_flood_tolerance "$FOREGROUND_FLOOD_TOLERANCE"
  --color_background_tolerance "$COLOR_BACKGROUND_TOLERANCE"
  --color_foreground_inset "$COLOR_FOREGROUND_INSET"
  --foreground_parser_background "$FOREGROUND_PARSER_BACKGROUND"
  --fg_threshold "$FG_THRESHOLD"
  --background_color_tolerance "$BACKGROUND_COLOR_TOLERANCE"
  --route_confidence_threshold "$ROUTE_CONFIDENCE_THRESHOLD"
  --route_margin_threshold "$ROUTE_MARGIN_THRESHOLD"
  --outer_route_confidence_threshold "$OUTER_ROUTE_CONFIDENCE_THRESHOLD"
  --outer_route_margin_threshold "$OUTER_ROUTE_MARGIN_THRESHOLD"
  --outer_uv_min_coverage "$OUTER_UV_MIN_COVERAGE"
  --outer_uv_min_source_pixels "$OUTER_UV_MIN_SOURCE_PIXELS"
  --outer_semantic_presence_threshold "$OUTER_SEMANTIC_PRESENCE_THRESHOLD"
  --outer_semantic_coverage_threshold "$OUTER_SEMANTIC_COVERAGE_THRESHOLD"
  --outer_rescue_confidence_threshold "$OUTER_RESCUE_CONFIDENCE_THRESHOLD"
  --outer_rescue_margin_threshold "$OUTER_RESCUE_MARGIN_THRESHOLD"
  --outer_rescue_min_coverage "$OUTER_RESCUE_MIN_COVERAGE"
  --geometry_route_texel_consensus_weight "$GEOMETRY_ROUTE_TEXEL_CONSENSUS_WEIGHT"
  --geometry_route_preserve_outer_confidence "$GEOMETRY_ROUTE_PRESERVE_OUTER_CONFIDENCE"
  --geometry_route_preserve_outer_margin "$GEOMETRY_ROUTE_PRESERVE_OUTER_MARGIN"
  --geometry_route_consensus_outer_confidence "$GEOMETRY_ROUTE_CONSENSUS_OUTER_CONFIDENCE"
  --geometry_route_consensus_outer_margin "$GEOMETRY_ROUTE_CONSENSUS_OUTER_MARGIN"
  --outer_uv_occupancy_blend_weight "$OUTER_UV_OCCUPANCY_BLEND_WEIGHT"
  --outer_uv_occupancy_gate_threshold "$OUTER_UV_OCCUPANCY_GATE_THRESHOLD"
  --outer_uv_occupancy_rescue_threshold "$OUTER_UV_OCCUPANCY_RESCUE_THRESHOLD"
  --outer_uv_occupancy_rescue_route_threshold "$OUTER_UV_OCCUPANCY_RESCUE_ROUTE_THRESHOLD"
  --color_aggregation "$COLOR_AGGREGATION"
  --affine_refine_translation_px "$AFFINE_REFINE_TRANSLATION_PX"
  --affine_refine_scale "$AFFINE_REFINE_SCALE"
  --alpha_threshold "$ALPHA_THRESHOLD"
  --device "$DEVICE"
)

if [[ -n "$FOREGROUND_PROBABILITY_OUTPUT" ]]; then
  args+=(--foreground_probability_output "$FOREGROUND_PROBABILITY_OUTPUT")
fi
if [[ -n "$FOREGROUND_MASK_OUTPUT" ]]; then
  args+=(--foreground_mask_output "$FOREGROUND_MASK_OUTPUT")
fi
if [[ -n "$FOREGROUND_RAW_MASK_OUTPUT" ]]; then
  args+=(--foreground_raw_mask_output "$FOREGROUND_RAW_MASK_OUTPUT")
fi
if [[ -n "$FOREGROUND_CUTOUT_OUTPUT" ]]; then
  args+=(--foreground_cutout_output "$FOREGROUND_CUTOUT_OUTPUT")
fi
if [[ -n "$FOREGROUND_PARSER_INPUT_OUTPUT" ]]; then
  args+=(--foreground_parser_input_output "$FOREGROUND_PARSER_INPUT_OUTPUT")
fi

if [[ "$GEOMETRY_ROUTE_TEXEL_CONSENSUS" == "true" ]]; then
  args+=(--geometry_route_texel_consensus)
else
  args+=(--no_geometry_route_texel_consensus)
fi

if [[ "$OUTER_UV_OCCUPANCY" == "true" ]]; then
  args+=(--outer_uv_occupancy)
else
  args+=(--no_outer_uv_occupancy)
fi

if [[ "$OUTER_GEOMETRY_RESCUE" == "true" ]]; then
  args+=(--outer_geometry_rescue)
else
  args+=(--no_outer_geometry_rescue)
fi

if [[ "$OUTER_SEMANTIC_RESCUE" == "true" ]]; then
  args+=(--outer_semantic_rescue)
else
  args+=(--no_outer_semantic_rescue)
fi

if [[ "$ALLOW_SEMANTIC_FALLBACK" == "true" ]]; then
  args+=(--allow_semantic_fallback)
fi

if [[ -n "$MAPPINGS_DIR" ]]; then
  args+=(--mappings_dir "$MAPPINGS_DIR")
fi
if [[ "$SEMANTIC_GATE" != "true" ]]; then
  args+=(--no_semantic_gate)
fi
if [[ "$AFFINE_REFINE" == "true" ]]; then
  args+=(--affine_refine)
else
  args+=(--no_affine_refine)
fi

if [[ -n "$COMBINED" ]]; then
  args+=(--combined "$COMBINED")
elif [[ -n "$VIEW_IMAGES" ]]; then
  read -r -a view_images_array <<< "$VIEW_IMAGES"
  args+=(--view_images "${view_images_array[@]}")
else
  args+=(--view_images "$FRONT" "$BACK")
fi

if [[ -n "$CONDITIONING_OUTPUT" ]]; then
  args+=(--conditioning_output "$CONDITIONING_OUTPUT")
fi

if [[ -n "$PARSER_UV_OUTPUT" ]]; then
  args+=(--parser_uv_output "$PARSER_UV_OUTPUT")
fi

if [[ -n "$SIMPLE_INPAINT_OUTPUT" ]]; then
  args+=(--simple_inpaint_output "$SIMPLE_INPAINT_OUTPUT")
fi

if [[ -n "$DEBUG_OUTPUT" ]]; then
  args+=(--debug_output "$DEBUG_OUTPUT")
fi

if [[ -n "$OVERLAY_OUTPUT" ]]; then
  args+=(--overlay_output "$OVERLAY_OUTPUT" --overlay_alpha "$OVERLAY_ALPHA")
fi

if [[ -n "$INNER_CUTOUT_OUTPUT" ]]; then
  args+=(--inner_cutout_output "$INNER_CUTOUT_OUTPUT")
fi

if [[ -n "$OUTER_CUTOUT_OUTPUT" ]]; then
  args+=(--outer_cutout_output "$OUTER_CUTOUT_OUTPUT")
fi
if [[ -n "$SECONDARY_CUTOUT_OUTPUT" ]]; then
  args+=(--secondary_cutout_output "$SECONDARY_CUTOUT_OUTPUT")
fi

if [[ -n "$COLOR_SOURCE_OUTPUT" ]]; then
  args+=(--color_source_output "$COLOR_SOURCE_OUTPUT")
fi

if [[ -n "$FACE_OUTPUT" ]]; then
  args+=(--face_output "$FACE_OUTPUT")
fi

if [[ -n "$LAYER_FACE_OUTPUT" ]]; then
  args+=(--layer_face_output "$LAYER_FACE_OUTPUT")
fi

if [[ -n "$RAW_FACE_OUTPUT" ]]; then
  args+=(--raw_face_output "$RAW_FACE_OUTPUT")
fi

if [[ -n "$RAW_LAYER_FACE_OUTPUT" ]]; then
  args+=(--raw_layer_face_output "$RAW_LAYER_FACE_OUTPUT")
fi

if [[ -n "$GEOMETRY_GRID_OUTPUT" ]]; then
  args+=(--geometry_grid_output "$GEOMETRY_GRID_OUTPUT")
fi

if [[ -n "$GEOMETRY_OVERLAY_OUTPUT" ]]; then
  args+=(--geometry_overlay_output "$GEOMETRY_OVERLAY_OUTPUT")
fi

if [[ -n "$GEOMETRY_ROUTED_OVERLAY_OUTPUT" ]]; then
  args+=(--geometry_routed_overlay_output "$GEOMETRY_ROUTED_OVERLAY_OUTPUT")
fi

if [[ -n "$GEOMETRY_FILL_OUTPUT" ]]; then
  args+=(--geometry_fill_output "$GEOMETRY_FILL_OUTPUT")
fi

if [[ -n "$OUTPUT" ]]; then
  args+=(--output "$OUTPUT")
fi

if [[ -z "$CONDITIONING_OUTPUT" && -z "$PARSER_UV_OUTPUT" && -z "$SIMPLE_INPAINT_OUTPUT" && -z "$DEBUG_OUTPUT" && -z "$OVERLAY_OUTPUT" && -z "$INNER_CUTOUT_OUTPUT" && -z "$OUTER_CUTOUT_OUTPUT" && -z "$SECONDARY_CUTOUT_OUTPUT" && -z "$COLOR_SOURCE_OUTPUT" && -z "$FACE_OUTPUT" && -z "$LAYER_FACE_OUTPUT" && -z "$RAW_FACE_OUTPUT" && -z "$RAW_LAYER_FACE_OUTPUT" && -z "$GEOMETRY_GRID_OUTPUT" && -z "$GEOMETRY_OVERLAY_OUTPUT" && -z "$GEOMETRY_ROUTED_OVERLAY_OUTPUT" && -z "$GEOMETRY_FILL_OUTPUT" && -z "$OUTPUT" ]]; then
  echo "Nothing to write. Set at least one parser/debug/final output." >&2
  exit 1
fi

echo "Parser checkpoint: $PARSER_CHECKPOINT"
echo "Foreground method: $FOREGROUND_METHOD"
if [[ "$FOREGROUND_METHOD" != "legacy" ]]; then
  echo "Foreground flood tolerance: $FOREGROUND_FLOOD_TOLERANCE"
  if [[ -n "$FOREGROUND_PROBABILITY_OUTPUT" ]]; then
    echo "Foreground probability output: $FOREGROUND_PROBABILITY_OUTPUT"
  fi
  if [[ -n "$FOREGROUND_MASK_OUTPUT" ]]; then
    echo "Foreground mask output: $FOREGROUND_MASK_OUTPUT"
  fi
  if [[ -n "$FOREGROUND_RAW_MASK_OUTPUT" ]]; then
    echo "Foreground raw mask output: $FOREGROUND_RAW_MASK_OUTPUT"
  fi
  if [[ -n "$FOREGROUND_CUTOUT_OUTPUT" ]]; then
    echo "Foreground cutout output: $FOREGROUND_CUTOUT_OUTPUT"
  fi
  if [[ -n "$FOREGROUND_PARSER_INPUT_OUTPUT" ]]; then
    echo "Foreground parser input: $FOREGROUND_PARSER_INPUT_OUTPUT"
  fi
fi
if [[ -n "$CONDITIONING_OUTPUT" ]]; then
  echo "Conditioning output: $CONDITIONING_OUTPUT"
fi
if [[ -n "$PARSER_UV_OUTPUT" ]]; then
  echo "Partial parser UV output: $PARSER_UV_OUTPUT"
fi
if [[ -n "$SIMPLE_INPAINT_OUTPUT" ]]; then
  echo "Simple parser UV repair output: $SIMPLE_INPAINT_OUTPUT"
fi
if [[ -n "$DEBUG_OUTPUT" ]]; then
  echo "Debug output: $DEBUG_OUTPUT"
fi
if [[ -n "$OVERLAY_OUTPUT" ]]; then
  echo "Overlay output: $OVERLAY_OUTPUT"
fi
if [[ -n "$INNER_CUTOUT_OUTPUT" ]]; then
  echo "Inner cutout output: $INNER_CUTOUT_OUTPUT"
fi
if [[ -n "$OUTER_CUTOUT_OUTPUT" ]]; then
  echo "Outer cutout output: $OUTER_CUTOUT_OUTPUT"
fi
if [[ -n "$SECONDARY_CUTOUT_OUTPUT" ]]; then
  echo "Secondary/deeper-surface output: $SECONDARY_CUTOUT_OUTPUT"
fi
if [[ -n "$COLOR_SOURCE_OUTPUT" ]]; then
  echo "Color-safe source output: $COLOR_SOURCE_OUTPUT"
fi
if [[ -n "$FACE_OUTPUT" ]]; then
  echo "Face output: $FACE_OUTPUT"
fi
if [[ -n "$LAYER_FACE_OUTPUT" ]]; then
  echo "Layer-face output: $LAYER_FACE_OUTPUT"
fi
if [[ -n "$RAW_FACE_OUTPUT" ]]; then
  echo "Raw face output: $RAW_FACE_OUTPUT"
fi
if [[ -n "$RAW_LAYER_FACE_OUTPUT" ]]; then
  echo "Raw layer-face output: $RAW_LAYER_FACE_OUTPUT"
fi
if [[ -n "$GEOMETRY_GRID_OUTPUT" ]]; then
  echo "Geometry grid output: $GEOMETRY_GRID_OUTPUT"
fi
if [[ -n "$GEOMETRY_OVERLAY_OUTPUT" ]]; then
  echo "Geometry overlay output: $GEOMETRY_OVERLAY_OUTPUT"
fi
if [[ -n "$GEOMETRY_ROUTED_OVERLAY_OUTPUT" ]]; then
  echo "Geometry routed overlay output: $GEOMETRY_ROUTED_OVERLAY_OUTPUT"
fi
if [[ -n "$GEOMETRY_FILL_OUTPUT" ]]; then
  echo "Geometry fill output: $GEOMETRY_FILL_OUTPUT"
fi
if [[ -n "$OUTPUT" ]]; then
  echo "Final output: $OUTPUT"
fi

"$PYTHON_BIN" "${args[@]}" "$@"
