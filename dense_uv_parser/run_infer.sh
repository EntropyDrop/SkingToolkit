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
    [[ -d "$dir" ]] || continue
    [[ -f "$dir/$checkpoint_name" ]] || continue

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

PYTHON_BIN="${PYTHON_BIN:-python}"

PARSER_RUNS_DIR="${PARSER_RUNS_DIR:-runs}"
PARSER_RUN_PREFIX="${PARSER_RUN_PREFIX:-dense_uv_parser_v}"
PARSER_CHECKPOINT_NAME="${PARSER_CHECKPOINT_NAME:-best.pt}"
PARSER_CHECKPOINT="${PARSER_CHECKPOINT:-${CHECKPOINT:-}}"

if [[ -z "$PARSER_CHECKPOINT" ]]; then
  PARSER_CHECKPOINT="$(find_latest_checkpoint "$PARSER_RUNS_DIR" "$PARSER_RUN_PREFIX" "$PARSER_CHECKPOINT_NAME")"
fi
if [[ -z "$PARSER_CHECKPOINT" ]]; then
  echo "No parser checkpoint found under ${PARSER_RUNS_DIR}/${PARSER_RUN_PREFIX}*/${PARSER_CHECKPOINT_NAME}." >&2
  echo "Train one first with ./run_dense_uv_parser_training.sh or set PARSER_CHECKPOINT=/path/to/best.pt." >&2
  exit 1
fi

INPAINT_MODEL="${INPAINT_MODEL:-full}"
INPAINT_RUNS_DIR="${INPAINT_RUNS_DIR:-../inverse_uv/runs}"
INPAINT_RUN_PREFIX="${INPAINT_RUN_PREFIX:-inverse_uv_${INPAINT_MODEL}_v}"
INPAINT_CHECKPOINT_NAME="${INPAINT_CHECKPOINT_NAME:-best.pt}"
INPAINT_CHECKPOINT="${INPAINT_CHECKPOINT:-}"
if [[ "$INPAINT_CHECKPOINT" == "none" ]]; then
  INPAINT_CHECKPOINT=""
elif [[ -z "$INPAINT_CHECKPOINT" ]]; then
  INPAINT_CHECKPOINT="$(find_latest_checkpoint "$INPAINT_RUNS_DIR" "$INPAINT_RUN_PREFIX" "$INPAINT_CHECKPOINT_NAME")"
fi

OUTPUT_WAS_SET=false
if [[ ${OUTPUT+x} ]]; then
  OUTPUT_WAS_SET=true
fi
OUTPUT="${OUTPUT-outputs/pred_uv.png}"
CONDITIONING_OUTPUT="${CONDITIONING_OUTPUT-outputs/parser_conditioning.png}"
DEBUG_OUTPUT="${DEBUG_OUTPUT-outputs/parser_debug.png}"
OVERLAY_OUTPUT="${OVERLAY_OUTPUT-outputs/parser_debug_overlay.png}"
INNER_CUTOUT_OUTPUT="${INNER_CUTOUT_OUTPUT-outputs/parser_debug_inner.png}"
OUTER_CUTOUT_OUTPUT="${OUTER_CUTOUT_OUTPUT-outputs/parser_debug_outer.png}"
SECONDARY_CUTOUT_OUTPUT="${SECONDARY_CUTOUT_OUTPUT-outputs/parser_debug_secondary.png}"
FACE_OUTPUT="${FACE_OUTPUT-outputs/parser_debug_face.png}"
LAYER_FACE_OUTPUT="${LAYER_FACE_OUTPUT-outputs/parser_debug_layer_face.png}"
RAW_FACE_OUTPUT="${RAW_FACE_OUTPUT-outputs/parser_debug_face_raw.png}"
RAW_LAYER_FACE_OUTPUT="${RAW_LAYER_FACE_OUTPUT-outputs/parser_debug_layer_face_raw.png}"
GEOMETRY_GRID_OUTPUT="${GEOMETRY_GRID_OUTPUT-outputs/parser_debug_geometry_grid.png}"
GEOMETRY_FILL_OUTPUT="${GEOMETRY_FILL_OUTPUT-outputs/parser_debug_geometry_fill.png}"

COMBINED="${COMBINED:-}"
VIEW_IMAGES="${VIEW_IMAGES:-}"
FRONT="${FRONT:-../test_imgs/front_rgba.png}"
BACK="${BACK:-../test_imgs/back_rgba.png}"
MAPPINGS_DIR="${MAPPINGS_DIR:-}"
FG_THRESHOLD="${FG_THRESHOLD:-0.5}"
ROUTE_CONFIDENCE_THRESHOLD="${ROUTE_CONFIDENCE_THRESHOLD:-0.0}"
ROUTE_MARGIN_THRESHOLD="${ROUTE_MARGIN_THRESHOLD:-0.0}"
OUTER_ROUTE_CONFIDENCE_THRESHOLD="${OUTER_ROUTE_CONFIDENCE_THRESHOLD:-0.10}"
OUTER_ROUTE_MARGIN_THRESHOLD="${OUTER_ROUTE_MARGIN_THRESHOLD:-0.20}"
OUTER_UV_MIN_COVERAGE="${OUTER_UV_MIN_COVERAGE:-0.5}"
COLOR_AGGREGATION="${COLOR_AGGREGATION:-exact_mode}"
ALLOW_SEMANTIC_FALLBACK="${ALLOW_SEMANTIC_FALLBACK:-false}"
SEMANTIC_GATE="${SEMANTIC_GATE:-true}"
AFFINE_REFINE="${AFFINE_REFINE:-true}"
AFFINE_REFINE_TRANSLATION_PX="${AFFINE_REFINE_TRANSLATION_PX:-2.0}"
AFFINE_REFINE_SCALE="${AFFINE_REFINE_SCALE:-0.0}"
ALPHA_THRESHOLD="${ALPHA_THRESHOLD:-0.5}"
DEVICE="${DEVICE:-auto}"
NO_ENFORCE_BASE_ALPHA="${NO_ENFORCE_BASE_ALPHA:-false}"
OVERLAY_ALPHA="${OVERLAY_ALPHA:-0.45}"

args=(
  infer.py
  --parser_checkpoint "$PARSER_CHECKPOINT"
  --fg_threshold "$FG_THRESHOLD"
  --route_confidence_threshold "$ROUTE_CONFIDENCE_THRESHOLD"
  --route_margin_threshold "$ROUTE_MARGIN_THRESHOLD"
  --outer_route_confidence_threshold "$OUTER_ROUTE_CONFIDENCE_THRESHOLD"
  --outer_route_margin_threshold "$OUTER_ROUTE_MARGIN_THRESHOLD"
  --outer_uv_min_coverage "$OUTER_UV_MIN_COVERAGE"
  --color_aggregation "$COLOR_AGGREGATION"
  --affine_refine_translation_px "$AFFINE_REFINE_TRANSLATION_PX"
  --affine_refine_scale "$AFFINE_REFINE_SCALE"
  --alpha_threshold "$ALPHA_THRESHOLD"
  --device "$DEVICE"
)

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

if [[ -n "$GEOMETRY_FILL_OUTPUT" ]]; then
  args+=(--geometry_fill_output "$GEOMETRY_FILL_OUTPUT")
fi

if [[ -n "$OUTPUT" ]]; then
  if [[ -n "$INPAINT_CHECKPOINT" ]]; then
    args+=(--inpaint_checkpoint "$INPAINT_CHECKPOINT" --output "$OUTPUT")
  elif [[ "$OUTPUT_WAS_SET" == "true" ]]; then
    echo "OUTPUT was set, but no inpaint checkpoint was found." >&2
    echo "Set INPAINT_CHECKPOINT=/path/to/best.pt, or set OUTPUT= to write only parser conditioning." >&2
    exit 1
  else
    echo "No inverse_uv checkpoint found under ${INPAINT_RUNS_DIR}/${INPAINT_RUN_PREFIX}*/${INPAINT_CHECKPOINT_NAME}; writing conditioning only." >&2
  fi
fi

if [[ -z "$CONDITIONING_OUTPUT" && -z "$DEBUG_OUTPUT" && -z "$OVERLAY_OUTPUT" && -z "$INNER_CUTOUT_OUTPUT" && -z "$OUTER_CUTOUT_OUTPUT" && -z "$SECONDARY_CUTOUT_OUTPUT" && -z "$FACE_OUTPUT" && -z "$LAYER_FACE_OUTPUT" && -z "$RAW_FACE_OUTPUT" && -z "$RAW_LAYER_FACE_OUTPUT" && -z "$GEOMETRY_GRID_OUTPUT" && -z "$GEOMETRY_FILL_OUTPUT" && ( -z "$OUTPUT" || -z "$INPAINT_CHECKPOINT" ) ]]; then
  echo "Nothing to write. Set a debug/conditioning output or OUTPUT with a valid INPAINT_CHECKPOINT." >&2
  exit 1
fi

if [[ "$NO_ENFORCE_BASE_ALPHA" == "true" ]]; then
  args+=(--no_enforce_base_alpha)
fi

echo "Parser checkpoint: $PARSER_CHECKPOINT"
if [[ -n "$INPAINT_CHECKPOINT" ]]; then
  echo "Inpaint checkpoint: $INPAINT_CHECKPOINT"
fi
if [[ -n "$CONDITIONING_OUTPUT" ]]; then
  echo "Conditioning output: $CONDITIONING_OUTPUT"
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
  echo "Secondary/backface output: $SECONDARY_CUTOUT_OUTPUT"
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
if [[ -n "$GEOMETRY_FILL_OUTPUT" ]]; then
  echo "Geometry fill output: $GEOMETRY_FILL_OUTPUT"
fi
if [[ -n "$OUTPUT" && -n "$INPAINT_CHECKPOINT" ]]; then
  echo "Final output: $OUTPUT"
fi

"$PYTHON_BIN" "${args[@]}" "$@"
