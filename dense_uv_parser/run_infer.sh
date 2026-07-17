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

parser_checkpoint_from_inpaint_config() {
  local inpaint_checkpoint="$1"
  local config_path
  local recorded
  local candidate

  [[ -n "$inpaint_checkpoint" ]] || return 0
  config_path="$(dirname "$inpaint_checkpoint")/config.json"
  [[ -f "$config_path" ]] || return 0
  recorded="$($PYTHON_BIN -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
print(
    config.get("metadata", {}).get("parser_checkpoint")
    or config.get("args", {}).get("parser_checkpoint")
    or ""
)
' "$config_path")"
  [[ -n "$recorded" ]] || return 0

  # Relative parser paths were recorded from semantic_uv_reconstruction/, while
  # this launcher runs from dense_uv_parser/. Try both working directories.
  for candidate in \
    "$recorded" \
    "../semantic_uv_reconstruction/$recorded" \
    "../$recorded"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
}

PYTHON_BIN="${PYTHON_BIN:-python}"

INPAINT_MODEL="${INPAINT_MODEL:-topology_maskgit}"
INPAINT_RUNS_DIR="${INPAINT_RUNS_DIR:-../semantic_uv_reconstruction/runs}"
INPAINT_RUN_PREFIX="${INPAINT_RUN_PREFIX:-semantic_uv_reconstruction_${INPAINT_MODEL}_v}"
INPAINT_CHECKPOINT_NAME="${INPAINT_CHECKPOINT_NAME:-best.pt}"
INPAINT_CHECKPOINT="${INPAINT_CHECKPOINT:-}"
if [[ "$INPAINT_CHECKPOINT" == "none" ]]; then
  INPAINT_CHECKPOINT=""
elif [[ -z "$INPAINT_CHECKPOINT" ]]; then
  INPAINT_CHECKPOINT="$(find_latest_checkpoint "$INPAINT_RUNS_DIR" "$INPAINT_RUN_PREFIX" "$INPAINT_CHECKPOINT_NAME")"
  if [[ -z "$INPAINT_CHECKPOINT" ]]; then
    INPAINT_CHECKPOINT="$(find_latest_family_checkpoint "$INPAINT_RUNS_DIR" "semantic_uv_reconstruction_${INPAINT_MODEL}" "$INPAINT_CHECKPOINT_NAME")"
  fi
fi
if [[ -n "$INPAINT_CHECKPOINT" && ! -f "$INPAINT_CHECKPOINT" ]]; then
  echo "Inpaint checkpoint not found: $INPAINT_CHECKPOINT" >&2
  exit 1
fi

PARSER_RUNS_DIR="${PARSER_RUNS_DIR:-runs}"
PARSER_RUN_PREFIX="${PARSER_RUN_PREFIX:-dense_uv_parser_v}"
PARSER_CHECKPOINT_NAME="${PARSER_CHECKPOINT_NAME:-best.pt}"
PARSER_CHECKPOINT="${PARSER_CHECKPOINT:-${CHECKPOINT:-}}"

if [[ -z "$PARSER_CHECKPOINT" ]]; then
  PARSER_CHECKPOINT="$(parser_checkpoint_from_inpaint_config "$INPAINT_CHECKPOINT")"
fi
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
  echo "No parser checkpoint found in $PARSER_RUNS_DIR or in the latest inpaint run config." >&2
  echo "Train one first with ./run_dense_uv_parser_training.sh or set PARSER_CHECKPOINT=/path/to/best.pt." >&2
  exit 1
fi
if [[ ! -f "$PARSER_CHECKPOINT" ]]; then
  echo "Parser checkpoint not found: $PARSER_CHECKPOINT" >&2
  exit 1
fi

echo "Using parser checkpoint: $PARSER_CHECKPOINT"
if [[ -n "$INPAINT_CHECKPOINT" ]]; then
  echo "Using inpaint checkpoint: $INPAINT_CHECKPOINT"
else
  echo "No inpaint checkpoint found; exporting parser conditioning only."
fi

OUTPUT_WAS_SET=false
if [[ ${OUTPUT+x} ]]; then
  OUTPUT_WAS_SET=true
fi
OUTPUT="${OUTPUT-outputs/pred_uv.png}"
CONDITIONING_OUTPUT="${CONDITIONING_OUTPUT-outputs/parser_conditioning.png}"
PARSER_UV_OUTPUT="${PARSER_UV_OUTPUT-outputs/parser_pred_uv.png}"
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
GEOMETRY_OVERLAY_OUTPUT="${GEOMETRY_OVERLAY_OUTPUT-outputs/parser_debug_geometry_overlay.png}"
GEOMETRY_ROUTED_OVERLAY_OUTPUT="${GEOMETRY_ROUTED_OVERLAY_OUTPUT-outputs/parser_debug_geometry_routed_overlay.png}"
GEOMETRY_FILL_OUTPUT="${GEOMETRY_FILL_OUTPUT-outputs/parser_debug_geometry_fill.png}"

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
    DEFAULT_OUTER_GEOMETRY_RESCUE="false"
    DEFAULT_GEOMETRY_ROUTE_TEXEL_CONSENSUS="false"
    ;;
  conservative)
    # Prefer holes that topology completion can regenerate over persistent
    # wrong-color parser evidence. Intended for rendered/generated inputs with
    # antialiased boundaries or small geometry mismatch.
    DEFAULT_BACKGROUND_COLOR_TOLERANCE="0.25"
    DEFAULT_ROUTE_CONFIDENCE_THRESHOLD="0.0"
    DEFAULT_ROUTE_MARGIN_THRESHOLD="0.0"
    DEFAULT_OUTER_ROUTE_CONFIDENCE_THRESHOLD="0.80"
    DEFAULT_OUTER_ROUTE_MARGIN_THRESHOLD="0.55"
    DEFAULT_OUTER_UV_MIN_COVERAGE="0.25"
    DEFAULT_OUTER_GEOMETRY_RESCUE="true"
    DEFAULT_GEOMETRY_ROUTE_TEXEL_CONSENSUS="true"
    ;;
  *)
    echo "Unknown ROUTING_PROFILE=$ROUTING_PROFILE; expected balanced or conservative." >&2
    exit 1
    ;;
esac
FG_THRESHOLD="${FG_THRESHOLD:-0.5}"
BACKGROUND_COLOR_TOLERANCE="${BACKGROUND_COLOR_TOLERANCE:-$DEFAULT_BACKGROUND_COLOR_TOLERANCE}"
ROUTE_CONFIDENCE_THRESHOLD="${ROUTE_CONFIDENCE_THRESHOLD:-$DEFAULT_ROUTE_CONFIDENCE_THRESHOLD}"
ROUTE_MARGIN_THRESHOLD="${ROUTE_MARGIN_THRESHOLD:-$DEFAULT_ROUTE_MARGIN_THRESHOLD}"
OUTER_ROUTE_CONFIDENCE_THRESHOLD="${OUTER_ROUTE_CONFIDENCE_THRESHOLD:-$DEFAULT_OUTER_ROUTE_CONFIDENCE_THRESHOLD}"
OUTER_ROUTE_MARGIN_THRESHOLD="${OUTER_ROUTE_MARGIN_THRESHOLD:-$DEFAULT_OUTER_ROUTE_MARGIN_THRESHOLD}"
OUTER_UV_MIN_COVERAGE="${OUTER_UV_MIN_COVERAGE:-$DEFAULT_OUTER_UV_MIN_COVERAGE}"
OUTER_GEOMETRY_RESCUE="${OUTER_GEOMETRY_RESCUE:-$DEFAULT_OUTER_GEOMETRY_RESCUE}"
OUTER_RESCUE_CONFIDENCE_THRESHOLD="${OUTER_RESCUE_CONFIDENCE_THRESHOLD:-0.60}"
OUTER_RESCUE_MARGIN_THRESHOLD="${OUTER_RESCUE_MARGIN_THRESHOLD:-0.25}"
OUTER_RESCUE_MIN_COVERAGE="${OUTER_RESCUE_MIN_COVERAGE:-0.10}"
GEOMETRY_ROUTE_TEXEL_CONSENSUS="${GEOMETRY_ROUTE_TEXEL_CONSENSUS:-$DEFAULT_GEOMETRY_ROUTE_TEXEL_CONSENSUS}"
COLOR_AGGREGATION="${COLOR_AGGREGATION:-exact_mode}"
ALLOW_SEMANTIC_FALLBACK="${ALLOW_SEMANTIC_FALLBACK:-false}"
SEMANTIC_GATE="${SEMANTIC_GATE:-true}"
AFFINE_REFINE="${AFFINE_REFINE:-false}"
AFFINE_REFINE_TRANSLATION_PX="${AFFINE_REFINE_TRANSLATION_PX:-0.0}"
AFFINE_REFINE_SCALE="${AFFINE_REFINE_SCALE:-0.0}"
ALPHA_THRESHOLD="${ALPHA_THRESHOLD:-0.5}"
DEVICE="${DEVICE:-auto}"
NO_ENFORCE_BASE_ALPHA="${NO_ENFORCE_BASE_ALPHA:-false}"
OVERLAY_ALPHA="${OVERLAY_ALPHA:-0.45}"
INPAINT_STEPS="${INPAINT_STEPS:-4}"
INPAINT_TEMPERATURE="${INPAINT_TEMPERATURE:-0.0}"
INPAINT_SEED="${INPAINT_SEED:-1234}"
INPAINT_PALETTE_SNAP="${INPAINT_PALETTE_SNAP:-true}"
INPAINT_PALETTE_MIN_CONFIDENCE="${INPAINT_PALETTE_MIN_CONFIDENCE:-0.75}"
INPAINT_EVIDENCE_LOCK_THRESHOLD="${INPAINT_EVIDENCE_LOCK_THRESHOLD:-0.0}"

echo "Using routing profile: $ROUTING_PROFILE"

args=(
  infer.py
  --parser_checkpoint "$PARSER_CHECKPOINT"
  --fg_threshold "$FG_THRESHOLD"
  --background_color_tolerance "$BACKGROUND_COLOR_TOLERANCE"
  --route_confidence_threshold "$ROUTE_CONFIDENCE_THRESHOLD"
  --route_margin_threshold "$ROUTE_MARGIN_THRESHOLD"
  --outer_route_confidence_threshold "$OUTER_ROUTE_CONFIDENCE_THRESHOLD"
  --outer_route_margin_threshold "$OUTER_ROUTE_MARGIN_THRESHOLD"
  --outer_uv_min_coverage "$OUTER_UV_MIN_COVERAGE"
  --outer_rescue_confidence_threshold "$OUTER_RESCUE_CONFIDENCE_THRESHOLD"
  --outer_rescue_margin_threshold "$OUTER_RESCUE_MARGIN_THRESHOLD"
  --outer_rescue_min_coverage "$OUTER_RESCUE_MIN_COVERAGE"
  --color_aggregation "$COLOR_AGGREGATION"
  --affine_refine_translation_px "$AFFINE_REFINE_TRANSLATION_PX"
  --affine_refine_scale "$AFFINE_REFINE_SCALE"
  --alpha_threshold "$ALPHA_THRESHOLD"
  --inpaint_steps "$INPAINT_STEPS"
  --inpaint_temperature "$INPAINT_TEMPERATURE"
  --inpaint_seed "$INPAINT_SEED"
  --inpaint_palette_min_confidence "$INPAINT_PALETTE_MIN_CONFIDENCE"
  --inpaint_evidence_lock_threshold "$INPAINT_EVIDENCE_LOCK_THRESHOLD"
  --device "$DEVICE"
)

if [[ "$INPAINT_PALETTE_SNAP" == "true" ]]; then
  args+=(--inpaint_palette_snap)
else
  args+=(--no_inpaint_palette_snap)
fi

if [[ "$GEOMETRY_ROUTE_TEXEL_CONSENSUS" == "true" ]]; then
  args+=(--geometry_route_texel_consensus)
else
  args+=(--no_geometry_route_texel_consensus)
fi

if [[ "$OUTER_GEOMETRY_RESCUE" == "true" ]]; then
  args+=(--outer_geometry_rescue)
else
  args+=(--no_outer_geometry_rescue)
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
  if [[ -n "$INPAINT_CHECKPOINT" ]]; then
    args+=(--inpaint_checkpoint "$INPAINT_CHECKPOINT" --output "$OUTPUT")
  elif [[ "$OUTPUT_WAS_SET" == "true" ]]; then
    echo "OUTPUT was set, but no inpaint checkpoint was found." >&2
    echo "Set INPAINT_CHECKPOINT=/path/to/best.pt, or set OUTPUT= to write only parser conditioning." >&2
    exit 1
  else
    echo "No semantic_uv_reconstruction checkpoint found under ${INPAINT_RUNS_DIR}/${INPAINT_RUN_PREFIX}*/${INPAINT_CHECKPOINT_NAME}; writing conditioning only." >&2
  fi
fi

if [[ -z "$CONDITIONING_OUTPUT" && -z "$PARSER_UV_OUTPUT" && -z "$DEBUG_OUTPUT" && -z "$OVERLAY_OUTPUT" && -z "$INNER_CUTOUT_OUTPUT" && -z "$OUTER_CUTOUT_OUTPUT" && -z "$SECONDARY_CUTOUT_OUTPUT" && -z "$FACE_OUTPUT" && -z "$LAYER_FACE_OUTPUT" && -z "$RAW_FACE_OUTPUT" && -z "$RAW_LAYER_FACE_OUTPUT" && -z "$GEOMETRY_GRID_OUTPUT" && -z "$GEOMETRY_OVERLAY_OUTPUT" && -z "$GEOMETRY_ROUTED_OVERLAY_OUTPUT" && -z "$GEOMETRY_FILL_OUTPUT" && ( -z "$OUTPUT" || -z "$INPAINT_CHECKPOINT" ) ]]; then
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
if [[ -n "$PARSER_UV_OUTPUT" ]]; then
  echo "Preliminary parser UV output: $PARSER_UV_OUTPUT"
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
if [[ -n "$OUTPUT" && -n "$INPAINT_CHECKPOINT" ]]; then
  echo "Final output: $OUTPUT"
fi

"$PYTHON_BIN" "${args[@]}" "$@"
