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

COMBINED="${COMBINED:-}"
VIEW_IMAGES="${VIEW_IMAGES:-}"
FRONT="${FRONT:-../test_imgs/front_rgba.png}"
BACK="${BACK:-../test_imgs/back_rgba.png}"
MAPPINGS_DIR="${MAPPINGS_DIR:-}"
FG_THRESHOLD="${FG_THRESHOLD:-0.5}"
ALPHA_THRESHOLD="${ALPHA_THRESHOLD:-0.5}"
DEVICE="${DEVICE:-auto}"
NO_ENFORCE_BASE_ALPHA="${NO_ENFORCE_BASE_ALPHA:-false}"

args=(
  infer.py
  --parser_checkpoint "$PARSER_CHECKPOINT"
  --fg_threshold "$FG_THRESHOLD"
  --alpha_threshold "$ALPHA_THRESHOLD"
  --device "$DEVICE"
)

if [[ -n "$MAPPINGS_DIR" ]]; then
  args+=(--mappings_dir "$MAPPINGS_DIR")
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

if [[ -z "$CONDITIONING_OUTPUT" && ( -z "$OUTPUT" || -z "$INPAINT_CHECKPOINT" ) ]]; then
  echo "Nothing to write. Set CONDITIONING_OUTPUT and/or OUTPUT with a valid INPAINT_CHECKPOINT." >&2
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
if [[ -n "$OUTPUT" && -n "$INPAINT_CHECKPOINT" ]]; then
  echo "Final output: $OUTPUT"
fi

"$PYTHON_BIN" "${args[@]}" "$@"
