#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DATA_DIR="${DATA_DIR:-../../SKING_DDJ_Dataset/entropydrop_website_generations}"
PARSER_CHECKPOINT="${PARSER_CHECKPOINT:-runs/dense_uv_parser_v94/best.pt}"
RESULT_SUFFIX="${RESULT_SUFFIX:-_v94_result}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
LIMIT="${LIMIT:-0}"
FAIL_FAST="${FAIL_FAST:-true}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_PATH="${LOG_PATH:-runs/regenerate_v94_semantic_results.log}"

[[ -d "$DATA_DIR" ]] || { echo "DATA_DIR does not exist: $DATA_DIR" >&2; exit 1; }
[[ -f "$PARSER_CHECKPOINT" ]] || { echo "v94 checkpoint not found: $PARSER_CHECKPOINT" >&2; exit 1; }
[[ "$RESULT_SUFFIX" == _* && "$RESULT_SUFFIX" != */* ]] || {
  echo "RESULT_SUFFIX must start with '_' and contain no slash." >&2
  exit 1
}
[[ "$LIMIT" =~ ^[0-9]+$ ]] || { echo "LIMIT must be a non-negative integer." >&2; exit 1; }

mkdir -p "$(dirname "$LOG_PATH")"
total="$(find "$DATA_DIR" -type f -name '*_edited.png' | wc -l | tr -d ' ')"
echo "Regenerating strict v94 semantic labels: inputs=$total checkpoint=$PARSER_CHECKPOINT suffix=$RESULT_SUFFIX"
echo "Detailed inference logs: $LOG_PATH"

generated=0
skipped=0
failed=0
visited=0
while IFS= read -r -d '' source; do
  if (( LIMIT > 0 && visited >= LIMIT )); then
    break
  fi
  visited=$((visited + 1))
  base="${source%_edited.png}"
  result="${base}${RESULT_SUFFIX}.png"
  temporary="${base}.${RESULT_SUFFIX#_}.tmp.png"
  if [[ "$SKIP_EXISTING" == "true" && -f "$result" ]]; then
    skipped=$((skipped + 1))
    printf '\r[%d/%d] generated=%d skipped=%d failed=%d' "$visited" "$total" "$generated" "$skipped" "$failed"
    continue
  fi

  if PARSER_CHECKPOINT="$PARSER_CHECKPOINT" \
    PARSER_CHECKPOINT_FALLBACK=false \
    COMBINED="$source" \
    OUTPUT="$temporary" \
    CONDITIONING_OUTPUT="" \
    PARSER_UV_OUTPUT="" \
    SIMPLE_INPAINT_OUTPUT="" \
    SIMPLE_INPAINT_RENDER_OUTPUT="" \
    DEBUG_OUTPUT="" \
    OVERLAY_OUTPUT="" \
    INNER_CUTOUT_OUTPUT="" \
    OUTER_CUTOUT_OUTPUT="" \
    SECONDARY_CUTOUT_OUTPUT="" \
    COLOR_SOURCE_OUTPUT="" \
    FACE_OUTPUT="" \
    LAYER_FACE_OUTPUT="" \
    RAW_FACE_OUTPUT="" \
    RAW_LAYER_FACE_OUTPUT="" \
    CANONICAL_FOREGROUND_OUTPUT="" \
    GEOMETRY_GRID_OUTPUT="" \
    GEOMETRY_OVERLAY_OUTPUT="" \
    GEOMETRY_ROUTED_OVERLAY_OUTPUT="" \
    GEOMETRY_FILL_OUTPUT="" \
    OUTER_UV_OCCUPANCY_OUTPUT="" \
    HEAD_OUTER_STRUCTURE_OUTPUT="" \
    SEMANTIC_OUTPUT="" \
    SEMANTIC_PIXEL_OUTPUT="" \
    HEAD_EYE_SEMANTIC_OUTER_UV_OUTPUT="" \
    FOREGROUND_PROBABILITY_OUTPUT="" \
    FOREGROUND_MASK_OUTPUT="" \
    FOREGROUND_RAW_MASK_OUTPUT="" \
    FOREGROUND_CUTOUT_OUTPUT="" \
    FOREGROUND_PARSER_INPUT_OUTPUT="" \
    ./run_infer.sh >>"$LOG_PATH" 2>&1 \
    && "$PYTHON_BIN" - "$temporary" <<'PY'
import sys
from PIL import Image

with Image.open(sys.argv[1]) as image:
    if image.size != (64, 64):
        raise ValueError(f"Expected 64x64 result, got {image.size}")
    if image.mode != "RGBA":
        raise ValueError(f"Expected RGBA result, got {image.mode}")
PY
  then
    mv -f "$temporary" "$result"
    generated=$((generated + 1))
  else
    rm -f "$temporary"
    failed=$((failed + 1))
    echo >&2
    echo "Failed to regenerate: $source (see $LOG_PATH)" >&2
    if [[ "$FAIL_FAST" == "true" ]]; then
      exit 1
    fi
  fi
  printf '\r[%d/%d] generated=%d skipped=%d failed=%d' "$visited" "$total" "$generated" "$skipped" "$failed"
done < <(find "$DATA_DIR" -type f -name '*_edited.png' -print0 | sort -z)

echo
echo "v94 regeneration complete: visited=$visited generated=$generated skipped=$skipped failed=$failed"
if (( failed > 0 )); then
  exit 1
fi
