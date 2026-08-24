#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Train only the five-class per-pixel semantic classifier.  By default this
# command performs real-domain adaptation on archived *_edited / *_result
# pairs.  If a previous synthetic semantic checkpoint exists it is used only
# to initialize model weights; optimizer and checkpoint-selection history are
# reset and the real validation split chooses the new best checkpoint.
export TRAINING_STAGE="semantic"
export RUN_FAMILY="${RUN_FAMILY:-dense_uv_semantic_v}"
export REAL_SEMANTIC_DATA_DIR="${REAL_SEMANTIC_DATA_DIR-../../SKING_DDJ_Dataset/entropydrop_website_generations}"
export PRIVILEGED_VIEWS=""
export EPOCHS="${EPOCHS:-12}"
export LR="${LR:-5e-5}"
export BEST_METRIC="${BEST_METRIC:-semantic_foreground_macro_iou_error}"
export FEATURE_DROPOUT="${FEATURE_DROPOUT:-0.15}"
export REPRODUCIBLE="${REPRODUCIBLE:-true}"
export STRICT_DETERMINISM="${STRICT_DETERMINISM:-false}"
export BACKGROUND_AUGMENT="false"

if [[ -n "$REAL_SEMANTIC_DATA_DIR" && -z "${INITIALIZE:-}" ]]; then
  latest_version=-1
  latest_checkpoint=""
  shopt -s nullglob
  for checkpoint in runs/dense_uv_semantic_v*/best.pt; do
    run_name="$(basename "$(dirname "$checkpoint")")"
    version="${run_name#dense_uv_semantic_v}"
    [[ "$version" =~ ^[0-9]+$ ]] || continue
    if (( 10#$version > latest_version )); then
      latest_version=$((10#$version))
      latest_checkpoint="$checkpoint"
    fi
  done
  shopt -u nullglob
  if [[ -n "$latest_checkpoint" ]]; then
    export INITIALIZE="$latest_checkpoint"
    echo "Initializing real-domain semantics from: $INITIALIZE"
  else
    echo "No previous semantic checkpoint found; training real-domain semantics from scratch."
  fi
fi

export SEMANTIC_BACKBONE="siglip2"
export SIGLIP_TEXT_PROMPT_FUSION="true"
export DENSE_SEMANTIC_TARGET_VERSION="3"
export LAMBDA_DENSE_SEMANTICS="${LAMBDA_DENSE_SEMANTICS:-1.0}"
export DENSE_SEMANTIC_OUTER_FALSE_POSITIVE_WEIGHT="${DENSE_SEMANTIC_OUTER_FALSE_POSITIVE_WEIGHT:-1.0}"
export LAMBDA_TEXT_PROMPT_ROUTE="0"

export PREDICT_HEAD_OUTER_STRUCTURE="false"
export PREDICT_OUTER_UV_OCCUPANCY="false"
export OUTER_UV_OCCUPANCY_ROUTING="false"
export OUTER_UV_COMPONENT_ROUTING="false"
export CROSS_VIEW_SPATIAL_FUSION="false"

export LAMBDA_FOREGROUND="0"
export LAMBDA_LAYER="0"
export LAMBDA_PART="0"
export LAMBDA_FACE="0"
export LAMBDA_LAYER_FACE="0"
export LAMBDA_UV="0"
export LAMBDA_UV_CLASS="0"
export LAMBDA_AFFINE="0"
export LAMBDA_SURFACE="0"
export LAMBDA_OUTER_FALSE_POSITIVE="0"
export LAMBDA_OUTER_FALSE_NEGATIVE="0"
export LAMBDA_ROUTE_CONFIDENCE="0"
export LAMBDA_PRIMARY_ROUTE_SWAP="0"
export LAMBDA_ROUTE_TEXEL_CONSISTENCY="0"
export LAMBDA_ROUTE_TEXEL_SUPERVISION="0"
export LAMBDA_CROSS_VIEW_OUTER_VISIBILITY="0"
export LAMBDA_PRIVILEGED_VIEW_DISTILLATION="0"
export LAMBDA_ROUTE_PRIOR_REGULARIZATION="0"
export LAMBDA_SEMANTIC_PRESENCE="0"
export LAMBDA_SEMANTIC_COVERAGE="0"

export LAMBDA_HEAD_OUTER_PRESENCE="0"
export LAMBDA_HEAD_OUTER_COVERAGE="0"
export LAMBDA_HEAD_OUTER_OCCUPANCY="0"
export LAMBDA_HEAD_OUTER_TOPOLOGY="0"
export LAMBDA_HEAD_OUTER_SYMMETRY="0"
export LAMBDA_HEAD_OUTER_COMPONENT_HARD_RECALL="0"
export LAMBDA_HEAD_OUTER_SPARSE_RECALL="0"
export LAMBDA_HEAD_OUTER_SYMMETRY_SCORE="0"
export LAMBDA_HEAD_OUTER_RING_WORST_FACE_RECALL="0"
export LAMBDA_HEAD_OUTER_OPEN_TOP_SHAPE="0"
export LAMBDA_HEAD_OUTER_ACCESSORY_CLASSIFICATION="0"
export LAMBDA_HEAD_OUTER_ROUTE_CONNECTIVITY="0"
export LAMBDA_HEAD_OUTER_ROUTE_SYMMETRY="0"
export LAMBDA_HEAD_EYE_OCCUPANCY="0"
export LAMBDA_HEAD_EYE_PRESENCE="0"
export LAMBDA_HEAD_EYE_SYMMETRY="0"
export LAMBDA_HEAD_EYE_ROUTE_CONNECTIVITY="0"
export LAMBDA_HEAD_EYE_ROUTE_SYMMETRY="0"

export LAMBDA_OUTER_UV_OCCUPANCY="0"
export LAMBDA_OUTER_COMPONENT_RECALL="0"
export LAMBDA_OUTER_COMPONENT_FALSE_POSITIVE="0"
export LAMBDA_OUTER_TOPOLOGY="0"
export LAMBDA_OUTER_NEGATIVE_TOPOLOGY="0"
export LAMBDA_ROUTE_OCCUPANCY_AGREEMENT="0"

export LAMBDA_SOFT_UV_RGB="0"
export LAMBDA_SOFT_UV_ALPHA="0"
export LAMBDA_SOFT_UV_INNER_RECALL="0"
export LAMBDA_SOFT_UV_OUTER_RECALL="0"
export LAMBDA_RENDER_RGB="0"
export LAMBDA_RENDER_ALPHA="0"
export LAMBDA_OUTER_PROJECTION_FALSE_POSITIVE="0"
export LAMBDA_OUTER_PROJECTION_FALSE_NEGATIVE="0"
export LAMBDA_OUTER_PROJECTION_DICE="0"
export LAMBDA_OUTER_PROJECTED_AREA="0"

exec ./run_dense_uv_parser_training.sh "$@"
