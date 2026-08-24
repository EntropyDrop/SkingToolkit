#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Train only the five-class per-pixel semantic classifier.  By default this
# command performs real-domain adaptation on archived *_edited / *_v94_result
# pairs. Files using the historical *_result suffix are deliberately ignored.
# If a previous synthetic semantic checkpoint exists it is used only
# to initialize model weights; optimizer and checkpoint-selection history are
# reset and the real validation split chooses the new best checkpoint.
export TRAINING_STAGE="semantic"
export RUN_FAMILY="${RUN_FAMILY:-dense_uv_semantic_v}"
export REAL_SEMANTIC_DATA_DIR="${REAL_SEMANTIC_DATA_DIR-../../SKING_DDJ_Dataset/entropydrop_website_generations}"
export REAL_SEMANTIC_RESULT_SUFFIX="${REAL_SEMANTIC_RESULT_SUFFIX:-_v94_result}"
export PRIVILEGED_VIEWS=""
export EPOCHS="${EPOCHS:-24}"
export LR="${LR:-1e-4}"
export BEST_METRIC="${BEST_METRIC:-semantic_outer_macro_iou_error}"
export FEATURE_DROPOUT="${FEATURE_DROPOUT:-0.05}"
export REPRODUCIBLE="${REPRODUCIBLE:-true}"
export STRICT_DETERMINISM="${STRICT_DETERMINISM:-false}"
export BACKGROUND_AUGMENT="false"

if [[ -n "$REAL_SEMANTIC_DATA_DIR" ]]; then
  v94_pair_count="$(find "$REAL_SEMANTIC_DATA_DIR" -type f -name "*${REAL_SEMANTIC_RESULT_SUFFIX}.png" | wc -l | tr -d ' ')"
  if (( v94_pair_count == 0 )); then
    echo "No *${REAL_SEMANTIC_RESULT_SUFFIX}.png labels found under $REAL_SEMANTIC_DATA_DIR." >&2
    echo "Generate them first with ./regenerate_v94_semantic_results.sh" >&2
    exit 1
  fi
  echo "Found versioned semantic UV labels: $v94_pair_count"
fi

if [[ -n "$REAL_SEMANTIC_DATA_DIR" && -z "${INITIALIZE:-}" ]]; then
  # Restart adaptation from the high-recall synthetic semantic model. The
  # previous real-domain run learned an over-conservative local optimum under
  # the old hard-negative objective and must not initialize this corrected
  # loss. An explicit INITIALIZE still overrides this choice.
  SEMANTIC_BASE_CHECKPOINT="${SEMANTIC_BASE_CHECKPOINT:-runs/dense_uv_semantic_v1/best.pt}"
  if [[ -f "$SEMANTIC_BASE_CHECKPOINT" ]]; then
    export INITIALIZE="$SEMANTIC_BASE_CHECKPOINT"
    echo "Initializing corrected real-domain semantics from: $INITIALIZE"
  else
    echo "Semantic base checkpoint not found: $SEMANTIC_BASE_CHECKPOINT"
    echo "Training corrected real-domain semantics from scratch."
  fi
fi

export SEMANTIC_BACKBONE="siglip2"
export SIGLIP_TEXT_PROMPT_FUSION="true"
export DENSE_SEMANTIC_TARGET_VERSION="3"
export LAMBDA_DENSE_SEMANTICS="${LAMBDA_DENSE_SEMANTICS:-1.0}"
export DENSE_SEMANTIC_OUTER_FALSE_POSITIVE_WEIGHT="${DENSE_SEMANTIC_OUTER_FALSE_POSITIVE_WEIGHT:-0.20}"
export DENSE_SEMANTIC_OUTER_UNION_WEIGHT="${DENSE_SEMANTIC_OUTER_UNION_WEIGHT:-1.0}"
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
