#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODEL="${MODEL:-full}"
CHECKPOINT="${CHECKPOINT:-}"

if [[ -z "$CHECKPOINT" ]]; then
  v=1
  latest_cp=""
  while [[ -d "runs/inverse_uv_${MODEL}_v${v}" ]]; do
    latest_cp="runs/inverse_uv_${MODEL}_v${v}/best.pt"
    ((v++))
  done

  if [[ -n "$latest_cp" ]]; then
    CHECKPOINT="$latest_cp"
  else
    CHECKPOINT="runs/inverse_uv_${MODEL}_v1/best.pt"
  fi
fi

python3 ./infer.py \
  --checkpoint "$CHECKPOINT" \
  --output ../test_imgs/output.png \
  --mappings_dir ../../github/differentiable_minecraft_renderer/mappings \
  --view_images ../test_imgs/banana_output3_front_rgba.png ../test_imgs/banana_output3_back_rgba.png 
  #--view_images ../test_imgs/walk_front_both_layer_ortho_pyvista.png ../test_imgs/walk_back_both_layer_ortho_pyvista.png
  #--view_images ../test_imgs/banana_front_rgba.png ../test_imgs/banana_back_rgba.png
  #--view_images ../test_imgs/front.png ../test_imgs/back.png 