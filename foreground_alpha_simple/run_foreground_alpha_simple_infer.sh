#!/bin/bash

# Simple Magic-Wand (Flood Fill from 0,0) Background Removal Launcher

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLKIT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd "${TOOLKIT_DIR}/.." && pwd)"

export PYTHONPATH="${WORKSPACE_DIR}:${PYTHONPATH}"

python3 "${SCRIPT_DIR}/infer.py" \
  --inputs "../../SkingDataset/DDJ_real2render/test_output/img4_template7.png" \
  --output_front "../test_imgs/front_rgba.png" \
  --output_back "../test_imgs/back_rgba.png" \
  --seed "0,0" \
  --tolerance 15 \
  --color_space "RGB" \
  --uncompose
