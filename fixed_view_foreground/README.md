# Fixed-View Foreground Segmentation

This module trains a lightweight foreground/background model specifically for
the front/back Minecraft views used by `dense_uv_parser`.

The renderer supplies exact foreground labels, so no hand-authored masks are
required. Training composites each rendered character over difficult procedural
backgrounds: solid colors, gray gradients, low-frequency textures, and colors
sampled near the character palette. The network also receives the fixed-view
inner/outer geometry silhouettes as two conditioning channels. Low-contrast
pixels, arm pixels, opaque interiors, and silhouette boundaries receive extra
loss weight; checkpoint selection explicitly penalizes global and arm false
negatives.

## Train

```bash
./run_training.sh
```

Runs are versioned automatically:

```text
runs/fixed_view_foreground_v1/
  best.pt
  latest.pt
  config.json
  metrics.jsonl
  train.log
  previews/epoch_XXXX.png
```

The preview rows are input, probability, thresholded prediction, target, and
background-removed cutout. `best.pt` minimizes validation foreground IoU error
plus explicit global, opaque-interior, and arm-recall penalties.

Common overrides:

```bash
MAPPINGS_DIR=../differentiable_minecraft_renderer/mappings_256x512 \
MAX_SAMPLES=30000 \
BATCH_SIZE=32 \
EPOCHS=30 \
./run_training.sh
```

Resume the highest current run (only when its architecture matches):

```bash
RESUME=latest EPOCHS=45 ./run_training.sh
```

Geometry conditioning changes the first convolution and therefore cannot be
added by resuming a pre-geometry checkpoint. Start without `RESUME`; the launcher
automatically chooses the next unused `fixed_view_foreground_vN` directory:

```bash
./run_training.sh
```

## Dense Parser Integration

`dense_uv_parser/run_infer.sh` automatically selects the numerically highest
`fixed_view_foreground_vN/best.pt`. The predicted mask replaces the former
solid-background color gate; inner/outer/secondary routing remains owned by the
dense parser.

Inference writes these intermediate products by default:

```text
dense_uv_parser/outputs/foreground_probability.png
dense_uv_parser/outputs/foreground_mask_raw.png
dense_uv_parser/outputs/foreground_mask.png
dense_uv_parser/outputs/foreground_cutout.png
dense_uv_parser/outputs/foreground_parser_input.png
```

`foreground_mask_raw.png` is the thresholded network output.
`foreground_mask.png` is the mask actually consumed by the parser: it restores
only an eroded, guaranteed inner-layer geometry core and fills background holes
that are completely enclosed by foreground. Gaps connected to the exterior and
optional outer-layer texels are not force-filled.

`foreground_cutout.png` is an RGBA image whose rejected background has zero
alpha. Internally, dense parser features receive the same cutout composited over
a deterministic adaptive high-contrast color. The color is selected from a
fixed palette by maximizing the low-quantile RGB distance to high-confidence
foreground boundary colors. `foreground_parser_input.png` shows the exact RGB
given to the parser. UV color splatting continues to sample the untouched source
image so foreground colors are not altered.

Use the former neutral gray background for an ablation:

```bash
FOREGROUND_PARSER_BACKGROUND=neutral ./run_infer.sh
```

Choose or disable the checkpoint explicitly:

```bash
FOREGROUND_CHECKPOINT=../fixed_view_foreground/runs/fixed_view_foreground_v2/best.pt \
./run_infer.sh

FOREGROUND_CHECKPOINT=none ./run_infer.sh
```

When disabled, inference falls back to the existing border-connected background
color estimator. `FOREGROUND_THRESHOLD` defaults to `0.35`, favoring recall;
dense UV routing still requires a valid fixed-geometry candidate, which limits
the effect of residual background false positives.
