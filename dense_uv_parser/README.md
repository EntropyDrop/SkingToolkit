# Dense UV Parser

This module trains a dense parser for slightly deformed Minecraft render inputs:

```text
front/back render pixels
  -> foreground + layer + body part + face + UV coordinate
  -> splat pixels back to 64x64 UV conditioning
  -> inverse_uv inpaint model fills invisible/ambiguous texels
```

It keeps the existing two-image `view_images` contract by default:

```text
walk_front_both_layer_ortho,walk_back_both_layer_ortho
```

## Train Parser

```bash
./run_dense_uv_parser_training.sh
```

The trainer derives supervision from renderer mappings, so no manual labels are needed. For each GT skin it renders the configured views, labels every foreground pixel with:

- foreground/background
- layer: base or outer
- body part: head/body/arms/legs
- face index within the part
- continuous UV coordinate in the 64x64 Minecraft skin
- discrete UV x/y texel classes, used for sharper splatting

Parser training defaults to no augmentation so the model first learns the exact canonical mapping. After the clean parser preview is stable, enable mild render-space translation/scale augmentation for a second run if you need slightly deformed or misaligned inputs:

```bash
AUGMENT=true TRANSLATION_SCALE=0.035 SCALE_RANGE=0.035 ./run_dense_uv_parser_training.sh
```

Training previews are saved under `runs/<run>/previews`:

- `epoch_XXXX.png`: predicted inner/outer RGB rows, followed by GT inner/outer RGB rows.
- `epoch_XXXX_debug.png`: rendered input, predicted/GT foreground, predicted/GT part, predicted/GT layer, and predicted/GT UV color maps.

For good parser splatting, watch `acc_uv_exact`, `acc_uv_within1`, and `loss_uv_l1_px`. High part/layer accuracy alone can still produce fragmented RGB if UV is off by a few texels.

## Infer With Inpaint

Use the latest parser checkpoint automatically:

```bash
./run_infer.sh
```

By default it looks for the highest `runs/dense_uv_parser_v*/best.pt`, then looks for the highest `../inverse_uv/runs/inverse_uv_full_v*/best.pt`. It writes:

- `outputs/parser_conditioning.png`
- `outputs/pred_uv.png` when an inverse_uv inpaint checkpoint is found

Common overrides:

```bash
FRONT=/path/to/front.png BACK=/path/to/back.png ./run_infer.sh
COMBINED=/path/to/combined.png ./run_infer.sh
PARSER_CHECKPOINT=runs/dense_uv_parser_v3/best.pt ./run_infer.sh
INPAINT_CHECKPOINT=../inverse_uv/runs/inverse_uv_full_v34/best.pt ./run_infer.sh
OUTPUT= CONDITIONING_OUTPUT=outputs/parser_conditioning.png ./run_infer.sh
```

Use a trained parser checkpoint plus an existing `inverse_uv` inpaint checkpoint:

```bash
python infer.py \
  --parser_checkpoint runs/dense_uv_parser_v1/best.pt \
  --inpaint_checkpoint ../inverse_uv/runs/inverse_uv_full_v1/best.pt \
  --view_images front_both.png back_both.png \
  --output pred_uv.png
```

To inspect just the parser splat before inpainting:

```bash
python infer.py \
  --parser_checkpoint runs/dense_uv_parser_v1/best.pt \
  --view_images front_both.png back_both.png \
  --conditioning_output parser_conditioning.png
```

The conditioning preview shows the predicted inner-layer RGB row and outer-layer RGB row.
