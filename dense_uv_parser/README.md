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

Parser training defaults to mild render-space translation/scale augmentation so it can handle slightly deformed or misaligned inputs:

```bash
AUGMENT=true TRANSLATION_SCALE=0.035 SCALE_RANGE=0.035 ./run_dense_uv_parser_training.sh
```

## Infer With Inpaint

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

