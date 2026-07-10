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

The parser is explicitly conditioned on the configured view index, so front and back renders cannot become ambiguous for plain or symmetric skins. Inference must preserve the checkpoint's view order.

Parser training uses solid-color background randomization by default. Parser inputs are always treated as RGB, so transparent alpha is ignored and ordinary PNG/JPEG input has the same format. This lets inference accept arbitrary solid backgrounds without treating the fixed gray training canvas as part of the skin.

The input still needs an unobstructed Minecraft render in the configured camera/pose; arbitrary background does not mean arbitrary photo composition or a character hidden behind other objects.

Render-space augmentation is enabled by default. It applies one affine transform to the whole character with translation and uniform scale sampled up to `+/-3%`, matching nearly fixed views with mild global placement/size drift. It does not move individual limbs independently.

```bash
./run_dense_uv_parser_training.sh
```

The validation set uses the same `+/-3%` range with a fixed random seed, so `best.pt` is selected on repeatable perturbed inputs. Set `AUGMENT=false` and `AUGMENT_VALIDATION=false` only for canonical-view experiments.

Set `BACKGROUND_AUGMENT=false` to use a fixed gray RGB background; the input still remains RGB with alpha fixed to one internally.

Training previews are saved under `runs/<run>/previews`:

- `epoch_XXXX.png`: predicted inner/outer RGB rows, followed by GT inner/outer RGB rows.
- `epoch_XXXX_debug.png`: rendered input, predicted/GT foreground, predicted/GT part, predicted/GT layer, and predicted/GT UV color maps.

For good parser splatting, watch `acc_uv_exact`, `acc_uv_within1`, and `loss_uv_l1_px`. High part/layer accuracy alone can still produce fragmented RGB if UV is off by a few texels.

Predicted splatting keeps the highest-confidence render sample for each layer/UV texel instead of averaging all candidates, reducing color mixing at pixel boundaries.

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
