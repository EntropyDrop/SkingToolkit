# Foreground Alpha

Train a small model that converts an RGB Minecraft render into an RGBA render by predicting the foreground alpha mask.

This is useful when production renders have an opaque background but `inverse_uv` works best with explicit alpha.

## Train

From `SkingToolkit/`:

```bash
python foreground_alpha/train.py \
  --data_dir ./skins \
  --output_dir runs/foreground_alpha_test1 \
  --views walk_perspective_ortho,walk_perspective_back_ortho \
  --mappings_dir ../differentiable_minecraft_renderer/mappings \
  --background_mode random \
  --batch_size 4 \
  --epochs 50 \
  --val_split 0.1 \
  --save_every 1 \
  --preview_every 1
```

`--background_mode random` composites each rendered character over a random solid RGB background. Use `black`, `white`, `gray`, or `color --bg_color r,g,b` when you want to match a fixed production background.

The preview layout is:

```text
input RGB rows, predicted alpha rows, target alpha rows
```

## Infer

For one image:

```bash
python foreground_alpha/infer.py \
  --checkpoint runs/foreground_alpha_test1/best.pt \
  --input /path/walk_perspective_ortho_black.png \
  --output /path/walk_perspective_ortho_rgba.png
```

For a known solid background, enable uncomposition to recover cleaner edge RGB:

```bash
python foreground_alpha/infer.py \
  --checkpoint runs/foreground_alpha_test1/best.pt \
  --input /path/walk_perspective_ortho_black.png \
  --output /path/walk_perspective_ortho_rgba.png \
  --bg_color 0,0,0 \
  --uncompose
```

Then pass the generated RGBA images to `inverse_uv/infer.py` in the same view order as the checkpoint.
