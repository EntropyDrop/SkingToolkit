# Inverse UV Training

This module trains a supervised inpainting model for:

```text
fixed-view render images -> unprojected UV conditioning -> original 64x64 RGBA Minecraft skin UV
```

It is intentionally separate from the existing Flux/LoRA training path. The main
supervision is GT UV reconstruction; differentiable render consistency and a
light edge loss are auxiliary terms. Renderer mappings are used to unproject
each view into aligned UV space before the U-Net sees the sample, so the model
learns UV-space inpainting instead of long-distance render-to-UV pixel moves.

## Train

```bash
python SkingToolkit/inverse_uv/train.py \
  --data_dir /path/to/gt_skins \
  --output_dir runs/inverse_uv_static \
  --views static_front,static_back,top_front_45,top_back_45 \
  --batch_size 16 \
  --epochs 20
```

Useful knobs:

- `--lambda_rgb`: visible-RGB UV reconstruction weight.
- `--lambda_alpha`: alpha reconstruction weight.
- `--lambda_render`: differentiable render consistency weight.
- `--lambda_edge`: UV-space edge reconstruction weight for sharper pixel boundaries.
- `--render_size`: deprecated compatibility option; UV unprojection uses native mapping sizes.
- `--include_alpha`: deprecated compatibility option; conditioning always uses RGBA plus masks.

## Infer

```bash
python SkingToolkit/inverse_uv/infer.py \
  --checkpoint runs/inverse_uv_static/best.pt \
  --view_images /path/to/static_front.png /path/to/static_back.png /path/to/top_front_45.png /path/to/top_back_45.png \
  --output /path/to/pred_uv.png
```

For a side-by-side image whose panels match the checkpoint view order:

```bash
python SkingToolkit/inverse_uv/infer.py \
  --checkpoint runs/inverse_uv_static/best.pt \
  --combined /path/to/combined_views.png \
  --output /path/to/pred_uv.png
```

## Generate Render Pairs

```bash
python SkingToolkit/inverse_uv/generate_pairs.py \
  --data_dir /path/to/gt_skins \
  --output_dir /path/to/render_pairs \
  --views static_front,static_back,top_front_45,top_back_45 \
  --combined
```
