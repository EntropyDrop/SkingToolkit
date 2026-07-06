# Inverse UV Training

This module trains a supervised inpainting model for:

```text
fixed-view render images -> unprojected UV conditioning -> original 64x64 RGBA Minecraft skin UV
```

It is intentionally separate from the existing Flux/LoRA training path.

---

## 🧬 Working Principles & Core Architecture

Reconstructing a flat 64x64 Minecraft skin layout from 2D camera renders is a difficult spatial mapping problem. Standard Convolutional Neural Networks (CNNs) struggle to map pixels over long coordinate distances (e.g., from the side of a character's arm in a render to its exact coordinate on a flat 64x64 texture map). 

To solve this, `inverse_uv` utilizes coordinate-guided unprojection combined with UV-space inpainting:

### 1. Coordinate-Guided Unprojection
Instead of learning the render-to-sheet translation from scratch, the system utilizes camera mapping files (`*.pt`) from the `DifferentiableRenderer`. 
* Each camera view has a matching coordinate grid that details exactly where each pixel in the 2D render maps onto the 64x64 flat UV canvas.
* Before feeding the views into the neural network, the dataset script uses these coordinates to **unproject** the multi-view renders into a combined 64x64 UV conditioning tensor (with layers for inner, outer, and known masks).
* This shifts the network's objective from *complex coordinate translation* to *UV-space image inpainting/denoising*, which U-Nets are highly suited for.

### 2. Network Architecture (`InverseUVNet`)
* The model is a custom 2D U-Net-like segmentation network.
* **Encoder**: Consists of a Conv Stem followed by 3 Down-sampling blocks. Each down-block uses Group Normalization (dynamically selecting groups based on channels) and SiLU activations.
* **Decoder**: Features 3 Up-sampling blocks with bilinear interpolation and skip connections from the encoder to retain high-frequency textures.
* **Output Head**: Predicts a 4-channel (RGBA) flat skin in `[0, 1]` using a Sigmoid activation.

### 3. Multi-Term Loss Formulation (`InverseUVLoss`)
To guarantee both flat UV accuracy and visual rendering consistency, training optimizes a weighted sum of four loss terms:
1. **Alpha-Masked RGB L1 Loss (`loss_rgb`)**: Supervises RGB reconstruction strictly on valid skin UV regions, ignoring empty padding and inner-layer texels hidden behind opaque matching outer-layer texels.
2. **Alpha Binary Cross-Entropy (`loss_alpha`)**: Supervises the transparency layout (sigmoidal BCE) on the same visible UV supervision mask.
3. **Differentiable Render Consistency L1 Loss (`loss_render`)**: Passively runs the predicted 64x64 skin through the `DifferentiableRenderer` to generate 2D camera views, comparing them against the ground truth renders. This forces the network to resolve overlapping texture layers correctly.
4. **UV-Space Edge L1 Loss (`loss_edge`)**: Computes L1 difference between the gradients (x and y directions) of predicted vs ground truth skins to enforce sharp pixel boundaries.

---

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

- `--unproject_mode`: aggregation method for render pixels unprojected into 64x64 UV texels (`mode`=most frequent 8-bit color, `mean`=average, `medoid`=spatial median). Defaults to `mode` to prevent color averaging at block boundaries.
- `--lambda_rgb`: visible-RGB UV reconstruction weight.
- `--lambda_alpha`: alpha reconstruction weight.
- `--lambda_render`: differentiable render consistency weight.
- `--lambda_edge`: UV-space edge reconstruction weight for sharper pixel boundaries.
- `--coordconv` / `--no-coordconv`: append normalized x/y coordinates inside `InverseUVNet` so the model sees absolute UV position.
- `--bottleneck_attention` / `--no-bottleneck-attention`: enable or disable lightweight bottleneck self-attention.
- `--attention_heads`: number of bottleneck self-attention heads.
- `--supervise_covered_inner`: keep supervising inner-layer UV texels even when opaque matching outer-layer texels hide them.
- `--covered_inner_alpha_threshold`: GT outer-layer alpha threshold used to decide covered inner texels; defaults to `0.1`.
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
