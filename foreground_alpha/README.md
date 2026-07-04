# Foreground Alpha

Train a small model that converts an RGB Minecraft render into an RGBA render by predicting the foreground alpha mask.

This is useful when production renders have an opaque background but `inverse_uv` works best with explicit alpha.

---

## 🧬 Working Principles & Core Architecture

Renders of Minecraft characters generated in production often have opaque solid backgrounds (black, white, or gray). To feed these renders into the `inverse_uv` reconstruction pipeline, we need to extract a precise 1-channel alpha mask to separate the character from the background. 

The `foreground_alpha` module implements a segmentation pipeline to solve this:

### 1. Data Synthesis & Background Randomization
To train the model to ignore backgrounds and prevent removing character parts with colors matching the background, the dataloader uses clean 4-channel ground truth character renders and dynamically composites them over randomized backgrounds:
* **Random Mode**: composites the character over a mixture of random solid RGB colors, gradients, patterns, and **hard background colors sampled directly from the character's own palette** (`--hard_bg_prob`).
* **Fixed Mode**: composites over a solid black, white, gray, gradient, or pattern background.
This background randomization forces the convolutional filters to learn spatial structures (skin boundaries and silhouettes) rather than specific pixel values or color matching shortcuts.

### 2. Network Architecture (`ForegroundAlphaNet`)
* Uses a lightweight, high-performance U-Net architecture.
* **Encoder**: 3 Conv blocks downsampled using $4 \times 4$ strided convolutions, regularized with Group Normalization and SiLU activations.
* **Decoder**: Bilinear interpolation and skip connections to recover fine details along anti-aliased character edges.
* **Output Head**: Predicts a single-channel alpha mask (shape `1 x H x W`, range `[0, 1]`) via a Sigmoid activation.

### 3. Segmentation Loss Formulation
To optimize mask coverage, avoid interior transparent holes, and sharp outlines, the training uses a five-term loss function:
1. **Binary Cross Entropy (`loss_bce`)**: Standard pixel-wise binary classifier supervising transparency.
2. **L1 Reconstruction Loss (`loss_l1`)**: Enforces absolute error minimization.
3. **Dice Loss (`loss_dice`)**: Evaluates the overlap of the predicted and ground truth masks to mitigate foreground-background imbalance.
4. **Edge L1 Loss (`loss_edge`)**: Minimizes L1 difference between the spatial gradients of the predicted mask and ground truth mask to ensure sharp, clean boundaries.
5. **Interior Hole Loss (`loss_hole`)**: Penalizes false negatives inside the solid character silhouette to prevent interior erosion when character colors match background colors.

### 4. Edge Color Uncomposition & Hole Filling (Inference)
When background blending is enabled, anti-aliased pixels on the character's boundary will mix the foreground color with the background color:
$$C_{observed} = \alpha \cdot C_{foreground} + (1 - \alpha) \cdot C_{background}$$

During inference:
* If the background color $C_{background}$ is known, `--uncompose` mathematically reconstructs clean, background-free foreground color for boundary pixels.
* `--fill_holes` performs morphological hole filling to fill any isolated transparent pixels inside the character mask.

---

## Train

From `SkingToolkit/`:

```bash
python foreground_alpha/train.py \
  --data_dir ./skins \
  --output_dir runs/foreground_alpha_test1 \
  --views walk_front_both_layer_ortho,walk_back_both_layer_ortho \
  --mappings_dir ../differentiable_minecraft_renderer/mappings \
  --background_mode random \
  --hard_bg_prob 0.3 \
  --lambda_hole 1.0 \
  --batch_size 4 \
  --epochs 50 \
  --val_split 0.1 \
  --save_every 1 \
  --preview_every 1
```

`--background_mode random` composites each rendered character over random backgrounds (including hard color matching). Use `--hard_bg_prob` (default `0.3`) to control the frequency of sampling background colors directly from the character.

The preview layout is:

```text
input RGB rows, predicted alpha rows, target alpha rows
```

## Infer

For one image with hole filling and uncomposition:

```bash
python foreground_alpha/infer.py \
  --checkpoint runs/foreground_alpha_test1/best.pt \
  --input /path/walk_front_both_layer_ortho_black.png \
  --output /path/walk_front_both_layer_ortho_rgba.png \
  --fill_holes \
  --bg_color 0,0,0 \
  --uncompose
```

Then pass the generated RGBA images to `inverse_uv/infer.py` in the same view order as the checkpoint.

