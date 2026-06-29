# Foreground Alpha

Train a small model that converts an RGB Minecraft render into an RGBA render by predicting the foreground alpha mask.

This is useful when production renders have an opaque background but `inverse_uv` works best with explicit alpha.

---

## 🧬 Working Principles & Core Architecture

Renders of Minecraft characters generated in production often have opaque solid backgrounds (black, white, or gray). To feed these renders into the `inverse_uv` reconstruction pipeline, we need to extract a precise 1-channel alpha mask to separate the character from the background. 

The `foreground_alpha` module implements a segmentation pipeline to solve this:

### 1. Data Synthesis & Background Randomization
To train the model to ignore backgrounds, the dataloader uses clean 4-channel ground truth character renders and dynamically composites them over randomized backgrounds:
* **Random Mode**: composites the character over a random solid RGB color.
* **Fixed Mode**: composites over a solid black, white, or gray background.
This background randomization forces the convolutional filters to learn spatial structures (skin boundaries and silhouettes) rather than specific pixel values.

### 2. Network Architecture (`ForegroundAlphaNet`)
* Uses a lightweight, high-performance U-Net architecture.
* **Encoder**: 3 Conv blocks downsampled using $4 \times 4$ strided convolutions, regularized with Group Normalization and SiLU activations.
* **Decoder**: Bilinear interpolation and skip connections to recover fine details along anti-aliased character edges.
* **Output Head**: Predicts a single-channel alpha mask (shape `1 x H x W`, range `[0, 1]`) via a Sigmoid activation.

### 3. Segmentation Loss Formulation
To optimize mask coverage and sharp outlines, the training uses a four-term loss function:
1. **Binary Cross Entropy (`loss_bce`)**: Standard pixel-wise binary classifier supervising transparency.
2. **L1 Reconstruction Loss (`loss_l1`)**: Enforces absolute error minimization.
3. **Dice Loss (`loss_dice`)**: Evaluates the overlap of the predicted and ground truth masks to mitigate foreground-background imbalance.
4. **Edge L1 Loss (`loss_edge`)**: Minimizes L1 difference between the spatial gradients of the predicted mask and ground truth mask to ensure sharp, clean boundaries.

### 4. Edge Color Uncomposition (Inference)
When background blending is enabled, anti-aliased pixels on the character's boundary will mix the foreground color with the background color:
$$C_{observed} = \alpha \cdot C_{foreground} + (1 - \alpha) \cdot C_{background}$$

During inference, if the background color $C_{background}$ is known, the model can mathematically **uncompose** (reconstruct) the clean, background-free foreground color for boundary pixels:
$$C_{foreground} = \frac{C_{observed} - (1 - \alpha) \cdot C_{background}}{\alpha}$$
This prevents background color bleeding along the edges of the reconstructed skin.

---

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
