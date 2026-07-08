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
To guarantee both flat UV accuracy and visual rendering consistency, training optimizes a weighted sum of reconstruction terms:
1. **Alpha-Masked RGB L1 Loss (`loss_rgb`)**: Supervises RGB reconstruction strictly on valid skin UV regions, ignoring empty padding and inner-layer texels hidden behind opaque matching outer-layer texels.
2. **Alpha Binary Cross-Entropy (`loss_alpha`)**: Supervises the transparency layout (sigmoidal BCE) on the same visible UV supervision mask.
3. **Alpha Dice Loss (`loss_alpha_dice`)**: Supervises the full alpha region so outlines do not become transparent and optional outer-layer alpha does not leak.
4. **Alpha Edge L1 Loss (`loss_alpha_edge`)**: Computes alpha-gradient differences to keep silhouettes, cutouts, and outer-layer boundaries crisp.
5. **Differentiable Render Consistency L1 Loss (`loss_render`)**: Passively runs the predicted 64x64 skin through the `DifferentiableRenderer` to generate 2D camera views, comparing them against the ground truth renders. This forces the network to resolve overlapping texture layers correctly.
6. **UV-Space Edge L1 Loss (`loss_edge`)**: Computes RGB-gradient differences (x and y directions) between predicted and ground truth skins to enforce sharp pixel boundaries.

---

## Train

Recommended first-stage training is color-first and edge-heavy:

```bash
./run_inverse_uv_training.sh
```

The shell script defaults to `LAMBDA_GAN=0`, `LAMBDA_RGB=2.0`, `LAMBDA_ALPHA=0.8`, `LAMBDA_ALPHA_DICE=0.5`, `LAMBDA_ALPHA_EDGE=0.5`, `LAMBDA_RENDER=0.2`, `LAMBDA_RENDER_ALPHA=0.4`, `LAMBDA_EDGE=1.0`, and `EPOCHS=30`. This avoids early GAN color drift while pushing visible RGB, alpha stability, and UV pixel boundaries harder.

For a short sharpening finetune after the first run, resume from the best checkpoint with a very small GAN weight:

```bash
RESUME=runs/inverse_uv_full_v1/best.pt RESUME_LR=5e-5 EPOCHS=15 LAMBDA_GAN=0.005 ./run_inverse_uv_training.sh
```

Use the actual run folder name in `RESUME`. Avoid jumping straight back to `LAMBDA_GAN=0.03` unless the colors are already stable.

```bash
python SkingToolkit/inverse_uv/train.py \
  --data_dir /path/to/gt_skins \
  --output_dir runs/inverse_uv_static \
  --views static_front,static_back,top_front_45,top_back_45 \
  --batch_size 16 \
  --epochs 20
```

Useful knobs:

- `--unproject_mode`: aggregation method for render pixels unprojected into 64x64 UV texels (`mean`=average, `mode`=most frequent 8-bit color, `medoid`=spatial median). Batched training currently supports `mean`; `mode`/`medoid` are for unbatched inference or debugging.
- `--best_metric`: checkpoint selection metric. Defaults to `loss_recon_total` so `best.pt` is not dominated by GAN oscillation.
- `--scheduler` / `--min_lr`: optional learning-rate scheduler controls. The training shell script defaults to cosine decay.
- `--log_every`: progress-bar metric sync interval in batches. Larger values reduce GPU/CPU synchronization overhead.
- `--prefetch_factor`: DataLoader prefetch depth when `--num_workers > 0`.
- `--matmul_precision` / `--cudnn_benchmark`: CUDA backend throughput controls for fixed-size training.
- `--lambda_rgb`: visible-RGB UV reconstruction weight.
- `--lambda_alpha`: alpha reconstruction weight.
- `--lambda_alpha_dice`: alpha region consistency weight for reducing transparent holes and false alpha leaks.
- `--lambda_alpha_edge`: alpha boundary reconstruction weight for cleaner silhouettes and outer-layer cutouts.
- `--lambda_render`: differentiable render consistency weight.
- `--lambda_render_alpha`: rendered alpha consistency weight for visible holes/false positives.
- `--lambda_gan`: PatchGAN adversarial weight. Defaults to `0` for color-first reconstruction; use tiny values like `0.005` for a later sharpening finetune.
- `--lambda_edge`: UV-space edge reconstruction weight for sharper pixel boundaries.
- `--coordconv` / `--no-coordconv`: append normalized x/y coordinates inside `InverseUVNet` so the model sees absolute UV position.
- `--bottleneck_attention` / `--no-bottleneck-attention`: enable or disable lightweight bottleneck self-attention.
- `--attention_heads`: number of bottleneck self-attention heads.
- `--supervise_covered_inner`: keep supervising inner-layer UV texels even when opaque matching outer-layer texels hide them.
- `--covered_inner_alpha_threshold`: GT outer-layer alpha threshold used to decide covered inner texels; defaults to `0.1`.
- `--render_size`: deprecated compatibility option; UV unprojection uses native mapping sizes.
- `--include_alpha`: deprecated compatibility option; conditioning always uses RGBA plus masks.

Performance notes:

- `run_inverse_uv_training.sh` defaults `PERSPECTIVE_SCALE=0.0` so augmentation stays on the batched affine path. Re-enable perspective only when the extra pose robustness is worth the slower per-sample transform.
- For lower console overhead on fast GPUs, raise `LOG_EVERY` (for example `LOG_EVERY=100`).

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
