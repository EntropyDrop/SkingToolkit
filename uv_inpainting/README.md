# UV Inpainting Training

This module trains a supervised inpainting model for:

```text
fixed-view render images -> dense parser UV conditioning -> complete 64x64 RGBA Minecraft skin UV
```

It is the completion stage after `dense_uv_parser` and is trained from the same parser-generated conditioning used at inference.

---

## 🧬 Working Principles & Core Architecture

Reconstructing a flat 64x64 Minecraft skin layout from 2D camera renders is a difficult spatial mapping problem. A convolutional network should not have to rediscover the camera-to-UV geometry while also learning how to fill invisible texture regions.

To solve this, `dense_uv_parser` performs coordinate-guided routing first and `uv_inpainting` concentrates on UV-space completion:

### 1. Dense Parser Conditioning
* The frozen parser classifies visible pixels as inner, outer, or secondary/deeper surfaces.
* Fixed renderer mappings route those pixels into a 10-channel tensor: inner RGBA plus known mask, followed by outer RGBA plus known mask.
* Unknown and ambiguous texels remain empty, turning the task into UV-space inpainting rather than camera-coordinate translation.

### 2. Network Architecture (`UVInpaintingNet`)
* The model is a custom 2D U-Net-like segmentation network.
* **Encoder**: Consists of a Conv Stem followed by 3 Down-sampling blocks. Each down-block uses Group Normalization (dynamically selecting groups based on channels) and SiLU activations.
* **Decoder**: Features 3 Up-sampling blocks with bilinear interpolation and skip connections from the encoder to retain high-frequency textures.
* **Output Head**: Predicts a 4-channel (RGBA) flat skin in `[0, 1]` using a Sigmoid activation.

### 3. Multi-Term Loss Formulation (`UVInpaintingLoss`)
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
./run_uv_inpainting_training.sh
```

The shell script defaults to the original two fixed `view_images`:

```text
walk_front_both_layer_ortho,walk_back_both_layer_ortho
```

Geometric augmentation is disabled by default to preserve fixed-view pixel alignment. The other defaults are `LAMBDA_GAN=0`, `LAMBDA_RGB=2.0`, `LAMBDA_ALPHA=0.8`, `LAMBDA_ALPHA_DICE=0.5`, `LAMBDA_ALPHA_EDGE=0.5`, `LAMBDA_RENDER=0.2`, `LAMBDA_RENDER_ALPHA=0.4`, `LAMBDA_EDGE=1.0`, and `EPOCHS=30`.

For a short sharpening finetune after the first run, resume from the best checkpoint with a very small GAN weight:

```bash
RESUME=runs/uv_inpainting_full_v1/best.pt RESUME_LR=5e-5 EPOCHS=15 LAMBDA_GAN=0.005 ./run_uv_inpainting_training.sh
```

Use the actual run folder name in `RESUME`. Avoid jumping straight back to `LAMBDA_GAN=0.03` unless the colors are already stable.

```bash
python SkingToolkit/uv_inpainting/train.py \
  --data_dir /path/to/gt_skins \
  --output_dir runs/uv_inpainting_static \
  --parser_checkpoint ../dense_uv_parser/runs/dense_uv_parser_v1/best.pt \
  --views static_front,static_back,top_front_45,top_back_45 \
  --batch_size 16 \
  --epochs 20
```

Useful knobs:

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
- `--supervise_covered_inner`: keep supervising inner-layer UV texels even when opaque matching outer-layer texels hide them.
- `--covered_inner_alpha_threshold`: GT outer-layer alpha threshold used to decide covered inner texels; defaults to `0.1`.
- `--render_size`: deprecated compatibility option; UV unprojection uses native mapping sizes.
- `--include_alpha`: deprecated compatibility option; conditioning always uses RGBA plus masks.

Performance notes:

- `run_uv_inpainting_training.sh` disables geometric augmentation by default: `AUGMENT=false`, `TRANSLATION_SCALE=0.0`, `SCALE_RANGE=0.0`, and `PERSPECTIVE_SCALE=0.0`.
- Validation also uses fixed canonical geometry, and parser affine refinement is disabled. Pure solid-background randomization remains enabled.
- Parser conditioning uses the same conservative outer gates as production inference: confidence `0.55`, relative margin `0.35`, and UV coverage `0.65`.
- For lower console overhead on fast GPUs, raise `LOG_EVERY` (for example `LOG_EVERY=100`).

### Train From Dense Parser Conditioning

UV Inpainting training always uses parser-generated conditioning, matching `dense_uv_parser/infer.py` at inference time:

```bash
./run_uv_inpainting_training.sh
```

This automatically finds the newest `../dense_uv_parser/runs/dense_uv_parser_v*/best.pt`. Its frozen parser also sees the same randomized solid-color RGB backgrounds used for parser training, so UV inpainting learns to complete parser conditioning from arbitrary-solid-background inputs. To finetune from an existing `uv_inpainting` checkpoint:

Visible UV texels recovered by the parser are copied directly into the final output; the inpaint network only determines unknown texels. This prevents already observed colors from being blurred by reconstruction.

```bash
RESUME=runs/uv_inpainting_full_v34/best.pt \
RESUME_LR=5e-5 \
EPOCHS=15 \
./run_uv_inpainting_training.sh
```

Override parser selection with:

```bash
PARSER_CHECKPOINT=../dense_uv_parser/runs/dense_uv_parser_v3/best.pt \
./run_uv_inpainting_training.sh
```

## Inference

UV inpainting does not expose a standalone inference entry point. Production inference must first build conditioning with the same dense parser used during training:

```bash
cd SkingToolkit/dense_uv_parser
FRONT=/path/to/front.png BACK=/path/to/back.png ./run_infer.sh
```

`dense_uv_parser/infer.py` loads `UVInpaintingNet` after parser splatting, then writes the completed skin to `outputs/pred_uv.png`.

## Generate Render Pairs

```bash
python SkingToolkit/uv_inpainting/generate_pairs.py \
  --data_dir /path/to/gt_skins \
  --output_dir /path/to/render_pairs \
  --views static_front,static_back,top_front_45,top_back_45 \
  --combined
```
