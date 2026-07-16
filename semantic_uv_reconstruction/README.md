# Semantic UV Reconstruction Training

This module contains two reconstruction paths. The new primary path learns:

```text
fixed front/back renders -> CNN + frozen SigLIP2 semantics -> complete 64x64 RGBA UV
```

Start it with `./run_semantic_uv_reconstruction_training.sh`. It does not need a
parser checkpoint or a finite garment concept table.

The directory also retains the compatible parser-conditioned completion path:

```text
fixed-view renders -> dense parser UV conditioning -> complete 64x64 RGBA UV
```

Start that legacy-compatible path with `./run_parser_conditioned_training.sh`.

---

## 🧬 Working Principles & Core Architecture

Reconstructing a flat 64x64 Minecraft skin layout from 2D camera renders is a difficult spatial mapping problem. A convolutional network should not have to rediscover the camera-to-UV geometry while also learning how to fill invisible texture regions.

To solve this, `dense_uv_parser` performs coordinate-guided routing first and `semantic_uv_reconstruction` concentrates on UV-space completion:

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
./run_parser_conditioned_training.sh
```

The shell script defaults to the original two fixed `view_images`:

```text
walk_front_both_layer_ortho,walk_back_both_layer_ortho
```

Geometric augmentation is disabled by default to preserve fixed-view pixel alignment. The other defaults are `LAMBDA_GAN=0`, `LAMBDA_RGB=2.0`, `LAMBDA_ALPHA=0.8`, `LAMBDA_ALPHA_DICE=0.5`, `LAMBDA_ALPHA_EDGE=0.5`, `LAMBDA_RENDER=0.2`, `LAMBDA_RENDER_ALPHA=0.4`, `LAMBDA_EDGE=1.0`, and `EPOCHS=30`.

For a short sharpening finetune after the first run, resume from the best checkpoint with a very small GAN weight:

```bash
RESUME=runs/semantic_uv_reconstruction_full_v1/best.pt RESUME_LR=5e-5 EPOCHS=15 LAMBDA_GAN=0.005 ./run_parser_conditioned_training.sh
```

Use the actual run folder name in `RESUME`. Avoid jumping straight back to `LAMBDA_GAN=0.03` unless the colors are already stable.

```bash
python SkingToolkit/semantic_uv_reconstruction/train.py \
  --data_dir /path/to/gt_skins \
  --output_dir runs/semantic_uv_reconstruction_static \
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

- `run_parser_conditioned_training.sh` disables geometric augmentation by default: `AUGMENT=false`, `TRANSLATION_SCALE=0.0`, `SCALE_RANGE=0.0`, and `PERSPECTIVE_SCALE=0.0`.
- Validation also uses fixed canonical geometry, and parser affine refinement is disabled. Pure solid-background randomization remains enabled.
- Parser conditioning uses the same semantic-first routing as production inference: semantic outer confidence `0.55`, relative margin `0.35`, projected-texel consensus disabled, and geometric UV coverage filtering disabled (`0`). The fixed grid assigns approximate UV coordinates but does not override the predicted inner/outer meaning.
- For lower console overhead on fast GPUs, raise `LOG_EVERY` (for example `LOG_EVERY=100`).

### Train From Dense Parser Conditioning

Semantic UV Reconstruction training always uses parser-generated conditioning, matching `dense_uv_parser/infer.py` at inference time:

```bash
./run_parser_conditioned_training.sh
```

This automatically finds the newest `../dense_uv_parser/runs/dense_uv_parser_v*/best.pt`. Its frozen parser also sees the same randomized solid-color RGB backgrounds used for parser training, so Semantic UV reconstruction learns to complete parser conditioning from arbitrary-solid-background inputs. To finetune from an existing `semantic_uv_reconstruction` checkpoint:

Visible UV texels recovered by the parser are copied directly into the final output; the inpaint network only determines unknown texels. This prevents already observed colors from being blurred by reconstruction.

```bash
RESUME=runs/semantic_uv_reconstruction_full_v34/best.pt \
RESUME_LR=5e-5 \
EPOCHS=15 \
./run_parser_conditioned_training.sh
```

Override parser selection with:

```bash
PARSER_CHECKPOINT=../dense_uv_parser/runs/dense_uv_parser_v3/best.pt \
./run_parser_conditioned_training.sh
```

## Inference

Semantic UV reconstruction does not expose a standalone inference entry point. Production inference must first build conditioning with the same dense parser used during training:

```bash
cd SkingToolkit/dense_uv_parser
FRONT=/path/to/front.png BACK=/path/to/back.png ./run_infer.sh
```

`dense_uv_parser/infer.py` loads `UVInpaintingNet` after parser splatting, then writes the completed skin to `outputs/pred_uv.png`.

## Semantic Fixed-View UV Reconstruction

`train_semantic_uv_reconstruction.py` is a separate training framework for learning the inverse
mapping directly from clean, fixed front/back renders. It does not consume
`dense_uv_parser` conditioning and does not measure the rendered cuboid to decide
the layer. Each view passes through two complementary branches:

- a trainable high-resolution CNN whose `/8` detail tokens preserve grid edges,
  colors, and exact local texture, alongside coarser `/16` context tokens;
- a frozen SigLIP2 vision tower that supplies continuous, language-aligned
  patch and global features without defining a garment vocabulary.

Learned UV queries jointly attend to both branches from both views. Architecture
version 2 uses a 32x32 UV query grid and a learned PixelShuffle decoder to reach
64x64; it does not bilinearly enlarge 16x16 features. Full-UV RGB supervision is
weighted more strongly and an explicit UV RGB-gradient loss penalizes blurred
texel boundaries. SigLIP2 is not used to generate
captions or closed labels. Concepts such as caps, hoods, scarves, unusual
costumes, and concepts absent from the skin dataset remain positions in the
continuous pretrained embedding space rather than entries in a finite table.

The default run intentionally has no render randomization:

- the differentiable renderer produces only the configured canonical views;
- no camera, perspective, translation, scale, body-part, or outer-thickness
  perturbation is applied;
- validation uses the same fixed rendering contract;
- all source skins are used unless `MAX_SAMPLES` is explicitly set.

Start training with:

```bash
./run_semantic_uv_reconstruction_training.sh
```

The launcher auto-detects `mappings_256x512` whether
`differentiable_minecraft_renderer` is inside `SkingToolkit`, beside it, or
under the legacy `github/` workspace directory. Override detection when the
renderer lives elsewhere:

```bash
MAPPINGS_DIR=/absolute/path/to/mappings_256x512 \
./run_semantic_uv_reconstruction_training.sh
```

The selected directory must contain
`walk_front_both_layer_ortho_mapping.pt` and
`walk_back_both_layer_ortho_mapping.pt`. If they do not exist, generate them
from the renderer repository:

```bash
python generate_mappings.py \
  --views walk_front_both_layer_ortho,walk_back_both_layer_ortho \
  --sizes 256x512
```

The semantic bottleneck has exact supervision that can be derived without
manual labels:

- occupied inner/outer UV texels grouped by six body parts;
- outer-layer presence and coverage for head, torso, arms, and legs;
- mean visible color for all twelve part/layer combinations;
- full UV RGB and outer alpha;
- differentiable front/back pixel/alpha re-render consistency;
- cosine consistency between the frozen SigLIP2 embeddings of the input views
  and the predicted UV's differentiable re-renders.

The only finite default classes describe Minecraft's known atlas topology:
transparent valid texel, six occupied inner body parts, and six occupied outer
body parts. They are structural targets, not visual concepts such as `cap` or
`scarf`.

Dense teacher labels are single-channel 64x64 PNG files whose pixel values are
class IDs and whose value `255` means ignore. Filenames must match the source
skins. This extension is only for extra structural topology, never for a list
of appearance concepts. Configure both the directory and structural class
count:

```bash
SEMANTIC_LABELS_DIR=/path/to/semantic_uv_labels \
SEMANTIC_CLASSES=24 \
./run_semantic_uv_reconstruction_training.sh
```

Checkpoints and previews are written to `runs/semantic_uv_reconstruction_v*/`;
preview rows are
the input views, predicted UV, ground-truth UV, predicted outer alpha, and
ground-truth outer alpha.

Architecture version 2 is intentionally incompatible with checkpoints created
by the former 16x16-query/bilinear decoder. Do not set `RESUME` to one of those
checkpoints: start a new versioned run. The startup summary must show
`"architecture_version": 2`, `"query_size": 32`, `"lambda_uv_rgb": 2.0`, and
`"lambda_uv_edge": 1.0`. Epoch metrics additionally report `loss_uv_edge` and
`rgb_mae_255`; the latter is the occupied-texel RGB MAE expressed on a 0-255
scale and should keep falling along with visual sharpness.

### Remote training environment

The default semantic backbone is the frozen FixRes checkpoint
`google/siglip2-base-patch16-224`. FixRes is intentional: the implementation
uses one differentiable aspect-preserving letterbox for source and predicted
renders, so the semantic re-render loss can propagate into the UV decoder.
NaFlex checkpoints are rejected by this adapter.

Install the additional dependencies on the training computer:

```bash
pip install -U transformers sentencepiece safetensors
```

The first run downloads the Hugging Face checkpoint. To use an already cached
copy on an offline training machine:

```bash
SIGLIP_LOCAL_FILES_ONLY=true ./run_semantic_uv_reconstruction_training.sh
```

The frozen SigLIP2 weights are deliberately excluded from each epoch
checkpoint; resuming reloads them from the Hugging Face cache and restores only
the trainable CNN, fusion, query, and decoder weights. The long-running,
VRAM-rich default uses batch size 4. If the 32x32 query grid, `/8` detail memory,
and semantic re-render pass exceed available VRAM, first reduce
`BATCH_SIZE`; as a fallback set `LAMBDA_SIGLIP_RENDER=0` while retaining SigLIP2
tokens in the forward model. For a dependency-free structural ablation use:

```bash
SEMANTIC_BACKBONE=none LAMBDA_SIGLIP_RENDER=0 ./run_semantic_uv_reconstruction_training.sh
```

### Training throughput

The default launcher keeps the high-resolution texture path but avoids several
unnecessary costs:

- padded SigLIP2 letterbox patches are removed before UV cross-attention, which
  also allows fused scaled-dot-product attention on supported CUDA builds;
- the expensive differentiable SigLIP2 re-render cycle runs once every four
  batches and is multiplied by four when used, preserving its expected loss
  weight;
- CUDA uses channels-last convolution storage and fused AdamW when available;
- progress values are copied from CUDA only every 50 batches instead of forcing
  a device synchronization after every optimizer step;
- preview inference uses the configured mixed precision.

The startup summary reports `siglip_render_every`, `channels_last`,
`fused_optimizer`, and `torch_compile`. The default long-running configuration
is equivalent to:

```bash
BATCH_SIZE=4 SIGLIP_RENDER_EVERY=4 LOG_EVERY=50 \
TORCH_COMPILE=true COMPILE_MODE=reduce-overhead \
./run_semantic_uv_reconstruction_training.sh
```

Every train/validation metric block includes `epoch_seconds` and
`samples_per_second`, making before/after throughput directly comparable on the
remote GPU. Set `LOG_EVERY` to change the progress refresh interval.

Use `SIGLIP_RENDER_EVERY=1` to compute the semantic cycle on every batch. For a
faster run that still uses frozen SigLIP2 tokens to understand the source views,
disable only the predicted-render cycle:

```bash
LAMBDA_SIGLIP_RENDER=0 ./run_semantic_uv_reconstruction_training.sh
```

PyTorch compilation is enabled by default. Its first batches are slower while
kernels are generated, after which long runs can benefit. If the remote
PyTorch/CUDA combination produces graph breaks or a compiler error, disable it:

```bash
TORCH_COMPILE=false ./run_semantic_uv_reconstruction_training.sh
```

Set `FUSED_OPTIMIZER=false` only when diagnosing an optimizer compatibility
problem. Unsupported fused AdamW implementations automatically fall back to the
standard optimizer.

## Generate Render Pairs

```bash
python SkingToolkit/semantic_uv_reconstruction/generate_pairs.py \
  --data_dir /path/to/gt_skins \
  --output_dir /path/to/render_pairs \
  --views static_front,static_back,top_front_45,top_back_45 \
  --combined
```
