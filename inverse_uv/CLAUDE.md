# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`inverse_uv` is a submodule of `SkingToolkit` that trains a supervised U-Net to reconstruct 64×64 Minecraft skin UV sheets from multi-view 2D camera renders. The core insight is **coordinate-guided unprojection**: instead of learning the difficult render→UV spatial mapping from scratch, the dataset uses precomputed camera mapping grids (`DifferentiableRenderer` `*.pt` files) to unproject each pixel into UV space, converting the problem into UV-space inpainting/denoising that U-Nets handle well.

## Running from the correct directory

All scripts manipulate `sys.path` to import from the workspace root (parent of `SkingToolkit/`). Commands in the README assume you run from the workspace root:

```
cd /Users/ha/Documents/entropydrop_website
python SkingToolkit/inverse_uv/train.py ...
```

The shell scripts `run_inverse_uv_training.sh` and `run_inverse_uv_infer.sh` `cd` to their own directory first, so they can be invoked from anywhere.

## Key commands

### Training

```bash
# Via shell script (recommended) — configure via env vars:
MODEL=light DATA_DIR=../skins EPOCHS=50 bash run_inverse_uv_training.sh

# Direct Python:
python SkingToolkit/inverse_uv/train.py \
  --data_dir /path/to/gt_skins \
  --output_dir runs/inverse_uv_static \
  --views static_front,static_back,top_front_45,top_back_45 \
  --batch_size 16 --epochs 20
```

### Inference

```bash
CHECKPOINT=runs/inverse_uv_light_v1/best.pt bash run_inverse_uv_infer.sh

# Direct:
python SkingToolkit/inverse_uv/infer.py \
  --checkpoint runs/inverse_uv_static/best.pt \
  --combined /path/to/combined_views.png \
  --output /path/to/pred_uv.png
```

### Generate render pairs (precompute training data)

```bash
python SkingToolkit/inverse_uv/generate_pairs.py \
  --data_dir /path/to/gt_skins \
  --output_dir /path/to/render_pairs \
  --views static_front,static_back --combined
```

## Architecture

### Data flow

```
GT skin (64×64 RGBA)
  → DifferentiableRenderer.forward_view() → multi-view renders (per view)
  → unproject_renders_to_uv()            → 10-channel conditioning (64×64)
      [inner: RGBA + known-mask, outer: RGBA + known-mask]
  → InverseUVNet / LightInverseUVNet     → predicted 64×64 RGBA UV
  → apply_uv_mask()                      → masked to valid skin regions
```

### Input conditioning (10 channels)

Built by `dataset.unproject_renders_to_uv()`: two layers (inner, outer) × 5 channels (RGB + Alpha + known-mask). The known-mask indicates which UV texels received pixel data from the renders. This is always 10 channels regardless of view count. Render pixels mapping to each UV texel are aggregated via `--unproject_mode` (`mean`=average, `mode`=most frequent color, `medoid`=spatial median). Batched training currently supports `mean`; `mode`/`medoid` are for unbatched inference or debugging.

### Model variants (`model.py`)

| Variant | Class | ~Params | Key traits |
|---------|-------|---------|------------|
| Light (default) | `LightInverseUVNet` | ~1M | 2 downsample levels, ConvBlock, no attention, CoordConv at stem only |
| Full | `InverseUVNet` | ~14M | 3 downsample levels, ResBlocks, bottleneck self-attention, multi-scale CoordConv |

Both models force input to 64×64 via interpolation and output sigmoid-clamped RGBA.

Key architectural primitives in `model.py`:
- `CoordConv` — appends normalized x/y coordinate channels so the model sees absolute UV position
- `SpatialSelfAttention` — lightweight bottleneck self-attention with zero-init projection
- `ConvBlock` / `ResBlock` — basic building blocks with GroupNorm + SiLU
- `DownBlock` / `UpBlock` — strided conv down, bilinear upsample with skip connections
- `PixelShuffleUpBlock` — alternative upsampling for sharper outputs (light model only)

### Loss function (`losses.py`)

`InverseUVLoss` computes a weighted sum of 5 terms:

1. **Alpha-masked RGB L1** (`loss_rgb`) — RGB error only on visible UV texels
2. **Alpha BCE** (`loss_alpha`) — binary cross-entropy on alpha channel
3. **Differentiable render consistency L1** (`loss_render`) — re-renders the predicted skin through `DifferentiableRenderer` and compares against GT renders; this provides a cross-domain signal that resolves overlapping layer ambiguity
4. **Edge L1** (`loss_edge`) — gradient-domain L1 for sharp pixel boundaries
5. **SSIM** (`loss_ssim`) — masked structural similarity (off by default for light model)

The loss also handles **covered inner layer masking**: inner-layer UV texels hidden behind opaque matching outer-layer texels can be excluded from supervision (`--supervise_covered_inner` to include them).

### Dataset (`dataset.py`)

`InverseUVDataset` loads GT skins, normalizes Alex→Steve models via `mc_skin_utils.alice_to_steve`, and optionally builds conditioning tensors. The dataset does NOT store conditioning — conditioning is built on-the-fly during training inside `train.run_epoch()` using `build_conditioning()` so augmentation can be applied to renders before unprojection.

`RenderAugmenter` applies online augmentation in render space (before UV unprojection): random translation, perspective warp, and elastic grid distortion. This is key for generalization because the distortion patterns in render space translate to realistic UV-space variations.

### Training loop (`train.py`)

- `WarmupCosineScheduler` — linear warmup then cosine decay
- Mixed precision via `torch.autocast` + `GradScaler` (fp16 on CUDA, bf16 where supported)
- Conditioning is built inside the training loop (not in `__getitem__`) so data augmentation transforms renders before unprojection
- Checkpoints save model config alongside weights — `infer.py` reconstructs the exact architecture from checkpoint metadata
- Model-adaptive defaults: `apply_model_defaults()` sets different LRs, lambdas, and channel widths depending on `--model light` vs `--model full`

### Inference (`infer.py`)

Reconstructs the model architecture from checkpoint metadata (supports multiple checkpoint formats for backward compatibility). Accepts either individual `--view_images`, a `--combined` side-by-side image, or legacy `--front`/`--back` flags.

## Key dependencies

- **`SkingToolkit.renderer.DifferentiableRenderer`** (`renderer.py` in the parent directory) — PyTorch differentiable Minecraft skin renderer; loads per-view UV mapping grids from `*.pt` files. The mappings directory is typically `../../github/differentiable_minecraft_renderer/mappings`.
- **`mc_skin_utils`** — external package providing `alice_to_steve()` for normalizing slim/Alex models to the standard Steve layout.

## Common conventions

- Model type defaults to `"light"` in training; shell scripts default to `MODEL=full`.
- The training shell script auto-increments `RUN_NAME` version if the directory exists.
- The inference shell script auto-detects the latest version checkpoint.
- `--coordconv`, `--bottleneck_attention`, `--resnet`, `--pixelshuffle`, `--multi_scale_coord` use `BooleanOptionalAction` (Python 3.9+), so they're toggled with `--coordconv` / `--no-coordconv`.
- `--render_size` and `--include_alpha` are deprecated; UV unprojection always uses native mapping sizes and RGBA+mask conditioning.
