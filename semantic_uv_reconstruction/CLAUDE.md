# Semantic UV Reconstruction Development Guide

## Overview

`semantic_uv_reconstruction` contains the primary open-semantic fixed-view
render-to-UV trainer and the compatible parser-conditioned completion trainer.
The former fuses a high-resolution CNN with frozen SigLIP2 features; the latter
retains `UVInpaintingNet` so existing checkpoints remain usable.

The direct reconstructor's architecture version 2 uses `/8` CNN detail memory,
32x32 learned UV queries, PixelShuffle upsampling, and UV RGB edge supervision.
Version-1 direct checkpoints used 16x16 queries and bilinear upsampling and must
not be resumed into version 2.

The direct launcher defaults to batch size 4, a four-step SigLIP render-cycle
interval, log refresh every 50 batches, compiled execution, compacted letterbox
tokens, channels-last, and fused AdamW. `SIGLIP_RENDER_EVERY=1` restores
per-batch cycle evaluation; `TORCH_COMPILE=false` is the compatibility fallback.

## Commands

From the workspace root:

```bash
cd SkingToolkit/semantic_uv_reconstruction
./run_semantic_uv_reconstruction_training.sh
```

For the parser-conditioned compatibility path:

```bash
python SkingToolkit/semantic_uv_reconstruction/train.py \
  --data_dir /path/to/skins \
  --parser_checkpoint SkingToolkit/dense_uv_parser/runs/dense_uv_parser_v1/best.pt \
  --output_dir SkingToolkit/semantic_uv_reconstruction/runs/semantic_uv_reconstruction_full_v1
```

The launcher is preferred because it selects the newest parser checkpoint and assigns a versioned run directory:

```bash
cd SkingToolkit/semantic_uv_reconstruction
./run_parser_conditioned_training.sh
```

The full production inference path starts in the parser module:

```bash
cd SkingToolkit/dense_uv_parser
FRONT=/path/to/front.png BACK=/path/to/back.png ./run_infer.sh
```

## Parser-Conditioned Data Flow

```text
GT skin (64×64 RGBA)
  → DifferentiableRenderer fixed views
  → frozen DenseUVParserNet
  → parser splat
  → 10-channel UV conditioning
       [inner RGBA + known mask, outer RGBA + known mask]
  → UVInpaintingNet
  → completed 64×64 RGBA skin
  → differentiable render consistency losses
```

`UVInpaintingDataset` loads ground-truth skins. Conditioning is generated inside `train.run_epoch()` so render augmentation and parser background randomization can be applied online. Training must use the same parser views, routing settings, mappings, and view order as inference.

## Model

`UVInpaintingNet` is a three-level U-Net-style network with GroupNorm, SiLU activations, skip connections, and spatial self-attention at 32×32 and 16×16. It always predicts four sigmoid-clamped RGBA channels.

When `preserve_known=True`, observed texels from the 10-channel conditioning are copied into the final prediction. Network output is therefore used only for unknown texels. `LightUVInpaintingNet` remains an alias of the same architecture.

## Losses

`UVInpaintingLoss` combines:

1. Alpha-masked UV RGB reconstruction.
2. Alpha BCE and Dice supervision.
3. Alpha-edge reconstruction.
4. Differentiable multi-view RGB and alpha rendering consistency.
5. UV-space RGB edge reconstruction.
6. Optional PatchGAN generator loss.

Inner texels covered by opaque outer-layer texels are ignored by default because they cannot be verified from the configured views. Use `--supervise_covered_inner` only when that behavior is intentional.

## Checkpoints

Checkpoints contain a plain model `state_dict`, optimizer state, arguments, and metrics. Renaming the Python package or model class does not change parameter keys, so checkpoints produced under the former package name remain loadable when their path is supplied explicitly.

The launcher creates run directories named `runs/semantic_uv_reconstruction_<model>_vN`. `best.pt` defaults to the lowest `loss_recon_total`, keeping checkpoint selection independent of optional GAN oscillation.

## Important Files

- `dataset.py`: skin loading, render helpers, UV aggregation, and augmentation.
- `model.py`: `UVInpaintingNet` and `PatchGANDiscriminator`.
- `losses.py`: UV, alpha, edge, render, and GAN losses.
- `train.py`: parser-conditioned training and checkpointing.
- `train_semantic_uv_reconstruction.py`: open-semantic direct reconstruction.
- `run_parser_conditioned_training.sh`: standard training configuration.
- `run_semantic_uv_reconstruction_training.sh`: primary direct training entry.

`dense_uv_parser/infer.py` is the only inference entry point, ensuring input construction exactly matches training.
