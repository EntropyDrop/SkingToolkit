# Semantic UV Reconstruction Development Guide

> Experimental/legacy module. Production Dense UV Parser training and
> inference no longer import this package. The default final UV is repaired by
> `dense_uv_parser/simple_inpainting.py` and requires only a parser checkpoint.

## Overview

`semantic_uv_reconstruction` contains the primary open-semantic fixed-view
render-to-UV trainer and a geometry-first parser-conditioned completion trainer.
The former fuses a high-resolution CNN with frozen SigLIP2 features; the latter
defaults to `TopologyAwareUVCompletionNet` while retaining `UVInpaintingNet`
for existing checkpoints.

The direct reconstructor's architecture version 3 uses `/8` CNN detail memory,
256 Perceiver-style memory latents, 32x32 learned UV queries with cheap spatial
mixers, PixelShuffle upsampling, and UV RGB edge supervision. Version-1/2 direct
checkpoints must not be resumed into version 3.

The direct launcher defaults to a memory-mapped frozen SigLIP global cache,
batch size 4, 256 memory latents, a four-step SigLIP render-cycle interval after
two warmup epochs, doubled RGB/edge weight during those warmup epochs, log
refresh every 50 batches, compiled execution, channels-last, and fused AdamW.
`SIGLIP_RENDER_EVERY=1` restores per-batch cycle evaluation after warmup.
Compilation defaults to
`max-autotune-no-cudagraphs`; `TORCH_COMPILE=false` is the compatibility
fallback.

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
  --completion_model topology_maskgit \
  --output_dir SkingToolkit/semantic_uv_reconstruction/runs/semantic_uv_reconstruction_topology_maskgit_v1
```

The launcher is preferred because it selects the newest parser checkpoint and assigns a versioned run directory:

```bash
cd SkingToolkit/semantic_uv_reconstruction
./run_parser_conditioned_training.sh
```

The production inference path starts in the parser module and does not load a
semantic reconstruction checkpoint:

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
  → 12-channel confidence-aware UV conditioning
       [inner RGBA + evidence + confidence,
        outer RGBA + evidence + confidence]
  → cuboid topology graph + inner/outer correspondence edges
  → TopologyAwareUVCompletionNet iterative masked generation
  → completed 64×64 RGBA skin
  → differentiable render consistency losses
```

`UVInpaintingDataset` loads ground-truth skins. Conditioning is generated inside `train.run_epoch()` so render augmentation and parser background randomization can be applied online. Training must use the same parser views, routing settings, mappings, and view order as inference.

## Model

`TopologyAwareUVCompletionNet` assigns each valid texel a layer, part, face,
face-local coordinate, four cuboid-surface neighbours, and an exact paired-layer
texel. Graph blocks combine local seam-aware attention with global attention
over 72 part/face/layer surface tokens. RGB uses 256-way categorical heads and
outer alpha uses a binary head; inference reveals unknown texels iteratively.

Evidence at or above the checkpoint's hard-lock threshold is copied exactly.
Lower-confidence parser evidence conditions the topology model but may be
regenerated. Legacy 10-channel topology checkpoints retain their original
hard-known behavior. `UVInpaintingNet` remains a three-level continuous U-Net
and `LightUVInpaintingNet` remains its compatibility alias.

## Losses

`UVInpaintingLoss` combines:

1. Alpha-masked UV RGB reconstruction.
2. Alpha BCE and Dice supervision.
3. Alpha-edge reconstruction.
4. Differentiable multi-view RGB and alpha rendering consistency.
5. UV-space RGB edge reconstruction.
6. Optional PatchGAN generator loss.
7. Unknown-only categorical RGB and binary-alpha token losses for the topology model.

Inner texels covered by opaque outer-layer texels are ignored by default because they cannot be verified from the configured views. Use `--supervise_covered_inner` only when that behavior is intentional.

## Checkpoints

Checkpoints contain a plain model `state_dict`, optimizer state, arguments,
metrics, and a `model_config` that lets inference reconstruct the correct
completion architecture. Checkpoints without `model_config` continue to load as
legacy U-Net checkpoints.

The launcher creates run directories named `runs/semantic_uv_reconstruction_<model>_vN`. `best.pt` defaults to the lowest `loss_recon_total`, keeping checkpoint selection independent of optional GAN oscillation.

## Important Files

- `dataset.py`: skin loading, render helpers, UV aggregation, and augmentation.
- `model.py`: `UVInpaintingNet` and `PatchGANDiscriminator`.
- `topology.py`: Steve atlas metadata, cuboid seam graph, and layer pairing.
- `topology_model.py`: topology-aware discrete masked generator.
- `losses.py`: UV, alpha, edge, render, and GAN losses.
- `train.py`: parser-conditioned training and checkpointing.
- `train_semantic_uv_reconstruction.py`: open-semantic direct reconstruction.
- `cache_siglip_globals.py`: compatibility entry point for
  `dense_uv_parser/cache_semantic_features.py`.
- `run_parser_conditioned_training.sh`: standard training configuration.
- `run_semantic_uv_reconstruction_training.sh`: primary direct training entry.

`dense_uv_parser/infer.py` is the only inference entry point, ensuring input construction exactly matches training.
