# SkingToolkit

`SkingToolkit` reconstructs a standard 64×64 RGBA Minecraft skin UV atlas from fixed front/back character renders. The current pipeline separates geometric parsing from texture completion:

```text
front/back renders
  → dense_uv_parser
  → partial inner/outer UV conditioning
  → semantic_uv_reconstruction
  → complete 64×64 RGBA skin
```

The shared differentiable renderer uses precomputed camera-to-UV mappings and PyTorch operations, allowing render-space losses to backpropagate into both training stages.

## Project Structure

```text
SkingToolkit/
├── renderer.py
├── dense_uv_parser/
│   ├── model.py
│   ├── losses.py
│   ├── utils.py
│   ├── train.py
│   ├── infer.py
│   ├── run_dense_uv_parser_training.sh
│   └── run_infer.sh
└── semantic_uv_reconstruction/
    ├── model.py
    ├── losses.py
    ├── dataset.py
    ├── train.py
    ├── run_parser_conditioned_training.sh
    ├── semantic_backbone.py
    ├── semantic_model.py
    ├── semantic_losses.py
    ├── train_semantic_uv_reconstruction.py
    └── run_semantic_uv_reconstruction_training.sh
```

## Components

### `dense_uv_parser`

The parser classifies render pixels as background, directly visible inner skin, directly visible outer skin, or a secondary/deeper renderer surface. Fixed Steve geometry supplies the exact body-part, face, and UV mappings. Its predictions are splatted into a 10-channel UV conditioning tensor:

```text
[inner RGBA + known mask, outer RGBA + known mask]
```

Training includes supervised routing losses plus a differentiable soft-UV and multi-view rendering branch. See [dense_uv_parser/README.md](dense_uv_parser/README.md) for configuration and diagnostics.

### `semantic_uv_reconstruction`

The module's primary model reconstructs a complete atlas directly from fixed
front/back renders by fusing a high-resolution CNN with frozen SigLIP2
semantics. It also retains the parser-conditioned completion model for existing
checkpoints and production inference.

The public classes are `UVInpaintingNet`, `UVInpaintingDataset`, and `UVInpaintingLoss`. See [semantic_uv_reconstruction/README.md](semantic_uv_reconstruction/README.md) for all training options.

The same directory includes `SemanticUVReconstructor`, an independent
fixed-view training path that jointly encodes clean front/back renders and
predicts the complete atlas directly. It fuses a high-resolution CNN with a
frozen, language-aligned SigLIP2 vision tower, uses source-skin-derived
structural attributes and differentiable pixel/semantic re-render losses, and
has no finite garment concept table. This path does not require a parser
checkpoint and currently applies no randomized render variation.

### `renderer.py`

`DifferentiableRenderer` loads view mapping files and renders Minecraft skins through `torch.nn.functional.grid_sample` and alpha compositing. Both parser and inpainting losses use the same mappings as inference.

## Requirements

Install the core Python dependencies:

```bash
pip install torch torchvision tqdm pillow numpy
```

The optional semantic fixed-view trainer additionally requires:

```bash
pip install -U transformers sentencepiece safetensors
```

The skin dataset must contain standard 64×64 PNG files. Slim/Alex skins are normalized to the standard Steve layout when `mc_skin_utils` is available. Renderer mapping files must match the configured view names and image sizes.

## Training

Train the dense parser first:

```bash
cd SkingToolkit/dense_uv_parser
./run_dense_uv_parser_training.sh
```

Then train Semantic UV reconstruction from the parser-generated conditioning:

```bash
cd ../semantic_uv_reconstruction
./run_parser_conditioned_training.sh
```

Or train the semantic fixed-view render-to-UV model directly from all source
skins:

```bash
./run_semantic_uv_reconstruction_training.sh
```

The parser-conditioned launcher automatically selects the newest
`dense_uv_parser_v*/best.pt` unless `PARSER_CHECKPOINT` is set explicitly.

## Inference

Run the complete parser plus inpainting pipeline from `dense_uv_parser`:

```bash
cd SkingToolkit/dense_uv_parser
FRONT=/path/to/front.png BACK=/path/to/back.png ./run_infer.sh
```

By default the launcher selects the newest parser checkpoint and the newest `../semantic_uv_reconstruction/runs/semantic_uv_reconstruction_full_v*/best.pt`. Important outputs include:

- `outputs/parser_pred_uv.png`: preliminary UV assembled from parser-visible texels.
- `outputs/parser_conditioning.png`: inner/outer conditioning preview.
- `outputs/parser_debug_geometry_overlay.png`: fitted geometry projected onto the source renders.
- `outputs/pred_uv.png`: completed UV atlas when an inpainting checkpoint is available.

You can override checkpoint selection explicitly:

```bash
PARSER_CHECKPOINT=runs/dense_uv_parser_v3/best.pt \
INPAINT_CHECKPOINT=../semantic_uv_reconstruction/runs/semantic_uv_reconstruction_full_v4/best.pt \
./run_infer.sh
```

## Tests

From the directory containing `SkingToolkit`:

```bash
python -m unittest discover -s SkingToolkit -p 'test*.py'
```
