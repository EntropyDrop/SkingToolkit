# SkingToolkit

`SkingToolkit` reconstructs a standard 64×64 RGBA Minecraft skin UV atlas
from normalized fixed front/back character renders:

```text
front/back renders
  → deterministic foreground extraction
  → fixed Steve geometry + semantic-conditioned Dense UV Parser
  → visible inner/outer/secondary UV evidence
  → per-part deterministic topology repair of missing inner texels
  → final 64×64 RGBA skin
```

The pipeline uses one trainable component and one checkpoint:
`dense_uv_parser`. Visible colors are selected directly from fitted source
grid cells. Final repair never generates a new color, never copies across body
parts, and never creates or overwrites outer-layer texels.

## Project Structure

```text
SkingToolkit/
├── renderer.py
├── dense_uv_parser/
│   ├── model.py
│   ├── losses.py
│   ├── utils.py
│   ├── uv_layout.py
│   ├── uv_topology.py
│   ├── simple_inpainting.py
│   ├── skin_dataset.py
│   ├── semantic_backbone.py
│   ├── semantic_cache.py
│   ├── cache_semantic_features.py
│   ├── train.py
│   ├── infer.py
│   ├── run_dense_uv_parser_training.sh
│   └── run_infer.sh
```

## Dense UV Parser

The parser classifies source pixels as background, directly visible inner skin,
directly visible outer skin, or a secondary/deeper renderer surface. Fixed Steve
geometry supplies body-part, face, exact surface, and UV candidates. A frozen
SigLIP2 backbone supplies cached spatial features and fused front/back semantic
context; TIPSv2 remains available as an online ablation.

Training combines supervised foreground/route/surface losses with
differentiable soft-UV and multi-view rendering losses. The renderer and parser
therefore use the same fixed camera-to-UV mappings in training, validation, and
inference.

The parser produces internal conditioning in this format:

```text
[inner RGBA + evidence, outer RGBA + evidence]
```

Missing inner texels are repaired deterministically, one body part at a time.
Front/back faces use border-to-centre rings; left/right faces use rows from both
edges toward the centre and prefer same-row sources; top/bottom faces use
border-to-centre rings. An available mirrored texel is preferred, followed by
the closest currently defined texel in canonical 3D space from the same part.

See [dense_uv_parser/README.md](dense_uv_parser/README.md) for loss,
routing, cache, and diagnostic details.

## Requirements

```bash
pip install torch torchvision tqdm pillow numpy
pip install -U transformers sentencepiece safetensors
```

The training dataset must contain standard 64×64 skin images. Slim/Alex skins
are normalized to the Steve layout. Renderer mapping files must match the
configured fixed view names and sizes.

## Training

```bash
cd SkingToolkit/dense_uv_parser
./run_dense_uv_parser_training.sh
```

The launcher builds or reuses frozen SigLIP2 pooled and spatial feature caches
through `cache_semantic_features.py`, then trains a versioned
`runs/dense_uv_parser_vN` run. No second model needs to be trained.

## Inference

```bash
cd SkingToolkit/dense_uv_parser
COMBINED=/path/to/front_back.png ./run_infer.sh
```

The launcher automatically chooses the newest compatible parser checkpoint.
Important outputs are:

- `outputs/foreground_cutout.png`: flood-filled transparent foreground.
- `outputs/parser_pred_uv.png`: visible parser evidence; unknown texels remain
  transparent.
- `outputs/parser_pred_uv_simple_inpainting.png`: deterministic repair
  artifact.
- `outputs/pred_uv.png`: final UV, identical to the deterministic repair.
- `outputs/parser_debug_geometry_overlay.png`: fitted geometry projected on
  the source views.
- `outputs/parser_debug_geometry_routed_overlay.png`: routed source pixels
  inside their selected layer grids.

Override checkpoint selection when needed:

```bash
PARSER_CHECKPOINT=runs/dense_uv_parser_v3/best.pt \
COMBINED=/path/to/front_back.png \
./run_infer.sh
```

Use `PARSER_ONLY=true` to export parser evidence and diagnostics without final
repair.

## Tests

From the directory containing `SkingToolkit`:

```bash
python -m unittest discover -s SkingToolkit/dense_uv_parser -p 'test_*.py'
```
