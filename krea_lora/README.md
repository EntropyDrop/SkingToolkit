# Krea-2-Raw Minecraft Preview LoRA

## Recommended: SKING_DDJ paired conditional LoRA

`configs/ddj_conditional.json` is the real reference-image training path. It
uses 8,087 matching `_source` and `_result` records from
`/home/ds/llms/SKING_DDJ_Dataset`. Existing historical `_edited` files are not
used as ground truth because many contain gradients, smooth detail, or older
camera layouts. Every target is regenerated from the valid 64x64 RGBA result
skin with the renderer's `front_left` and `back_left` mappings.

This makes all target pixels consequences of Minecraft's base/outer UV layers:
there are no handheld meshes, capes, lighting effects, or geometry outside the
skin model. The target background is one uniform solid blue.

The conditional transformer sequence is:

```text
[text | noisy edited-target latent (t/h/w region 0) | clean source-image latent (region 1)]
```

Both images are compressed by the frozen Qwen Image VAE. Krea2 attention LoRA
learns to read the clean source tokens while predicting flow only for the
target tokens. The source is therefore present at every denoising step; this is
not ordinary high-strength img2img and does not depend on a generated text
caption.

Run the paired workflow remotely:

```bash
cd /home/ds/llms/SkingToolkit/krea_lora
bash scripts/11_prepare_ddj_pairs.sh
bash scripts/12_cache_ddj_prompt.sh
bash scripts/13_cache_ddj_latents.sh
bash scripts/14_train_ddj_conditional.sh
```

Or run the idempotent sequence with:

```bash
bash scripts/run_ddj_conditional.sh
```

Generate from an arbitrary image after training:

```bash
bash scripts/15_generate_ddj_conditional.sh \
  --source /path/to/reference.png \
  --output /path/to/mc_preview.png
```

The paired one-step integration test is `bash scripts/smoke_test_ddj.sh`.
Production training defaults to rank 32, 6,000 steps, effective batch 4, and a
65GB free-VRAM launch guard. Its final adapter is written to
`runs/ddj_conditional_raw_lora/final/pytorch_lora_weights.safetensors`.

The earlier `mc_preview.json` workflow below remains useful as an unpaired
camera/layout LoRA, but it cannot reproduce an arbitrary source identity as
reliably as the paired workflow.

This subproject trains a LoRA on the Krea2 transformer to make the requested
Minecraft preview layout more stable:

- the left half is a fixed front-left orthographic view;
- the right half is the matching back-right orthographic view;
- both views use the same neutral pose, scale, and camera elevation;
- all targets are rendered from valid 64x64 RGBA Minecraft base/outer layers;
- the background is pure white and no unsupported geometry is present.

It runs on the remote host under `/home/ds/llms/SkingToolkit/krea_lora` and uses
`/home/ds/llms/krea/Krea-2-Raw` as the frozen base model. The setup clones the
existing `krea` Conda environment into `krea-train`; it never modifies `c130`.

## Why this is a geometry LoRA

The 498k source skins have exact UV data but no reliable natural-language
appearance captions. Phase 1 therefore uses one task token (`mc45preview`) and
one constant format caption. A low-rank, low-learning-rate LoRA learns camera,
layout, block geometry, white background, and the visual consequences of valid
inner/outer layers while the frozen base model retains most text semantics.

This improves compliance statistically; it cannot mathematically guarantee a
valid hidden 64x64 UV atlas because the requested output is a rendered preview.
A later UV-atlas generator or parser should be used when a guaranteed reusable
skin file is the final artifact.

## Remote workflow

Run these from `/home/ds/llms/SkingToolkit/krea_lora`:

```bash
bash scripts/create_env.sh
bash scripts/01_prepare_dataset.sh
bash scripts/02_cache_prompt.sh
bash scripts/03_cache_latents.sh
bash scripts/04_train.sh
bash scripts/05_validate.sh
```

After reviewing the configuration, the same sequence can be run idempotently
with `bash scripts/run_all.sh`.

The default configuration renders 20,000 targets at 512x512. Start there;
increase `data.max_images` only after the first validation comparison. Prompt
embeddings and VAE latents are cached before training, so the 9GB Qwen3-VL text
encoder and Qwen Image VAE are not resident during the transformer LoRA loop.

The full one-step integration test is:

```bash
bash scripts/smoke_test.sh
```

It creates eight deterministic targets, caches prompt/latent tensors, executes
one backward/update step, and verifies that a Diffusers-compatible Krea2 LoRA
file is written.

## Important configuration

Edit `configs/mc_preview.json` before a production run:

- `data.training_prompt`: editable trigger/format prompt used for training;
- `data.max_images`: start with 20k, then consider 50k if validation improves;
- `training.rank`: 16 is the conservative first run; 32 is a later option;
- `training.layerwise_casting`: keep `false` with a free 96GB GPU; set `true`
  only when FP8 base-weight storage is needed to coexist with another job;
- `training.max_train_steps`: 2k-4k is the recommended first sweep;
- `training.learning_rate`: keep around `1e-4`; reduce if character semantics degrade;
- `validation.prompts`: fixed prompts used to compare checkpoints.

The selected LoRA targets are the image/text joint transformer's attention
projections (`to_q`, `to_k`, `to_v`, `to_out.0`). Text encoder and VAE weights
remain frozen.

## Production checks

Keep the Krea web service stopped or on a different GPU while training. Check
free VRAM before launch:

```bash
nvidia-smi
nvidia-smi pmon -s um -d 1
```

The production launcher requires 50GB free VRAM by default and exits instead
of competing with an existing job. Override `KREA_MIN_FREE_MB` only after
deliberately enabling `training.layerwise_casting` or measuring your run.

The final adapter is saved to:

`/home/ds/llms/SkingToolkit/krea_lora/runs/mc_preview_raw_lora/final/pytorch_lora_weights.safetensors`

Load it with `Krea2Pipeline.load_lora_weights(...)`. Use the Raw checkpoint's
defaults for evaluation: 28 steps and guidance 4.5.
