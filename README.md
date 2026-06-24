# SkingToolkit: Flux2Klein Differentiable Training Framework

`SkingToolkit` is a custom fine-tuning and training framework for **Flux2Klein** Minecraft Skin Generator models. It integrates a **Differentiable Renderer** directly into the PyTorch training pipeline, allowing gradients from rendering losses (visual differences on 3D characters) to flow backwards through the VAE and optimize the Flux model's weights.

---

## 🚀 Key Features

* **🎨 Differentiable Rendering Backpropagation**: Uses PyTorch `F.grid_sample` to warp the flat predicted `64x64` skin texture maps into multi-view 2D renders (such as `static_front` and `static_back`). The entire rendering operation is mathematically differentiable, enabling rendering losses to guide the texture generation.
* **📐 Top-to-Bottom Target VAE Layout `[RGB | Alpha]`**: Resolves the VAE's native 3-channel (RGB) limitation by packing the target `64x64` RGBA skin into a `256x512` RGB canvas:
  - **Top half (`256x256`)**: Skin UV RGB upscaled via Box filtering.
  - **Bottom half (`256x256`)**: Skin UV Alpha upscaled via Box filtering and represented as grayscale.
  - Gradients flow back smoothly through both active regions during training with no empty padding.
* **🖼️ Multi-View Control Image Loader**: Supports loading conditioning input as separate front and back views (each $256 \times 512$, e.g., from `front/` and `back/` directories). It automatically resizes and concatenates them side-by-side to construct a standard $512 \times 512$ conditioning input, with fallback to pre-combined files.
* **🎮 Voxel Texture Edge Consistency Resolver**: Reconstructs a temporary 3D voxel color grid during loading to resolve missing/transparent pixel conflicts at adjacent edges.
* **🔄 Slim-to-Standard Arm Expansion (Alex-to-Steve)**: Dynamically checks and converts Alex skins (3px arm width) into Steve skins (4px arm width) before training.
* **💾 Extreme VRAM Optimization**: Pre-encodes all text prompts into memory and completely unloads the Text Encoder(s) before the training loop starts. This saves massive amounts of VRAM (up to 8GB for models like Qwen-4B), freeing up space for larger batch sizes or differentiable rendering overhead.
* **👁️ Perceptual LPIPS Loss**: Supports optional LPIPS rendering loss to retain sharp pixel textures. Automatically enables when `--lambda_lpips` > 0.
* **🤖 Dual Architecture Compatibility**:
  - **Standard Flux**: Supports vanilla Hugging Face `diffusers` pipelines (T5 + CLIP text encoders, `FluxTransformer2DModel`).
  - **Flux2Klein (Custom)**: Supports custom sequence packing (`batched_prc_img`, `batched_prc_txt`), output scattering (`scatter_ids`), Qwen-based text encoders (hidden layer stacking of `[9, 18, 27]`), and custom VAE structures (small decoders).

---

## 📂 File Directory

```bash
SkingToolkit/
├── README.md              # Technical documentation & guide
├── flux2_src/             # Localized custom Flux2/Klein model package
│   ├── __init__.py
│   ├── model.py
│   ├── autoencoder.py
│   └── sampling.py
├── renderer.py            # Differentiable Renderer using PyTorch grid mapping
├── loss.py                # Multi-view MSE & LPIPS foreground-weighted losses
├── dataset.py             # MinecraftSkinDataset, Alex-Steve, & Voxel resolver
├── train.py               # Core training execution loop & backprop pipeline
├── test_toolkit_setup.py  # Local self-contained math/setup verification script
└── run_training.sh        # Shell launcher configured for Flux2Klein4B
```

---

## 🛠️ Setup & Verification

Before running training, verify the installation, coordinate mappings, and gradient backpropagation math on your machine.

### 1. Requirements
Install necessary dependencies:
```bash
pip install torch torchvision diffusers transformers accelerate peft einops tqdm pillow numpy
```
*Note: To run perceptual losses, optionally install `lpips` (`pip install lpips`).*

### 2. Verify Setup
Run the self-contained setup test script to mathematically prove that gradients flow correctly through the VAE and Renderer back to the model:
```bash
python SkingToolkit/test_toolkit_setup.py
```
This script will mock a small dataset batch, compile views, run a 10-step mock backpropagation fitting, and display the gradient norm.

---

## 🏋️ How to Train

Use [run_training.sh](run_training.sh) to quickly configure parameters and launch training:
```bash
bash SkingToolkit/run_training.sh
```

### Script Configuration Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model_path` | string | *Required* | Path containing model weights/safetensors folder. |
| `--model_type` | string | `standard` | Set to `flux2klein` for custom Qwen-based models, otherwise `standard`. |
| `--text_encoder_type` | string | `t5_clip` | Text encoder structure. Choices: `t5_clip`, `qwen`. |
| `--text_encoder_path` | string | `None` | Path to the Qwen model if `qwen` type is selected (defaults to `Qwen/Qwen3-4B`). |
| `--data_dir` | string | *Required* | Folder containing target flat `64x64` skin PNGs. |
| `--photos_dir` | string | `None` | Folder containing conditioning control photos. Looks for separate `front/{id}.png` and `back/{id}.png` (each 256x512) to combine side-by-side, otherwise falls back to `{id}.png` (512x512). |
| `--mappings_dir` | string | `None` | Folder containing the `.pt` view mapping coordinates. |
| `--output_dir` | string | `output` | Folder to save fine-tuned LoRA weights. |
| `--use_lora` | bool | `True` | Wrap transformer with PEFT LoRA fine-tuning. |
| `--lora_target_modules` | string | `None` | Comma-separated target modules (e.g. `qkv,linear1,linear2,proj` for custom models). |
| `--lr` | float | `1e-4` | Learning rate. |
| `--batch_size` | int | `1` | Training batch size. |
| `--mixed_precision` | string | `bf16` | Precision mode (`bf16`, `fp16`, or `no`). |
| `--lambda_latent` | float | `1.0` | Coefficient weight of Flow Matching latent loss. |
| `--lambda_uv` | float | `1.0` | Coefficient weight of flat skin UV loss ($L_{uv}$). **Recommended: 10.0 - 20.0** |
| `--lambda_render` | float | `1.0` | Coefficient weight of rendering loss ($L_{render}$). **Recommended: 20.0 - 50.0** |
| `--lambda_lpips` | float | `0.0` | Coefficient weight of LPIPS rendering loss. Automatically activates LPIPS if `> 0`. **Recommended: 0.1 - 1.0** |
| `--views` | string | `static_front,static_back` | Views to include in the render loss. |
| `--foreground_weight` | float | `1.0` | Focus multiplier weight on foreground character pixels. |

---

## 💡 Training Tips & Loss Scaling

Because **Latent MSE** is calculated on flow-matching noise velocities (which have a large variance and magnitude, e.g., ~0.1 - 0.5), while **UV/Render MSE** are calculated on normalized pixel arrays in `[0, 1]` (resulting in extremely small absolute squared errors like ~0.001), you must scale up the auxiliary losses significantly to make them affect the gradients.

* **Scale Up Lambdas:** We highly recommend setting `LAMBDA_UV=20.0` and `LAMBDA_RENDER=50.0` in your `run_training.sh` script to force the optimizer to respect the 3D structure.
* **Progress Bar Display:** The real-time progress bar logs display the **raw, unscaled** MSE values. Therefore, seeing `UV MSE=0.0008` during training is completely normal and healthy; the scaling multiplier is applied automatically in the backend gradients.
* **LPIPS for Texture:** Standard MSE loss often produces blurry or overly smooth textures. Setting `LAMBDA_LPIPS=0.5` significantly sharpens the pixel art grain and fabric folds.

---

## 🧬 Differentiable rendering workflow

```mermaid
graph TD
    subgraph Input Data
        Photo[Conditioning Photo]
        Prompt[Text Prompt]
        GTSkin[Ground Truth 64x64 RGBA Skin]
        Composite[Target 256x512 RGB+Alpha Image]
    end

    subgraph Differentiable Forward Pass
        VAEEnc[VAE Encoder] -->|Encode| LatentGT[x_0 Latent]
        LatentGT -->|Flow Matching Noising| LatentXT[x_t Latent]
        
        Photo & Prompt & LatentXT -->|Flux Transformer| VPred[Predicted Velocity Vector]
        
        VPred & LatentXT -->|Flow Math| PredX0[Predicted x_0 Latent]
        PredX0 -->|VAE Decoder| Decoded[256x512 Decoded RGB+Alpha]
        
        Decoded -->|Slice & Interpolate| PredSkin[Predicted 64x64 RGBA Skin]
        
        PredSkin -->|Differentiable Renderer| PredRender[Predicted Character Views]
        GTSkin -->|Differentiable Renderer| GTRender[GT Character Views]
    end

    subgraph Loss Formulation
        PredSkin & GTSkin -->|UV MSE| Luv[UV Loss]
        PredRender & GTRender -->|View MSE + LPIPS| Lrender[Render Loss]
        VPred -->|Velocity MSE| Llatent[Latent Loss]
        
        Luv & Lrender & Llatent -->|Weighted Sum| L[Total Loss]
    end

    L -->|Backpropagation| Gradients[Gradients flow back to Flux LoRA/Weights]
```
