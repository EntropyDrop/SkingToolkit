# `foreground_alpha_simple`: Simple Magic Wand Flood-Fill Background Removal

`foreground_alpha_simple` provides a fast, lightweight background removal algorithm based on **Flood Fill (Magic Wand)**. It samples the background color at pixel coordinate `(0, 0)` and removes all contiguous, similar background colors to produce a clean 4-channel RGBA image.

Unlike deep learning segmentation approaches, this simple algorithm requires **no neural network checkpoints**, runs instantly on CPU, and is ideal for 2D/3D character renders with solid or near-solid backgrounds.

---

## 🚀 Quick Start

### 1. Command Line Interface (CLI)

Run background removal on a single image or multiple images:

```bash
# Simple background removal using (0,0) seed color
python SkingToolkit/foreground_alpha_simple/infer.py \
    --input input_render.png \
    --output_dir output_folder \
    --tolerance 15

# Batch processing
python SkingToolkit/foreground_alpha_simple/infer.py \
    --inputs image1.png image2.png image3.png \
    --output_dir output_folder \
    --tolerance 20
```

Or use the provided launcher script:

```bash
bash SkingToolkit/foreground_alpha_simple/run_foreground_alpha_simple_infer.sh /path/to/render.png output_dir 15
```

---

## 💡 Python API Usage

You can use `foreground_alpha_simple` directly in your Python code:

```python
from PIL import Image
from SkingToolkit.foreground_alpha_simple import remove_background_simple, flood_fill_alpha_simple

# Load image
img = Image.open("character_render.png")

# Remove background (seed at 0,0 by default)
rgba_img = remove_background_simple(
    image=img,
    seed=(0, 0),        # Seed point coordinate (x, y)
    tolerance=15,       # Color matching tolerance (0-255)
    color_space="RGB",  # "RGB" or "LAB"
    uncompose=True      # Remove background color bleeding on edges
)

# Save result
rgba_img.save("character_no_bg.png")
```

---

## ⚙️ CLI Options & Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input` | string | `None` | Path to a single input RGB/RGBA image file. |
| `--inputs` | strings | `None` | Space-separated list of multiple input image files. |
| `--output` | string | `None` | Explicit output RGBA file path (for single input). |
| `--output_dir` | string | `foreground_alpha_simple_outputs` | Output folder for batch inputs. |
| `--seed` | string | `0,0` | Seed pixel coordinate `x,y` to extract initial background color. |
| `--tolerance` | float | `15.0` | Magic wand color difference tolerance (0-255). |
| `--color_space` | string | `RGB` | Color space for difference calculation (`RGB` or `LAB`). |
| `--floating_range` | flag | `False` | Compare pixel color to adjacent pixel (floating) instead of initial seed color. |
| `--uncompose` | flag | `False` | Recover uncomposed foreground RGB to remove edge color bleeding. |
| `--fill_holes` | flag | `False` | Fill enclosed transparent holes inside the foreground character body. |
| `--opening_size` | int | `0` | Kernel size for morphological opening to remove thin edge noise. |
| `--split_merged` | string | `auto` | Auto-split merged 2:1 front+back side-by-side renders (`auto`, `always`, `never`). |
| `--save_split` | flag | `False` | Save separate `_front_rgba.png` and `_back_rgba.png` images when processing merged renders. |
