"""Generate augmentation deformation examples for visual inspection.

Applies RenderAugmenter variants to a render view and saves combined results.
"""
import os
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.utils import save_image

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TOOLKIT_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from SkingToolkit.semantic_uv_reconstruction.dataset import RenderAugmenter  # noqa: E402


def main():
    img_path = Path("../../test_imgs/walk_front_both_layer_ortho_pyvista.png")
    if not img_path.exists():
        img_path = Path("../test_imgs/walk_front_both_layer_ortho_pyvista.png")
    if not img_path.exists():
        print(f"Render view image not found at {img_path.resolve()}")
        sys.exit(1)

    img = Image.open(img_path).convert("RGBA")
    tensor = TF.to_tensor(img)  # (4, H, W)
    bg_color = img.getpixel((0, 0))[:3]

    # Augmenter variants: translate, scale, translate+scale, translate+scale+persp
    variants = [
        ("original", None),
        ("translate 0.03", RenderAugmenter(translation_scale=0.03, scale_range=0.0, perspective_scale=0.0, bg_color=bg_color)),
        ("translate 0.06", RenderAugmenter(translation_scale=0.06, scale_range=0.0, perspective_scale=0.0, bg_color=bg_color)),
        ("scale ±3%", RenderAugmenter(translation_scale=0.0, scale_range=0.03, perspective_scale=0.0, bg_color=bg_color)),
        ("scale ±6%", RenderAugmenter(translation_scale=0.0, scale_range=0.06, perspective_scale=0.0, bg_color=bg_color)),
        ("trans+scale", RenderAugmenter(translation_scale=0.03, scale_range=0.03, perspective_scale=0.0, bg_color=bg_color)),
        ("trans+scale 2", RenderAugmenter(translation_scale=0.04, scale_range=0.02, perspective_scale=0.0, bg_color=bg_color)),
        ("+perspective", RenderAugmenter(translation_scale=0.03, scale_range=0.03, perspective_scale=0.008, bg_color=bg_color)),
        ("+perspective 2", RenderAugmenter(translation_scale=0.02, scale_range=0.01, perspective_scale=0.02, bg_color=bg_color)),
    ]

    torch.manual_seed(42)  # reproducible examples
    results = []
    for label, augmenter in variants:
        if augmenter is None:
            results.append(tensor)
        else:
            results.append(augmenter(tensor.clone()))
        print(f"  {label}")

    # Add labels as a row of text? For now just tile them in a grid.
    grid = torch.stack(results, dim=0)  # (N, 4, H, W)

    out_dir = Path(__file__).resolve().parent / "examples"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "augmentation_examples.png"
    save_image(grid.clamp(0.0, 1.0), out_path, nrow=3)
    print(f"\nSaved {len(results)} augmented views to {out_path}")


if __name__ == "__main__":
    main()
