#!/usr/bin/env python3
"""Convert a Krea two-view draft into a valid Minecraft skin and fixed preview."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


TOOLKIT_DIR = Path(__file__).resolve().parents[1]
TOOLKIT_PARENT = TOOLKIT_DIR.parent
if str(TOOLKIT_PARENT) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_PARENT))

from SkingToolkit.dense_uv_parser.uv_layout import build_uv_masks  # noqa: E402
from SkingToolkit.renderer import DifferentiableRenderer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a legal 64x64 Minecraft UV atlas from a front_left/back_left "
            "Krea draft, validate it, and render a deterministic white-background preview."
        )
    )
    parser.add_argument("--draft", required=True)
    parser.add_argument("--skin", required=True)
    parser.add_argument("--preview", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--parser-checkpoint", required=True)
    parser.add_argument("--mappings-dir", required=True)
    parser.add_argument("--content-scale", type=float, default=0.875)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def run_dense_uv_parser(
    *,
    draft: Path,
    skin: Path,
    parser_checkpoint: Path,
    mappings_dir: Path,
    device: str,
) -> None:
    parser_script = TOOLKIT_DIR / "dense_uv_parser" / "infer.py"
    if not parser_script.is_file():
        raise FileNotFoundError(parser_script)
    for required in (draft, parser_checkpoint, mappings_dir):
        if not required.exists():
            raise FileNotFoundError(required)
    skin.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="krea-uv-") as temporary_dir:
        temporary_render = Path(temporary_dir) / "parser-render.png"
        command = [
            sys.executable,
            "-u",
            str(parser_script),
            "--parser_checkpoint",
            str(parser_checkpoint),
            "--combined",
            str(draft),
            "--mappings_dir",
            str(mappings_dir),
            "--output",
            str(skin),
            "--simple_inpaint_render_output",
            str(temporary_render),
            "--conditioning_output",
            "",
            "--parser_uv_output",
            "",
            "--simple_inpaint_output",
            "",
            "--foreground_probability_output",
            "",
            "--foreground_mask_output",
            "",
            "--foreground_raw_mask_output",
            "",
            "--foreground_cutout_output",
            "",
            "--foreground_parser_input_output",
            "",
            "--device",
            device,
        ]
        environment = os.environ.copy()
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        subprocess.run(
            command,
            cwd=TOOLKIT_DIR / "dense_uv_parser",
            env=environment,
            check=True,
        )
    if not skin.is_file():
        raise RuntimeError("Dense UV parser finished without a 64x64 skin")


def validate_skin(skin: Path) -> tuple[torch.Tensor, dict[str, int | list[int]]]:
    with Image.open(skin) as opened:
        if opened.size != (64, 64):
            raise ValueError(f"Minecraft skin must be 64x64, found {opened.size}")
        rgba = np.asarray(opened.convert("RGBA"), dtype=np.uint8).copy()
    tensor = torch.from_numpy(rgba).permute(2, 0, 1).float().div_(255.0)
    base_mask, outer_mask = build_uv_masks(is_slim=False)
    valid_mask = (base_mask + outer_mask).clamp_max(1).bool()[0]
    base_mask = base_mask.bool()[0]
    outer_mask = outer_mask.bool()[0]
    alpha = rgba[:, :, 3]
    opaque = alpha == 255
    missing_base = int(np.count_nonzero(~opaque & base_mask.numpy()))
    invalid_alpha = int(np.count_nonzero((alpha != 0) & ~valid_mask.numpy()))
    partial_alpha = int(np.count_nonzero((alpha != 0) & (alpha != 255)))
    if missing_base or invalid_alpha or partial_alpha:
        raise ValueError(
            "Invalid Minecraft atlas alpha: "
            f"missing_base={missing_base}, invalid_region={invalid_alpha}, partial={partial_alpha}"
        )
    base_colors = rgba[:, :, :3][base_mask.numpy()]
    unique_base_colors = int(np.unique(base_colors, axis=0).shape[0])
    if unique_base_colors < 2:
        raise ValueError("Parsed skin is blank or monochrome")
    validation: dict[str, int | list[int]] = {
        "size": [64, 64],
        "missing_base_alpha_pixels": missing_base,
        "invalid_region_alpha_pixels": invalid_alpha,
        "partial_alpha_pixels": partial_alpha,
        "opaque_base_pixels": int(np.count_nonzero(opaque & base_mask.numpy())),
        "opaque_outer_pixels": int(np.count_nonzero(opaque & outer_mask.numpy())),
        "unique_base_colors": unique_base_colors,
    }
    return tensor, validation


def scale_view(view: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    if tuple(view.shape[-2:]) == (height, width):
        return view
    return F.interpolate(view, size=(height, width), mode="nearest-exact")


def render_fixed_preview(
    skin_tensor: torch.Tensor,
    *,
    mappings_dir: Path,
    output: Path,
    content_scale: float,
) -> dict[str, int | float | list[int] | str]:
    if not 0.0 < content_scale <= 1.0:
        raise ValueError("content_scale must be in (0, 1]")
    renderer = DifferentiableRenderer(
        str(mappings_dir),
        bg_color=(1.0, 1.0, 1.0),
        sampling_mode="nearest",
    ).eval()
    expected_views = ("front_left", "back_left")
    missing = [view for view in expected_views if view not in renderer.views]
    if missing:
        raise ValueError(f"Renderer mappings are missing required views: {missing}")
    with torch.inference_mode():
        batch = skin_tensor.unsqueeze(0)
        front = renderer.forward_view(batch, expected_views[0])[:, :3]
        back = renderer.forward_view(batch, expected_views[1])[:, :3]
        canvas_height = 512
        canvas_width = 512
        half_width = canvas_width // 2
        scaled_height = round(canvas_height * content_scale)
        scaled_width = round(half_width * content_scale)
        front = scale_view(front, height=scaled_height, width=scaled_width)
        back = scale_view(back, height=scaled_height, width=scaled_width)
        preview = torch.ones(1, 3, canvas_height, canvas_width, dtype=front.dtype)
        y = (canvas_height - scaled_height) // 2
        left_x = (half_width - scaled_width) // 2
        right_x = half_width + left_x
        preview[:, :, y : y + scaled_height, left_x : left_x + scaled_width] = front
        preview[:, :, y : y + scaled_height, right_x : right_x + scaled_width] = back
    array = (
        preview[0]
        .clamp(0.0, 1.0)
        .mul(255)
        .round()
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(output, optimize=True)
    non_white = np.any(array != 255, axis=2)
    if not np.any(non_white):
        raise ValueError("Rendered preview is blank")
    return {
        "size": [canvas_width, canvas_height],
        "views": list(expected_views),
        "content_scale": content_scale,
        "sampling": "nearest",
        "background_rgb": [255, 255, 255],
        "non_white_pixels": int(np.count_nonzero(non_white)),
    }


def main() -> None:
    args = parse_args()
    draft = Path(args.draft).expanduser().resolve()
    skin = Path(args.skin).expanduser().resolve()
    preview = Path(args.preview).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    parser_checkpoint = Path(args.parser_checkpoint).expanduser().resolve()
    mappings_dir = Path(args.mappings_dir).expanduser().resolve()

    print("UV phase: extracting strict 64x64 Minecraft atlas", flush=True)
    run_dense_uv_parser(
        draft=draft,
        skin=skin,
        parser_checkpoint=parser_checkpoint,
        mappings_dir=mappings_dir,
        device=args.device,
    )
    skin_tensor, skin_validation = validate_skin(skin)
    print("UV phase: rendering fixed front_left/back_left preview", flush=True)
    preview_validation = render_fixed_preview(
        skin_tensor,
        mappings_dir=mappings_dir,
        output=preview,
        content_scale=args.content_scale,
    )
    report = {
        "draft": str(draft),
        "skin": str(skin),
        "preview": str(preview),
        "parser_checkpoint": str(parser_checkpoint),
        "mappings_dir": str(mappings_dir),
        "skin_validation": skin_validation,
        "preview_validation": preview_validation,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("UV validation: " + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
