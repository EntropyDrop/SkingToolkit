import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from SkingToolkit.fixed_view_foreground.augmentation import (
    composite_random_background,
)
from SkingToolkit.fixed_view_foreground.inference import (
    build_parser_input,
    find_latest_checkpoint,
    save_foreground_outputs,
)
from SkingToolkit.fixed_view_foreground.model import FixedViewForegroundNet


class FixedViewForegroundTest(unittest.TestCase):
    def test_model_preserves_spatial_shape_and_uses_view_ids(self):
        model = FixedViewForegroundNet(base_channels=8, view_classes=2)
        images = torch.rand(2, 3, 32, 48)
        logits = model(images, torch.tensor([0, 1]))
        self.assertEqual(tuple(logits.shape), (2, 1, 32, 48))

    def test_random_background_composition_preserves_opaque_pixels(self):
        rendered = torch.zeros(2, 4, 16, 16)
        rendered[:, :3] = 0.5
        rendered[:, 3:4, 4:12, 4:12] = 1.0
        rendered[:, :3, 4:12, 4:12] = torch.tensor(
            [0.9, 0.7, 0.5]
        ).view(1, 3, 1, 1)
        target = torch.zeros(2, 1, 16, 16)
        target[:, :, 4:12, 4:12] = 1.0
        composited, background = composite_random_background(
            rendered, target, (128, 128, 128)
        )
        expected = rendered[:, :3, 4:12, 4:12]
        actual = composited[:, :, 4:12, 4:12]
        # Training noise is intentionally small and must not destroy the source color.
        self.assertLess(float((expected - actual).abs().amax()), 0.04)
        self.assertEqual(tuple(background.shape), (2, 3, 16, 16))

    def test_latest_checkpoint_uses_numeric_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for version in (2, 10, 3):
                run = root / f"fixed_view_foreground_v{version}"
                run.mkdir()
                (run / "best.pt").touch()
            latest = find_latest_checkpoint(root)
            self.assertEqual(latest.parent.name, "fixed_view_foreground_v10")

    def test_cutout_is_transparent_but_parser_input_is_opaque_gray(self):
        rendered = torch.rand(2, 4, 8, 8)
        rendered[:, 3:4] = 1.0
        probability = torch.zeros(2, 1, 8, 8)
        probability[:, :, 2:6, 2:6] = 1.0
        with tempfile.TemporaryDirectory() as temporary:
            cutout_path = Path(temporary) / "cutout.png"
            mask = save_foreground_outputs(
                rendered,
                probability,
                threshold=0.5,
                view_count=2,
                cutout_output=cutout_path,
            )
            cutout = Image.open(cutout_path).convert("RGBA")
            self.assertEqual(cutout.getpixel((2, 2))[3], 0)
            self.assertEqual(cutout.getpixel((5, 5))[3], 255)

        parser_input = build_parser_input(
            rendered,
            mask,
            bg_color=(128, 128, 128),
            background_mode="neutral",
        )
        self.assertTrue(torch.all(parser_input[:, 3:4] == 1.0))
        expected_bg = torch.tensor(128.0 / 255.0)
        self.assertTrue(
            torch.allclose(parser_input[0, :3, 0, 0], expected_bg.expand(3))
        )

    def test_adaptive_background_is_deterministic_and_contrasts_black_edges(self):
        rendered = torch.zeros(1, 4, 16, 16)
        rendered[:, 3:4] = 1.0
        mask = torch.zeros(1, 16, 16, dtype=torch.bool)
        mask[:, 4:12, 4:12] = True
        first, color, indices = build_parser_input(
            rendered,
            mask,
            background_mode="adaptive",
            return_background=True,
        )
        second, second_color, second_indices = build_parser_input(
            rendered,
            mask,
            background_mode="adaptive",
            return_background=True,
        )
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(color, second_color))
        self.assertEqual(indices, second_indices)
        self.assertTrue(torch.all(color[0] > 0.9))


if __name__ == "__main__":
    unittest.main()
