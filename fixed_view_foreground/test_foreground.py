import tempfile
import unittest
from pathlib import Path

import torch

from SkingToolkit.fixed_view_foreground.augmentation import (
    composite_random_background,
)
from SkingToolkit.fixed_view_foreground.inference import find_latest_checkpoint
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


if __name__ == "__main__":
    unittest.main()
