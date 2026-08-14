from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from image_postprocess import minecraft_crisp_postprocess


class MinecraftCrispPostprocessTest(unittest.TestCase):
    def test_preserves_exact_white_canvas_after_posterize(self) -> None:
        pixels = np.full((8, 8, 3), 255, dtype=np.uint8)
        pixels[2:6, 2:6] = (93, 142, 211)
        image = minecraft_crisp_postprocess(Image.fromarray(pixels), posterize_bits=5)
        result = np.asarray(image)
        self.assertTrue(np.all(result[0] == 255))
        self.assertTrue(np.all(result[:, 0] == 255))

    def test_disabled_mode_only_snaps_near_white(self) -> None:
        pixels = np.array([[[252, 253, 254], [20, 40, 60]]], dtype=np.uint8)
        image = minecraft_crisp_postprocess(
            Image.fromarray(pixels),
            enabled=False,
            white_threshold=250,
        )
        result = np.asarray(image)
        np.testing.assert_array_equal(result[0, 0], [255, 255, 255])
        np.testing.assert_array_equal(result[0, 1], [20, 40, 60])

    def test_rejects_invalid_posterize_bits(self) -> None:
        with self.assertRaises(ValueError):
            minecraft_crisp_postprocess(Image.new("RGB", (2, 2)), posterize_bits=0)

    def test_rejects_negative_contrast_or_saturation(self) -> None:
        with self.assertRaises(ValueError):
            minecraft_crisp_postprocess(Image.new("RGB", (2, 2)), contrast=-0.1)
        with self.assertRaises(ValueError):
            minecraft_crisp_postprocess(Image.new("RGB", (2, 2)), saturation=-0.1)


if __name__ == "__main__":
    unittest.main()
