import unittest

import torch
import torch.nn as nn

from SkingToolkit.dense_uv_parser.losses import DenseUVParserLoss
from SkingToolkit.dense_uv_parser.model import DenseUVParserNet
from SkingToolkit.dense_uv_parser.utils import (
    augment_dense_batch,
    canonicalize_parser_render,
    canonicalize_tensor,
    splat_deterministic_targets_to_uv_conditioning,
    splat_parser_predictions_to_uv_conditioning,
)


class FakeRenderer(nn.Module):
    def __init__(self, height=8, width=8, valid_pixels=1):
        super().__init__()
        grid = torch.zeros(height, width, 2)
        grid[..., 0] = -1.0
        grid[..., 1] = -1.0
        mask = torch.zeros(height, width)
        mask.view(-1)[:valid_pixels] = 1.0
        self.register_buffer("front_inner_grid", grid)
        self.register_buffer("front_outer_grid", grid)
        self.register_buffer("front_inner_mask", mask)
        self.register_buffer("front_outer_mask", torch.zeros_like(mask))


def dense_targets(batch, height, width):
    return {
        "foreground": torch.ones(batch, 1, height, width),
        "layer": torch.zeros(batch, height, width, dtype=torch.long),
        "part": torch.zeros(batch, height, width, dtype=torch.long),
        "face": torch.zeros(batch, height, width, dtype=torch.long),
        "surface": torch.zeros(batch, height, width, dtype=torch.long),
        "uv": torch.zeros(batch, 2, height, width),
    }


class GlobalAffineRoutingTest(unittest.TestCase):
    def test_global_model_emits_surface_and_affine_losses(self):
        rendered = torch.rand(1, 4, 32, 32)
        rendered[:, 3] = 1.0
        _, targets = augment_dense_batch(
            rendered,
            dense_targets(1, 32, 32),
            translation_scale=0.0,
            scale_range=0.0,
        )
        model = DenseUVParserNet(
            base_channels=8,
            uv_classification=False,
            view_classes=1,
            predict_affine=True,
            surface_classes=4,
        )
        outputs = model(rendered, view_ids=torch.zeros(1, dtype=torch.long))
        self.assertEqual(tuple(outputs["surface"].shape), (1, 4, 32, 32))
        self.assertEqual(tuple(outputs["affine"].shape), (1, 3))
        losses = DenseUVParserLoss(use_uv=False)(outputs, targets)
        self.assertTrue(torch.isfinite(losses["loss_total"]))

    def test_affine_target_undoes_augmentation(self):
        height = width = 64
        rendered = torch.zeros(1, 4, height, width)
        rendered[:, 3] = 1.0
        rendered[:, 0, 20:44, 20:44] = 1.0

        augmented, targets = augment_dense_batch(
            rendered,
            dense_targets(1, height, width),
            translation_scale=0.03,
            scale_range=0.03,
            generator=torch.Generator().manual_seed(7),
        )
        recovered = canonicalize_tensor(augmented, targets["affine"])
        self.assertLess((recovered[:, :3] - rendered[:, :3]).abs().mean().item(), 0.03)

    def test_color_canonicalization_uses_nearest_texels(self):
        height = width = 64
        rendered = torch.zeros(1, 4, height, width)
        rendered[:, 3] = 1.0
        rendered[:, :3, :, ::2] = 1.0
        augmented, targets = augment_dense_batch(
            rendered,
            dense_targets(1, height, width),
            translation_scale=0.03,
            scale_range=0.03,
            generator=torch.Generator().manual_seed(7),
        )
        sharp = canonicalize_parser_render(augmented, {"affine": targets["affine"]})[:, :3]
        smooth = canonicalize_tensor(augmented, targets["affine"], mode="bilinear")[:, :3]
        self.assertTrue(torch.allclose(sharp, canonicalize_tensor(augmented, targets["affine"], mode="nearest")[:, :3]))
        interior = (slice(None), slice(None), slice(4, -4), slice(4, -4))
        sharp_fractional = (sharp[interior] - sharp[interior].round()).abs().mean()
        smooth_fractional = (smooth[interior] - smooth[interior].round()).abs().mean()
        self.assertLess(sharp_fractional, smooth_fractional)

    def test_mapping_mask_rejects_background_false_positives(self):
        renderer = FakeRenderer(valid_pixels=1)
        rendered = torch.rand(1, 4, 8, 8)
        rendered[:, 3] = 1.0
        outputs = {
            "foreground": torch.full((1, 1, 8, 8), 10.0),
            "layer": torch.cat(
                [torch.full((1, 1, 8, 8), 10.0), torch.full((1, 1, 8, 8), -10.0)],
                dim=1,
            ),
            "part": torch.zeros(1, 6, 8, 8),
            "face": torch.zeros(1, 6, 8, 8),
            "surface": torch.cat(
                [torch.full((1, 1, 8, 8), 10.0), torch.full((1, 1, 8, 8), -10.0)],
                dim=1,
            ),
            "affine": torch.zeros(1, 3),
        }

        conditioning, details = splat_parser_predictions_to_uv_conditioning(
            rendered,
            outputs,
            renderer=renderer,
            views=["front"],
            group_size=1,
            semantic_gate=False,
            return_details=True,
        )
        self.assertEqual(int(details["routing"]["foreground"].sum()), 1)
        self.assertEqual(int(conditioning[:, 4:5].sum()), 1)

        _, targets = augment_dense_batch(
            rendered,
            dense_targets(1, 8, 8),
            translation_scale=0.0,
            scale_range=0.0,
        )
        target_conditioning = splat_deterministic_targets_to_uv_conditioning(
            rendered,
            targets,
            renderer=renderer,
            views=["front"],
            group_size=1,
        )
        self.assertEqual(int(target_conditioning[:, 4:5].sum()), 1)


if __name__ == "__main__":
    unittest.main()
