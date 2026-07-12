import unittest

import torch
import torch.nn as nn

from SkingToolkit.dense_uv_parser.losses import DenseUVParserLoss, _balanced_cross_entropy
from SkingToolkit.dense_uv_parser.model import DenseUVParserNet
from SkingToolkit.dense_uv_parser.utils import (
    augment_dense_batch,
    canonicalize_parser_render,
    canonicalize_tensor,
    refine_parser_affine,
    splat_deterministic_targets_to_uv_conditioning,
    splat_parser_predictions_to_uv_conditioning,
    splat_to_uv_conditioning,
)


class FakeRenderer(nn.Module):
    def __init__(self, height=8, width=8, valid_pixels=1, mask=None):
        super().__init__()
        if mask is not None:
            height, width = mask.shape
        grid = torch.zeros(height, width, 2)
        grid[..., 0] = -1.0
        grid[..., 1] = -1.0
        if mask is None:
            mask = torch.zeros(height, width)
            mask.view(-1)[:valid_pixels] = 1.0
        else:
            mask = mask.float().clone()
        self.register_buffer("front_inner_grid", grid.clone())
        self.register_buffer("front_outer_grid", grid.clone())
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
    def test_balanced_cross_entropy_supports_bfloat16_logits(self):
        logits = torch.randn(2, 12, 8, 8, dtype=torch.bfloat16)
        target = torch.randint(0, 12, (2, 8, 8))
        loss = _balanced_cross_entropy(logits, target)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.dtype, torch.float32)

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
        self.assertEqual(tuple(outputs["layer_face"].shape), (1, 12, 32, 32))
        self.assertEqual(tuple(outputs["affine"].shape), (1, 3))
        losses = DenseUVParserLoss(use_uv=False)(outputs, targets)
        self.assertTrue(torch.isfinite(losses["loss_total"]))
        self.assertTrue(torch.isfinite(losses["loss_layer_face"]))

    def test_global_model_can_train_discrete_uv_routing_auxiliary(self):
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
            uv_classification=True,
            view_classes=1,
            predict_affine=True,
            surface_classes=4,
        )

        outputs = model(rendered, view_ids=torch.zeros(1, dtype=torch.long))
        losses = DenseUVParserLoss(use_uv=True)(outputs, targets)

        self.assertEqual(tuple(outputs["uv_x"].shape), (1, 64, 32, 32))
        self.assertEqual(tuple(outputs["uv_y"].shape), (1, 64, 32, 32))
        self.assertTrue(torch.isfinite(losses["loss_routing"]))

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

    def test_affine_refinement_snaps_a_one_pixel_translation(self):
        height = width = 32
        target = torch.zeros(height, width)
        target[7:25, 9:23] = 1.0
        shifted = torch.zeros_like(target)
        shifted[:, 1:] = target[:, :-1]
        renderer = FakeRenderer(mask=target)
        outputs = {
            "foreground": torch.where(
                shifted.view(1, 1, height, width) > 0.5,
                torch.tensor(10.0),
                torch.tensor(-10.0),
            ),
            "affine": torch.zeros(1, 3),
        }

        refined, details = refine_parser_affine(
            outputs,
            renderer,
            ["front"],
            translation_radius_px=2.0,
            scale_radius=0.0,
        )

        self.assertTrue(details["accepted"][0])
        self.assertAlmostEqual(float(details["translation_px"][0, 0]), 1.0, places=4)
        self.assertAlmostEqual(float(refined[0, 0]), 2.0 / width, places=4)

    def test_affine_refinement_preserves_exact_alignment(self):
        height = width = 32
        target = torch.zeros(height, width)
        target[7:25, 9:23] = 1.0
        renderer = FakeRenderer(mask=target)
        outputs = {
            "foreground": torch.where(
                target.view(1, 1, height, width) > 0.5,
                torch.tensor(10.0),
                torch.tensor(-10.0),
            ),
            "affine": torch.zeros(1, 3),
        }

        refined, details = refine_parser_affine(outputs, renderer, ["front"])

        self.assertFalse(details["accepted"][0])
        self.assertTrue(torch.equal(refined, outputs["affine"]))

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

    def test_base_silhouette_recovers_foreground_and_semantic_gate_holes(self):
        renderer = FakeRenderer(valid_pixels=1)
        rendered = torch.rand(1, 4, 8, 8)
        rendered[:, 3] = 1.0
        outputs = {
            "foreground": torch.full((1, 1, 8, 8), -10.0),
            "layer": torch.cat(
                [torch.full((1, 1, 8, 8), -10.0), torch.full((1, 1, 8, 8), 10.0)],
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
            semantic_gate=True,
            affine_refine=False,
            return_details=True,
        )

        self.assertEqual(int(details["routing"]["foreground"].sum()), 1)
        self.assertTrue(details["routing"]["semantic_fallback"][0, 0, 0])
        self.assertEqual(int(conditioning[:, 4:5].sum()), 1)

    def test_routing_reranks_surface_classes_that_are_invalid_at_pixel(self):
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
            # Surface 1 wins the global argmax but has no valid mapping pixels.
            "surface": torch.cat(
                [torch.full((1, 1, 8, 8), 5.0), torch.full((1, 1, 8, 8), 10.0)],
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
            affine_refine=False,
            return_details=True,
        )

        self.assertEqual(int(details["routing"]["foreground"].sum()), 1)
        self.assertEqual(int(details["routing"]["surface"][0, 0, 0]), 0)
        self.assertEqual(int(conditioning[:, 4:5].sum()), 1)

    def test_confidence_splat_uses_one_source_pixel_for_ties(self):
        rendered = torch.zeros(2, 4, 1, 1)
        rendered[0, 0, 0, 0] = 1.0
        rendered[1, 2, 0, 0] = 1.0
        rendered[:, 3, 0, 0] = 1.0
        fg = torch.ones(2, 1, 1, dtype=torch.bool)
        layer = torch.zeros(2, 1, 1, dtype=torch.long)
        flat_uv = torch.zeros(2, 1, 1, dtype=torch.long)
        confidence = torch.ones(2, 1, 1)

        conditioning = splat_to_uv_conditioning(
            rendered,
            fg,
            layer,
            flat_uv,
            group_size=2,
            confidence=confidence,
        )

        self.assertTrue(torch.equal(conditioning[0, :3, 0, 0], torch.tensor([1.0, 0.0, 0.0])))

    def test_uv_classification_reranks_ambiguous_surface_candidates(self):
        renderer = FakeRenderer(valid_pixels=1)
        renderer.front_outer_mask.copy_(renderer.front_inner_mask)
        renderer.front_outer_grid[0, 0, 0] = (1.0 / 63.0) * 2.0 - 1.0
        rendered = torch.rand(1, 4, 8, 8)
        rendered[:, 3] = 1.0
        uv_x = torch.full((1, 64, 8, 8), -10.0)
        uv_y = torch.full((1, 64, 8, 8), -10.0)
        uv_x[:, 0] = 10.0
        uv_y[:, 0] = 10.0
        outputs = {
            "foreground": torch.full((1, 1, 8, 8), 10.0),
            "layer": torch.zeros(1, 2, 8, 8),
            "part": torch.zeros(1, 6, 8, 8),
            "face": torch.zeros(1, 6, 8, 8),
            "surface": torch.cat(
                [torch.zeros(1, 1, 8, 8), torch.full((1, 1, 8, 8), 2.0)],
                dim=1,
            ),
            "uv_x": uv_x,
            "uv_y": uv_y,
            "affine": torch.zeros(1, 3),
        }

        _, details = splat_parser_predictions_to_uv_conditioning(
            rendered,
            outputs,
            renderer=renderer,
            views=["front"],
            group_size=1,
            semantic_gate=False,
            affine_refine=False,
            return_details=True,
        )

        self.assertEqual(int(details["routing"]["surface"][0, 0, 0]), 0)

    def test_joint_layer_face_reranks_inner_outer_top_surfaces(self):
        renderer = FakeRenderer(valid_pixels=1)
        renderer.front_outer_mask.copy_(renderer.front_inner_mask)
        renderer.front_inner_grid[0, 0, 0] = (8.0 / 63.0) * 2.0 - 1.0
        renderer.front_outer_grid[0, 0, 0] = (40.0 / 63.0) * 2.0 - 1.0
        rendered = torch.rand(1, 4, 8, 8)
        rendered[:, 3] = 1.0
        layer_face = torch.full((1, 12, 8, 8), -10.0)
        layer_face[:, 4] = -2.0
        layer_face[:, 10] = 10.0
        outputs = {
            "foreground": torch.full((1, 1, 8, 8), 10.0),
            "layer": torch.zeros(1, 2, 8, 8),
            "part": torch.zeros(1, 6, 8, 8),
            "face": torch.zeros(1, 6, 8, 8),
            "layer_face": layer_face,
            "surface": torch.zeros(1, 2, 8, 8),
            "affine": torch.zeros(1, 3),
        }

        _, details = splat_parser_predictions_to_uv_conditioning(
            rendered,
            outputs,
            renderer=renderer,
            views=["front"],
            group_size=1,
            semantic_gate=False,
            affine_refine=False,
            return_details=True,
        )

        self.assertEqual(int(details["routing"]["surface"][0, 0, 0]), 1)

        outputs["layer_face"].zero_()
        _, filtered_details = splat_parser_predictions_to_uv_conditioning(
            rendered,
            outputs,
            renderer=renderer,
            views=["front"],
            group_size=1,
            semantic_gate=False,
            affine_refine=False,
            route_margin_threshold=0.10,
            return_details=True,
        )
        self.assertEqual(int(filtered_details["routing"]["raw_foreground"].sum()), 1)
        self.assertEqual(int(filtered_details["routing"]["foreground"].sum()), 0)
        self.assertEqual(int(filtered_details["routing"]["rejected"].sum()), 1)


if __name__ == "__main__":
    unittest.main()
