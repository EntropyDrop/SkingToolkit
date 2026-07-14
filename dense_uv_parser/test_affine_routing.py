import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from SkingToolkit.dense_uv_parser.losses import DenseUVParserLoss, _balanced_cross_entropy
from SkingToolkit.dense_uv_parser.model import DenseUVParserNet
from SkingToolkit.dense_uv_parser import train as parser_train
from SkingToolkit.uv_inpainting import train as inpainting_train
from SkingToolkit.dense_uv_parser.utils import (
    augment_dense_batch,
    canonicalize_parser_render,
    canonicalize_tensor,
    classify_route_role,
    conditioning_to_pred_uv,
    estimate_solid_background_foreground,
    fill_geometry_grid_debug,
    overlay_geometry_grid_debug,
    refine_parser_affine,
    render_direct_uv,
    soft_splat_geometry_predictions_to_uv,
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
        self.register_buffer("bg_color", torch.tensor([0.5, 0.5, 0.5]))
        self.register_buffer("front_inner_grid", grid.clone())
        self.register_buffer("front_outer_grid", grid.clone())
        self.register_buffer("front_inner_mask", mask)
        self.register_buffer("front_outer_mask", torch.zeros_like(mask))


class FakeGeometryModel(nn.Module):
    def forward(self, rendered, view_ids=None):
        batch, _, height, width = rendered.shape
        return {
            "foreground": torch.full((batch, 1, height, width), 10.0),
            "layer": torch.cat(
                [
                    torch.full((batch, 1, height, width), 10.0),
                    torch.full((batch, 1, height, width), -10.0),
                    torch.full((batch, 1, height, width), -10.0),
                ],
                dim=1,
            ),
            "affine": torch.zeros(batch, 3),
        }


def dense_targets(batch, height, width):
    return {
        "foreground": torch.ones(batch, 1, height, width),
        "layer": torch.zeros(batch, height, width, dtype=torch.long),
        "route_role": torch.zeros(batch, height, width, dtype=torch.long),
        "part": torch.zeros(batch, height, width, dtype=torch.long),
        "face": torch.zeros(batch, height, width, dtype=torch.long),
        "surface": torch.zeros(batch, height, width, dtype=torch.long),
        "uv": torch.zeros(batch, 2, height, width),
    }


class GlobalAffineRoutingTest(unittest.TestCase):
    def test_soft_geometry_splat_backpropagates_to_role_and_surface_logits(self):
        renderer = FakeRenderer(valid_pixels=1)
        composite_grid = renderer.front_inner_grid.unsqueeze(0).clone()
        composite_grid[0, 0, 0, 0] = (1.0 / 63.0) * 2.0 - 1.0
        renderer.register_buffer("front_composite_grid_layers", composite_grid)
        renderer.register_buffer(
            "front_composite_mask_layers", renderer.front_inner_mask.unsqueeze(0).clone()
        )
        renderer.register_buffer(
            "front_composite_is_decor_layers",
            torch.zeros_like(renderer.front_inner_mask.unsqueeze(0), dtype=torch.bool),
        )
        rendered = torch.zeros(1, 4, 8, 8)
        rendered[:, 0, 0, 0] = 1.0
        rendered[:, 3] = 1.0
        foreground = torch.zeros(1, 1, 8, 8, requires_grad=True)
        role_logits = torch.zeros(1, 3, 8, 8, requires_grad=True)
        surface_logits = torch.zeros(1, 3, 8, 8, requires_grad=True)
        outputs = {
            "foreground": foreground,
            "layer": role_logits,
            "surface": surface_logits,
            "affine": torch.zeros(1, 3),
        }

        pred_uv = soft_splat_geometry_predictions_to_uv(
            rendered,
            outputs,
            renderer=renderer,
            views=["front"],
        )
        loss = pred_uv[0, 3, 0, 1] + pred_uv[0, 0, 0, 1]
        loss.backward()

        self.assertEqual(tuple(pred_uv.shape), (1, 4, 64, 64))
        self.assertGreater(float(foreground.grad.abs().sum()), 0.0)
        self.assertGreater(float(role_logits.grad.abs().sum()), 0.0)
        self.assertGreater(float(surface_logits.grad.abs().sum()), 0.0)

    def test_direct_uv_renderer_is_differentiable(self):
        renderer = FakeRenderer(valid_pixels=1)
        skin = torch.zeros(1, 4, 64, 64, requires_grad=True)

        rendered = render_direct_uv(skin, renderer, "front")
        rendered.sum().backward()

        self.assertEqual(tuple(rendered.shape), (1, 4, 8, 8))
        self.assertGreater(float(skin.grad.abs().sum()), 0.0)

    def test_conditioning_merges_to_preliminary_pred_uv(self):
        conditioning = torch.full((1, 10, 2, 2), 0.5)
        conditioning[:, 3] = 0.0
        conditioning[:, 4] = 0.0
        conditioning[:, 8] = 0.0
        conditioning[:, 9] = 0.0

        conditioning[0, 0:4, 0, 0] = torch.tensor([1.0, 0.0, 0.0, 1.0])
        conditioning[0, 4, 0, 0] = 1.0
        conditioning[0, 5:9, 0, 1] = torch.tensor([0.0, 0.0, 1.0, 1.0])
        conditioning[0, 9, 0, 1] = 1.0
        conditioning[0, 0:4, 1, 0] = torch.tensor([0.0, 1.0, 0.0, 1.0])
        conditioning[0, 4, 1, 0] = 1.0
        conditioning[0, 5:9, 1, 0] = torch.tensor([1.0, 1.0, 0.0, 1.0])
        conditioning[0, 9, 1, 0] = 1.0

        pred_uv = conditioning_to_pred_uv(conditioning)

        self.assertEqual(tuple(pred_uv.shape), (1, 4, 2, 2))
        self.assertTrue(
            torch.equal(pred_uv[0, :, 0, 0], torch.tensor([1.0, 0.0, 0.0, 1.0]))
        )
        self.assertTrue(
            torch.equal(pred_uv[0, :, 0, 1], torch.tensor([0.0, 0.0, 1.0, 1.0]))
        )
        self.assertTrue(
            torch.equal(pred_uv[0, :, 1, 0], torch.tensor([1.0, 1.0, 0.0, 1.0]))
        )
        self.assertTrue(
            torch.equal(pred_uv[0, :, 1, 1], torch.tensor([0.5, 0.5, 0.5, 0.0]))
        )

    def test_training_defaults_keep_geometry_fixed(self):
        parser_args = parser_train.build_arg_parser().parse_args([])
        self.assertFalse(parser_args.augment)
        self.assertFalse(parser_args.augment_validation)
        self.assertEqual(parser_args.translation_scale, 0.0)
        self.assertEqual(parser_args.scale_range, 0.0)
        self.assertFalse(parser_args.affine_refine)
        self.assertEqual(parser_args.affine_refine_translation_px, 0.0)
        self.assertGreater(parser_args.lambda_soft_uv_rgb, 0.0)
        self.assertGreater(parser_args.lambda_render_rgb, 0.0)

        inpainting_args = inpainting_train.build_arg_parser().parse_args(
            ["--data_dir", "unused"]
        )
        self.assertFalse(inpainting_args.augment)
        self.assertFalse(inpainting_args.augment_validation)
        self.assertEqual(inpainting_args.translation_scale, 0.0)
        self.assertEqual(inpainting_args.scale_range, 0.0)
        self.assertEqual(inpainting_args.perspective_scale, 0.0)

    def test_geometry_model_emits_exact_surface_head(self):
        model = DenseUVParserNet(
            base_channels=8,
            view_classes=1,
            predict_affine=True,
            surface_classes=3,
            geometry_only=True,
        )
        rendered = torch.rand(1, 4, 16, 16)

        outputs = model(rendered, view_ids=torch.zeros(1, dtype=torch.long))

        self.assertEqual(tuple(outputs["surface"].shape), (1, 3, 16, 16))
        self.assertNotIn("part", outputs)
        self.assertNotIn("uv", outputs)

    def test_geometry_surface_head_routes_secondary_to_exact_uv(self):
        renderer = FakeRenderer(valid_pixels=1)
        composite_grid = renderer.front_inner_grid.unsqueeze(0).clone()
        composite_grid[0, 0, 0, 0] = (1.0 / 63.0) * 2.0 - 1.0
        renderer.register_buffer("front_composite_grid_layers", composite_grid)
        renderer.register_buffer(
            "front_composite_mask_layers", renderer.front_inner_mask.unsqueeze(0).clone()
        )
        renderer.register_buffer(
            "front_composite_is_decor_layers",
            torch.zeros_like(renderer.front_inner_mask.unsqueeze(0), dtype=torch.bool),
        )
        rendered = torch.full((1, 4, 8, 8), 0.5)
        rendered[:, :3, 0, 0] = torch.tensor([1.0, 0.0, 0.0])
        rendered[:, 3] = 1.0
        role_logits = torch.full((1, 3, 8, 8), -10.0)
        role_logits[:, 2, 0, 0] = 10.0
        surface_logits = torch.full((1, 3, 8, 8), -10.0)
        surface_logits[:, 2, 0, 0] = 10.0
        outputs = {
            "foreground": torch.full((1, 1, 8, 8), 10.0),
            "layer": role_logits,
            "surface": surface_logits,
            "affine": torch.zeros(1, 3),
        }
        observed = torch.zeros(1, 8, 8, dtype=torch.bool)
        observed[:, 0, 0] = True

        conditioning, details = splat_parser_predictions_to_uv_conditioning(
            rendered,
            outputs,
            renderer=renderer,
            views=["front"],
            group_size=1,
            affine_refine=False,
            observed_foreground=observed,
            return_details=True,
        )

        routing = details["routing"]
        self.assertTrue(routing["foreground"][0, 0, 0])
        self.assertTrue(routing["secondary_routed"][0, 0, 0])
        self.assertEqual(int(routing["surface"][0, 0, 0]), 2)
        self.assertEqual(int(routing["flat_uv"][0, 0, 0]), 1)
        self.assertEqual(float(conditioning[0, 4, 0, 1]), 1.0)
        self.assertTrue(torch.equal(conditioning[0, :3, 0, 1], torch.tensor([1.0, 0.0, 0.0])))

    def test_surface_head_cannot_change_primary_role(self):
        renderer = FakeRenderer(valid_pixels=1)
        renderer.front_outer_mask.copy_(renderer.front_inner_mask)
        renderer.front_inner_grid[0, 0, 0] = (8.0 / 63.0) * 2.0 - 1.0
        renderer.front_outer_grid[0, 0, 0] = (40.0 / 63.0) * 2.0 - 1.0
        composite_grid = renderer.front_inner_grid.unsqueeze(0).clone()
        composite_grid[0, 0, 0, 0] = (48.0 / 63.0) * 2.0 - 1.0
        renderer.register_buffer("front_composite_grid_layers", composite_grid)
        renderer.register_buffer(
            "front_composite_mask_layers", renderer.front_inner_mask.unsqueeze(0).clone()
        )
        renderer.register_buffer(
            "front_composite_is_decor_layers",
            torch.zeros_like(renderer.front_inner_mask.unsqueeze(0), dtype=torch.bool),
        )
        rendered = torch.full((1, 4, 8, 8), 0.5)
        rendered[:, :3, 0, 0] = torch.tensor([1.0, 0.0, 0.0])
        rendered[:, 3] = 1.0
        role_logits = torch.full((1, 3, 8, 8), -8.0)
        role_logits[:, 0] = 8.0
        observed = torch.zeros(1, 8, 8, dtype=torch.bool)
        observed[:, 0, 0] = True

        for wrong_surface in (1, 2):
            with self.subTest(wrong_surface=wrong_surface):
                surface_logits = torch.full((1, 3, 8, 8), -20.0)
                surface_logits[:, wrong_surface] = 20.0
                outputs = {
                    "foreground": torch.full((1, 1, 8, 8), 10.0),
                    "layer": role_logits,
                    "surface": surface_logits,
                    "affine": torch.zeros(1, 3),
                }

                conditioning, details = splat_parser_predictions_to_uv_conditioning(
                    rendered,
                    outputs,
                    renderer=renderer,
                    views=["front"],
                    group_size=1,
                    affine_refine=False,
                    observed_foreground=observed,
                    return_details=True,
                )

                routing = details["routing"]
                self.assertEqual(int(routing["route_role"][0, 0, 0]), 0)
                self.assertFalse(routing["secondary"][0, 0, 0])
                self.assertEqual(int(routing["surface"][0, 0, 0]), 0)
                self.assertEqual(int(routing["flat_uv"][0, 0, 0]), 8)
                self.assertEqual(float(conditioning[0, 4, 0, 8]), 1.0)
                self.assertEqual(float(conditioning[0, 9].sum()), 0.0)
                self.assertTrue(
                    torch.equal(
                        conditioning[0, :3, 0, 8],
                        torch.tensor([1.0, 0.0, 0.0]),
                    )
                )

    def test_geometry_surface_routing_uses_projected_texel_consensus(self):
        renderer = FakeRenderer(mask=torch.ones(1, 4))
        composite_grid = renderer.front_inner_grid.unsqueeze(0).clone()
        composite_grid[..., 0] = (1.0 / 63.0) * 2.0 - 1.0
        renderer.register_buffer("front_composite_grid_layers", composite_grid)
        renderer.register_buffer(
            "front_composite_mask_layers", renderer.front_inner_mask.unsqueeze(0).clone()
        )
        renderer.register_buffer(
            "front_composite_is_decor_layers",
            torch.zeros_like(renderer.front_inner_mask.unsqueeze(0), dtype=torch.bool),
        )
        rendered = torch.rand(1, 4, 1, 4)
        rendered[:, 3] = 1.0
        role_logits = torch.full((1, 3, 1, 4), -8.0)
        role_logits[:, 2, 0, :3] = 8.0
        role_logits[:, 0, 0, 3] = 0.2
        role_logits[:, 2, 0, 3] = 0.0
        surface_logits = torch.full((1, 3, 1, 4), -8.0)
        surface_logits[:, 2, 0, :3] = 8.0
        surface_logits[:, 0, 0, 3] = 0.2
        surface_logits[:, 2, 0, 3] = 0.0
        outputs = {
            "foreground": torch.full((1, 1, 1, 4), 10.0),
            "layer": role_logits,
            "surface": surface_logits,
            "affine": torch.zeros(1, 3),
        }

        _, details = splat_parser_predictions_to_uv_conditioning(
            rendered,
            outputs,
            renderer=renderer,
            views=["front"],
            group_size=1,
            affine_refine=False,
            observed_foreground=torch.ones(1, 1, 4, dtype=torch.bool),
            return_details=True,
        )

        routing = details["routing"]
        self.assertTrue(torch.equal(routing["surface"], torch.full_like(routing["surface"], 2)))
        self.assertEqual(int(routing["secondary_routed"].sum()), 4)
        self.assertEqual(int(routing["foreground"].sum()), 4)

    def test_primary_outer_coverage_uses_direct_surface(self):
        renderer = FakeRenderer(mask=torch.ones(1, 4))
        renderer.front_outer_mask.copy_(renderer.front_inner_mask)
        composite_grid = renderer.front_outer_grid.unsqueeze(0).clone()
        composite_mask = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])
        renderer.register_buffer("front_composite_grid_layers", composite_grid)
        renderer.register_buffer("front_composite_mask_layers", composite_mask)
        renderer.register_buffer(
            "front_composite_is_decor_layers",
            torch.ones_like(composite_mask, dtype=torch.bool),
        )
        rendered = torch.rand(1, 4, 1, 4)
        rendered[:, 3] = 1.0
        role_logits = torch.full((1, 3, 1, 4), -8.0)
        role_logits[:, 1] = 8.0
        surface_logits = torch.full((1, 3, 1, 4), -8.0)
        surface_logits[:, 2] = 8.0
        outputs = {
            "foreground": torch.full((1, 1, 1, 4), 10.0),
            "layer": role_logits,
            "surface": surface_logits,
            "affine": torch.zeros(1, 3),
        }
        observed = torch.tensor([[[True, True, False, False]]])

        _, details = splat_parser_predictions_to_uv_conditioning(
            rendered,
            outputs,
            renderer=renderer,
            views=["front"],
            group_size=1,
            affine_refine=False,
            observed_foreground=observed,
            outer_route_confidence_threshold=0.0,
            outer_route_margin_threshold=0.0,
            outer_uv_min_coverage=0.75,
            return_details=True,
        )

        routing = details["routing"]
        self.assertEqual(int(routing["surface"][0, 0, 0]), 1)
        self.assertEqual(int(routing["foreground"].sum()), 0)
        self.assertTrue(
            torch.allclose(
                routing["outer_uv_coverage"][0, 0, :2],
                torch.full((2,), 0.5),
            )
        )

    def test_geometry_fill_hides_unclassified_theoretical_grid(self):
        rendered = torch.zeros(1, 4, 2, 2)
        rendered[:, :3] = torch.tensor([1.0, 0.0, 0.0]).view(1, 3, 1, 1)
        foreground = torch.tensor([[[True, False], [False, True]]])
        layer = torch.tensor([[[0, 0], [1, 1]]])
        grid = torch.ones(1, 3, 2, 2)
        geometry = torch.ones(1, 2, 2, dtype=torch.bool)
        geometry_debug = (grid, grid, geometry, geometry, geometry, geometry)

        inner, outer = fill_geometry_grid_debug(
            rendered,
            foreground,
            layer,
            geometry_debug,
        )

        gray = torch.tensor([128.0 / 255.0] * 3)
        self.assertTrue(torch.allclose(inner[0, :, 0, 0], torch.tensor([1.0, 0.0, 0.0])))
        self.assertTrue(torch.allclose(outer[0, :, 1, 1], torch.tensor([1.0, 0.0, 0.0])))
        self.assertTrue(torch.allclose(inner[0, :, 1, 1], gray))
        self.assertTrue(torch.allclose(outer[0, :, 0, 0], gray))

    def test_geometry_grid_overlay_preserves_source_and_marks_each_layer(self):
        rendered = torch.full((1, 4, 2, 2), 0.4)
        inner_mask = torch.tensor([[[True, True], [False, False]]])
        outer_mask = torch.tensor([[[False, False], [True, True]]])
        inner_edge = torch.tensor([[[True, False], [False, False]]])
        outer_edge = torch.tensor([[[False, False], [False, True]]])
        grid = torch.zeros(1, 3, 2, 2)
        geometry_debug = (
            grid,
            grid,
            inner_mask,
            outer_mask,
            inner_edge,
            outer_edge,
        )

        inner, outer = overlay_geometry_grid_debug(rendered, geometry_debug)

        self.assertTrue(torch.allclose(inner[0, :, 0, 0], torch.tensor([0.0, 1.0, 1.0])))
        self.assertTrue(torch.allclose(outer[0, :, 1, 1], torch.tensor([1.0, 0.25, 0.85])))
        self.assertTrue(torch.allclose(inner[0, :, 1, 0], torch.full((3,), 0.4)))
        self.assertFalse(torch.allclose(inner[0, :, 0, 1], torch.full((3,), 0.4)))

        inner_base = torch.zeros(1, 3, 2, 2)
        outer_base = torch.ones(1, 3, 2, 2)
        inner_routed, outer_routed = overlay_geometry_grid_debug(
            rendered,
            geometry_debug,
            base_images=(inner_base, outer_base),
        )
        self.assertTrue(torch.allclose(inner_routed[0, :, 1, 0], torch.zeros(3)))
        self.assertTrue(torch.allclose(outer_routed[0, :, 0, 0], torch.ones(3)))

    def test_solid_background_mask_preserves_enclosed_matching_color(self):
        rendered = torch.zeros(1, 4, 16, 16)
        rendered[:, :3] = torch.tensor([0.2, 0.7, 0.8]).view(1, 3, 1, 1)
        rendered[:, 3] = 1.0
        rendered[:, :3, 4:12, 4:12] = 0.9
        rendered[:, :3, 7:9, 7:9] = torch.tensor([0.2, 0.7, 0.8]).view(1, 3, 1, 1)

        foreground = estimate_solid_background_foreground(rendered)

        self.assertFalse(foreground[0, 0, 0])
        self.assertTrue(foreground[0, 5, 5])
        self.assertTrue(foreground[0, 7, 7])

    def test_geometry_routing_rejects_solid_background_inside_mapping(self):
        renderer = FakeRenderer(mask=torch.ones(16, 16))
        rendered = torch.zeros(1, 4, 16, 16)
        rendered[:, :3] = torch.tensor([0.2, 0.7, 0.8]).view(1, 3, 1, 1)
        rendered[:, :3, 6:10, 6:10] = 0.9
        rendered[:, 3] = 1.0
        outputs = {
            "foreground": torch.full((1, 1, 16, 16), 10.0),
            "layer": torch.cat(
                [
                    torch.full((1, 1, 16, 16), 10.0),
                    torch.full((1, 1, 16, 16), -10.0),
                    torch.full((1, 1, 16, 16), -10.0),
                ],
                dim=1,
            ),
            "affine": torch.zeros(1, 3),
        }

        _, details = splat_parser_predictions_to_uv_conditioning(
            rendered,
            outputs,
            renderer=renderer,
            views=["front"],
            group_size=1,
            affine_refine=False,
            return_details=True,
        )

        routing = details["routing"]
        self.assertEqual(int(routing["foreground"].sum()), 16)
        self.assertEqual(int(routing["background_rejected"].sum()), 240)

    def test_route_role_marks_only_mismatched_visible_uv_as_secondary(self):
        static = {
            "masks": torch.ones(2, 1, 2, dtype=torch.bool),
            "flat_uv": torch.tensor([[[7, 8]], [[40, 41]]]),
        }
        layer = torch.tensor([[[0, 1]]])
        visible_uv = torch.tensor([[[7, 55]]])
        valid = torch.ones(1, 1, 2, dtype=torch.bool)

        role = classify_route_role(static, layer, visible_uv, valid)

        self.assertTrue(torch.equal(role, torch.tensor([[[0, 2]]])))

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

    def test_geometry_model_only_emits_fit_and_visibility_heads(self):
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
            surface_classes=0,
            geometry_only=True,
        )
        outputs = model(rendered, view_ids=torch.zeros(1, dtype=torch.long))
        self.assertEqual(set(outputs), {"foreground", "layer", "affine"})
        losses = DenseUVParserLoss(use_uv=False)(outputs, targets)
        self.assertTrue(torch.isfinite(losses["loss_geometry"]))
        self.assertIn("precision_outer", losses)

    def test_geometry_routing_uses_fixed_inner_outer_uv_maps(self):
        renderer = FakeRenderer(valid_pixels=1)
        renderer.front_outer_mask.copy_(renderer.front_inner_mask)
        renderer.front_inner_grid[0, 0, 0] = (8.0 / 63.0) * 2.0 - 1.0
        renderer.front_outer_grid[0, 0, 0] = (40.0 / 63.0) * 2.0 - 1.0
        rendered = torch.rand(1, 4, 8, 8)
        rendered[:, 3] = 1.0
        outputs = {
            "foreground": torch.full((1, 1, 8, 8), 10.0),
            "layer": torch.cat(
                [
                    torch.full((1, 1, 8, 8), -10.0),
                    torch.full((1, 1, 8, 8), 10.0),
                    torch.full((1, 1, 8, 8), -10.0),
                ],
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
            affine_refine=False,
            return_details=True,
        )

        self.assertEqual(int(details["routing"]["surface"][0, 0, 0]), 1)
        self.assertEqual(int(details["routing"]["layer"][0, 0, 0]), 1)
        self.assertEqual(int(conditioning[:, 4:5].sum()), 0)
        self.assertEqual(int(conditioning[:, 9:10].sum()), 1)

        outputs["layer"] = torch.cat(
            [
                torch.full((1, 1, 8, 8), 10.0),
                torch.full((1, 1, 8, 8), -10.0),
                torch.full((1, 1, 8, 8), -10.0),
            ],
            dim=1,
        )
        inner_conditioning, inner_details = splat_parser_predictions_to_uv_conditioning(
            rendered,
            outputs,
            renderer=renderer,
            views=["front"],
            group_size=1,
            affine_refine=False,
            return_details=True,
        )
        self.assertEqual(int(inner_details["routing"]["surface"][0, 0, 0]), 0)
        self.assertEqual(int(inner_conditioning[:, 4:5].sum()), 1)
        self.assertEqual(int(inner_conditioning[:, 9:10].sum()), 0)

    def test_outer_uv_coverage_rejects_partial_texel_without_removing_inner(self):
        mask = torch.ones(1, 2)
        renderer = FakeRenderer(mask=mask)
        renderer.front_outer_mask.copy_(mask)
        rendered = torch.rand(1, 4, 1, 2)
        rendered[:, 3] = 1.0
        outputs = {
            "foreground": torch.full((1, 1, 1, 2), 10.0),
            "layer": torch.tensor(
                [[[[ -10.0, 10.0]], [[10.0, -10.0]], [[-10.0, -10.0]]]]
            ),
            "affine": torch.zeros(1, 3),
        }

        conditioning, details = splat_parser_predictions_to_uv_conditioning(
            rendered,
            outputs,
            renderer=renderer,
            views=["front"],
            group_size=1,
            affine_refine=False,
            outer_uv_min_coverage=0.75,
            geometry_route_texel_consensus=False,
            return_details=True,
        )

        self.assertEqual(int(conditioning[:, 9:10].sum()), 0)
        self.assertEqual(int(conditioning[:, 4:5].sum()), 1)
        self.assertEqual(int(details["routing"]["rejected"][0, 0, 0]), 1)
        self.assertAlmostEqual(float(details["routing"]["outer_uv_coverage"][0, 0, 0]), 0.5)

    def test_geometry_secondary_backface_is_visible_in_debug_but_not_splatted(self):
        renderer = FakeRenderer(valid_pixels=1)
        renderer.front_outer_mask.copy_(renderer.front_inner_mask)
        rendered = torch.rand(1, 4, 8, 8)
        rendered[:, 3] = 1.0
        outputs = {
            "foreground": torch.full((1, 1, 8, 8), 10.0),
            "layer": torch.cat(
                [
                    torch.full((1, 1, 8, 8), -10.0),
                    torch.full((1, 1, 8, 8), -10.0),
                    torch.full((1, 1, 8, 8), 10.0),
                ],
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
            affine_refine=False,
            return_details=True,
        )

        self.assertTrue(details["routing"]["secondary"][0, 0, 0])
        self.assertEqual(int(details["routing"]["foreground"].sum()), 0)
        self.assertEqual(int(conditioning[:, 4:5].sum() + conditioning[:, 9:10].sum()), 0)

    def test_geometry_route_role_uses_projected_texel_consensus(self):
        mask = torch.ones(1, 4)
        renderer = FakeRenderer(mask=mask)
        rendered = torch.rand(1, 4, 1, 4)
        rendered[:, 3] = 1.0
        outputs = {
            "foreground": torch.full((1, 1, 1, 4), 10.0),
            "layer": torch.tensor(
                [[[[8.0, 8.0, -8.0, 8.0]], [[-8.0] * 4], [[-8.0, -8.0, 8.0, -8.0]]]]
            ),
            "affine": torch.zeros(1, 3),
        }

        _, details = splat_parser_predictions_to_uv_conditioning(
            rendered,
            outputs,
            renderer=renderer,
            views=["front"],
            group_size=1,
            affine_refine=False,
            return_details=True,
        )

        routing = details["routing"]
        self.assertEqual(int((routing["raw_route_role"] == 2).sum()), 1)
        self.assertEqual(int(routing["secondary"].sum()), 0)
        self.assertEqual(int(routing["foreground"].sum()), 4)
        self.assertTrue(torch.equal(routing["route_role"], torch.zeros_like(routing["route_role"])))

    def test_geometry_secondary_requires_absolute_texel_majority(self):
        renderer = FakeRenderer(mask=torch.ones(1, 4))
        rendered = torch.rand(1, 4, 1, 4)
        rendered[:, 3] = 1.0
        outputs = {
            "foreground": torch.full((1, 1, 1, 4), 10.0),
            "layer": torch.tensor(
                [[[[0.0] * 4], [[-0.2] * 4], [[0.1] * 4]]]
            ),
            "affine": torch.zeros(1, 3),
        }

        _, details = splat_parser_predictions_to_uv_conditioning(
            rendered,
            outputs,
            renderer=renderer,
            views=["front"],
            group_size=1,
            affine_refine=False,
            return_details=True,
        )

        routing = details["routing"]
        self.assertEqual(int((routing["raw_route_role"] == 2).sum()), 4)
        self.assertEqual(int(routing["secondary"].sum()), 0)
        self.assertEqual(int(routing["foreground"].sum()), 4)

    def test_geometry_training_preview_has_no_semantic_heads(self):
        renderer = FakeRenderer(valid_pixels=1)
        rendered = torch.rand(1, 4, 8, 8)
        rendered[:, 3] = 1.0
        targets = dense_targets(1, 8, 8)
        targets["affine"] = torch.zeros(1, 3)
        args = SimpleNamespace(
            views="front",
            splat_fg_threshold=0.5,
            bg_color=(128, 128, 128),
            semantic_gate=True,
            affine_refine=False,
            affine_refine_translation_px=2.0,
            affine_refine_scale=0.0,
            route_confidence_threshold=0.0,
            route_margin_threshold=0.0,
            outer_route_confidence_threshold=0.1,
            outer_route_margin_threshold=0.2,
            outer_uv_min_coverage=0.5,
            splat_color_aggregation="exact_mode",
            allow_semantic_fallback=False,
        )
        loader = [{"uv": torch.zeros(1, 4, 64, 64), "path": ["test.png"]}]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "preview.png"
            with patch.object(
                parser_train,
                "build_parser_inputs",
                return_value=(rendered, targets, 1, torch.zeros(1, dtype=torch.long)),
            ):
                parser_train.save_preview(
                    FakeGeometryModel(),
                    renderer,
                    loader,
                    torch.device("cpu"),
                    args,
                    output,
                    max_items=1,
                )
            self.assertTrue(output.exists())
            self.assertTrue(output.with_name("preview_debug.png").exists())

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

    def test_canonical_render_uses_background_fill_outside_input(self):
        rendered = torch.zeros(1, 4, 16, 16)
        fill = torch.tensor([0.2, 0.7, 0.8, 1.0]).view(1, 4, 1, 1)
        rendered[:] = fill
        canonical = canonicalize_parser_render(
            rendered,
            {"affine": torch.tensor([[0.25, -0.25, 0.0]])},
            fill_color=fill,
        )
        self.assertTrue(torch.allclose(canonical, rendered))

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

    def test_affine_refinement_prefers_observed_mask_over_noisy_head(self):
        height = width = 32
        target = torch.zeros(height, width)
        target[7:25, 9:23] = 1.0
        shifted = torch.zeros_like(target)
        shifted[:, 1:] = target[:, :-1]
        renderer = FakeRenderer(mask=target)
        outputs = {
            "foreground": torch.where(
                target.view(1, 1, height, width) > 0.5,
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
            observed_foreground=shifted.unsqueeze(0) > 0.5,
        )

        self.assertTrue(details["accepted"][0])
        self.assertAlmostEqual(float(details["translation_px"][0, 0]), 1.0, places=4)
        self.assertAlmostEqual(float(refined[0, 0]), 2.0 / width, places=4)

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
            reject_semantic_fallback=True,
            return_details=True,
        )

        self.assertEqual(int(details["routing"]["foreground"].sum()), 1)
        self.assertTrue(details["routing"]["semantic_fallback"][0, 0, 0])
        self.assertEqual(int(conditioning[:, 4:5].sum()), 1)

        _, strict_details = splat_parser_predictions_to_uv_conditioning(
            rendered,
            outputs,
            renderer=renderer,
            views=["front"],
            group_size=1,
            semantic_gate=True,
            affine_refine=False,
            reject_semantic_fallback=True,
            reject_inner_semantic_fallback=True,
            return_details=True,
        )
        self.assertEqual(int(strict_details["routing"]["foreground"].sum()), 0)

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

    def test_exact_mode_uses_majority_color_after_layer_separation(self):
        rendered = torch.zeros(4, 4, 1, 1)
        rendered[0:2, 0, 0, 0] = 1.0
        rendered[2, 2, 0, 0] = 1.0
        rendered[3, 1, 0, 0] = 1.0
        rendered[:, 3, 0, 0] = 1.0
        fg = torch.ones(4, 1, 1, dtype=torch.bool)
        layer = torch.tensor([0, 0, 1, 1]).view(4, 1, 1)
        flat_uv = torch.zeros(4, 1, 1, dtype=torch.long)
        confidence = torch.tensor([0.6, 0.6, 0.99, 0.5]).view(4, 1, 1)

        conditioning = splat_to_uv_conditioning(
            rendered,
            fg,
            layer,
            flat_uv,
            group_size=4,
            confidence=confidence,
            color_aggregation="exact_mode",
        )

        self.assertTrue(torch.equal(conditioning[0, :3, 0, 0], torch.tensor([1.0, 0.0, 0.0])))
        self.assertTrue(torch.equal(conditioning[0, 5:8, 0, 0], torch.tensor([0.0, 0.0, 1.0])))

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

    def test_outer_threshold_does_not_remove_inner_pixels(self):
        renderer = FakeRenderer(valid_pixels=1)
        rendered = torch.rand(1, 4, 8, 8)
        rendered[:, 3] = 1.0
        outputs = {
            "foreground": torch.full((1, 1, 8, 8), 10.0),
            "layer": torch.zeros(1, 2, 8, 8),
            "part": torch.zeros(1, 6, 8, 8),
            "face": torch.zeros(1, 6, 8, 8),
            "surface": torch.cat(
                [torch.full((1, 1, 8, 8), 10.0), torch.full((1, 1, 8, 8), -10.0)],
                dim=1,
            ),
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
            route_confidence_threshold=0.0,
            route_margin_threshold=0.0,
            outer_route_confidence_threshold=1.0,
            outer_route_margin_threshold=1.0,
            return_details=True,
        )

        self.assertEqual(int(details["routing"]["foreground"].sum()), 1)


if __name__ == "__main__":
    unittest.main()
