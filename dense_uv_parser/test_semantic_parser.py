import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from SkingToolkit.dense_uv_parser.infer import (
    save_parser_uv,
    save_simple_inpaint_uv,
)
from SkingToolkit.dense_uv_parser.losses import DenseUVParserLoss
from SkingToolkit.dense_uv_parser.model import DenseUVParserNet
from SkingToolkit.dense_uv_parser.semantic import (
    attach_semantic_runtime,
    cached_semantic_batch,
)
from SkingToolkit.dense_uv_parser.train import outer_uv_occupancy_losses
from SkingToolkit.dense_uv_parser.utils import splat_to_uv_conditioning
from SkingToolkit.semantic_uv_reconstruction.semantic_losses import (
    build_part_layer_masks,
)


class SemanticDenseUVParserTest(unittest.TestCase):
    def build_model(self):
        torch.manual_seed(11)
        return DenseUVParserNet(
            base_channels=8,
            view_classes=2,
            predict_affine=True,
            surface_classes=4,
            geometry_only=True,
            semantic_feature_dim=16,
            semantic_channels=16,
            semantic_attention_heads=4,
            semantic_layers=1,
            semantic_dropout=0.0,
            predict_confidence=True,
            predict_outer_uv_occupancy=True,
        )

    def test_multiview_semantics_condition_dense_outputs(self):
        model = self.build_model()
        images = torch.rand(4, 4, 32, 32)
        view_ids = torch.tensor([0, 1, 0, 1])
        semantics = torch.rand(2, 2, 16)
        outputs = model(images, view_ids=view_ids, semantic_features=semantics)

        self.assertEqual(tuple(outputs["layer"].shape), (4, 3, 32, 32))
        self.assertEqual(tuple(outputs["route_confidence"].shape), (4, 1, 32, 32))
        self.assertEqual(tuple(outputs["outer_presence_logits"].shape), (2, 6))
        self.assertEqual(tuple(outputs["outer_coverage"].shape), (2, 6))
        self.assertEqual(
            tuple(outputs["outer_uv_occupancy_logits"].shape),
            (2, 1, 64, 64),
        )

        loss = (
            outputs["layer"].mean()
            + outputs["outer_coverage"].mean()
            + outputs["outer_uv_occupancy_logits"].mean()
        )
        loss.backward()
        gradient = model.semantic_fusion.input_projection[1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)
        occupancy_gradient = (
            model.outer_uv_occupancy_head[-1].weight.grad
        )
        self.assertIsNotNone(occupancy_gradient)
        self.assertGreater(float(occupancy_gradient.abs().sum()), 0.0)

    def test_spatial_semantics_start_as_zero_residual_then_learn(self):
        model = DenseUVParserNet(
            base_channels=8,
            view_classes=2,
            predict_affine=True,
            surface_classes=4,
            geometry_only=True,
            semantic_feature_dim=16,
            semantic_channels=16,
            semantic_attention_heads=4,
            semantic_layers=1,
            semantic_dropout=0.0,
            semantic_spatial_feature_dim=12,
            semantic_spatial_channels=8,
        )
        images = torch.rand(2, 4, 32, 32)
        view_ids = torch.tensor([0, 1])
        global_features = torch.rand(2, 16)
        first = model(
            images,
            view_ids=view_ids,
            semantic_features={
                "raw_global": global_features,
                "raw_spatial": torch.zeros(2, 12, 7, 5),
            },
        )
        second = model(
            images,
            view_ids=view_ids,
            semantic_features={
                "raw_global": global_features,
                "raw_spatial": torch.rand(2, 12, 7, 5),
            },
        )
        self.assertTrue(torch.equal(first["layer"], second["layer"]))

        second["layer"].square().mean().backward()
        gradient = model.semantic_spatial_fusion.output_projection.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_runtime_semantics_receive_neutralized_background(self):
        class FakeBackbone:
            raw_feature_dim = 16
            raw_spatial_feature_dim = 12

            def __init__(self):
                self.seen = None

            def encode_dense(self, images):
                self.seen = images.detach().clone()
                return {
                    "raw_global": torch.zeros(images.shape[0], 16),
                    "raw_spatial": torch.zeros(images.shape[0], 12, 4, 4),
                }

        model = DenseUVParserNet(
            base_channels=8,
            view_classes=2,
            geometry_only=True,
            semantic_feature_dim=16,
            semantic_channels=16,
            semantic_attention_heads=4,
            semantic_spatial_feature_dim=12,
            semantic_spatial_channels=8,
        )
        backbone = FakeBackbone()
        attach_semantic_runtime(
            model,
            "tipsv2",
            "fake",
            torch.device("cpu"),
            backbone=backbone,
        )
        images = torch.zeros(2, 4, 16, 16)
        images[:, :3, :, 8:] = 1.0
        foreground = torch.zeros(2, 16, 16, dtype=torch.bool)
        foreground[:, :, 8:] = True
        model(
            images,
            view_ids=torch.tensor([0, 1]),
            semantic_foreground=foreground,
        )
        self.assertTrue(
            torch.equal(
                backbone.seen[:, :, :, :8],
                torch.full_like(backbone.seen[:, :, :, :8], 0.5),
            )
        )
        self.assertTrue(
            torch.equal(
                backbone.seen[:, :, :, 8:],
                torch.ones_like(backbone.seen[:, :, :, 8:]),
            )
        )

    def test_cached_semantics_include_global_and_spatial_features(self):
        class FakeCache:
            has_spatial = True

            @staticmethod
            def get(filename):
                value = 1.0 if filename == "one.png" else 2.0
                return torch.full((2, 6), value)

            @staticmethod
            def get_spatial(filename):
                value = 3.0 if filename == "one.png" else 4.0
                return torch.full((2, 6, 3, 2), value, dtype=torch.float16)

        features = cached_semantic_batch(
            FakeCache(),
            ["/tmp/one.png", "/tmp/two.png"],
            torch.device("cpu"),
        )

        self.assertEqual(tuple(features["raw_global"].shape), (2, 2, 6))
        self.assertEqual(tuple(features["raw_spatial"].shape), (2, 2, 6, 3, 2))
        self.assertEqual(features["raw_spatial"].dtype, torch.float16)

    def test_outer_uv_occupancy_loss_uses_only_outer_atlas(self):
        logits = torch.zeros(1, 1, 64, 64, requires_grad=True)
        target_uv = torch.zeros(1, 4, 64, 64)
        _, outer_masks = build_part_layer_masks()
        occupied = outer_masks[:, 0].bool().any(dim=0)
        y, x = occupied.nonzero()[0]
        target_uv[0, 3, y, x] = 1.0

        losses = outer_uv_occupancy_losses(
            logits, target_uv, outer_masks
        )
        total = (
            losses["loss_outer_uv_occupancy_bce"]
            + losses["loss_outer_uv_occupancy_dice"]
        )
        total.backward()

        self.assertTrue(torch.isfinite(total))
        self.assertIsNotNone(logits.grad)
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_outer_uv_occupancy_head_does_not_shift_parser_trunk(self):
        model = self.build_model().train()
        outputs = model(
            torch.rand(2, 4, 32, 32),
            view_ids=torch.tensor([0, 1]),
            semantic_features=torch.rand(1, 2, 16),
        )

        outputs["outer_uv_occupancy_logits"].mean().backward()

        self.assertIsNotNone(
            model.outer_uv_occupancy_head[-1].weight.grad
        )
        self.assertIsNone(model.stem.block[0].weight.grad)
        self.assertIsNone(
            model.semantic_fusion.input_projection[1].weight.grad
        )

    def test_confidence_head_learns_current_route_correctness(self):
        model = self.build_model()
        with torch.no_grad():
            model.route_confidence.weight.zero_()
            model.route_confidence.bias.fill_(10.0)
        images = torch.rand(2, 4, 32, 32)
        view_ids = torch.tensor([0, 1])
        outputs = model(
            images,
            view_ids=view_ids,
            semantic_features=torch.rand(1, 2, 16),
        )
        height, width = outputs["layer"].shape[-2:]
        route_target = outputs["layer"].detach().argmax(dim=1)
        surface_target = outputs["surface"].detach().argmax(dim=1)
        targets = {
            "foreground": torch.ones(2, 1, height, width),
            "route_role": route_target,
            "layer": route_target.clamp_max(1),
            "part": torch.zeros(2, height, width, dtype=torch.long),
            "face": torch.zeros(2, height, width, dtype=torch.long),
            "surface": surface_target,
            "uv": torch.zeros(2, 2, height, width),
            "affine": torch.zeros(2, 3),
        }
        losses = DenseUVParserLoss(lambda_route_confidence=1.0)(outputs, targets)
        self.assertTrue(torch.isfinite(losses["loss_route_confidence"]))
        self.assertAlmostEqual(float(losses["precision_trusted_route"]), 1.0, places=5)

    def test_confidence_aware_splat_has_twelve_channels(self):
        rendered = torch.tensor([[[[0.2]], [[0.4]], [[0.6]], [[1.0]]]])
        conditioning = splat_to_uv_conditioning(
            rendered,
            fg=torch.ones(1, 1, 1, dtype=torch.bool),
            layer=torch.zeros(1, 1, 1, dtype=torch.long),
            flat_uv=torch.zeros(1, 1, 1, dtype=torch.long),
            confidence=torch.full((1, 1, 1), 0.7),
            include_confidence=True,
        )
        self.assertEqual(tuple(conditioning.shape), (1, 12, 64, 64))
        self.assertEqual(float(conditioning[0, 4, 0, 0]), 1.0)
        self.assertAlmostEqual(float(conditioning[0, 5, 0, 0]), 0.7, places=5)

    def test_parser_uv_diagnostic_leaves_unknown_base_texels_transparent(self):
        conditioning = torch.zeros(1, 12, 64, 64)
        conditioning[0, 0:4, 8, 8] = torch.tensor([1.0, 0.0, 0.0, 1.0])
        conditioning[0, 4, 8, 8] = 1.0
        conditioning[0, 5, 8, 8] = 1.0
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "parser.png"
            save_parser_uv(conditioning, output)
            image = Image.open(output).convert("RGBA")
            self.assertEqual(image.getpixel((8, 8)), (255, 0, 0, 255))
            self.assertEqual(image.getpixel((20, 20)), (0, 0, 0, 0))

    def test_simple_parser_uv_inpaint_writes_separate_completed_artifact(self):
        conditioning = torch.zeros(1, 12, 64, 64)
        conditioning[0, 0:4, 8, 8] = torch.tensor([1.0, 0.0, 0.0, 1.0])
        conditioning[0, 4, 8, 8] = 1.0
        conditioning[0, 5, 8, 8] = 1.0
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "simple.png"
            save_simple_inpaint_uv(conditioning, output)
            image = Image.open(output).convert("RGBA")

            self.assertEqual(image.size, (64, 64))
            self.assertEqual(image.getpixel((8, 8)), (255, 0, 0, 255))
            self.assertEqual(image.getpixel((20, 20)), (0, 0, 0, 0))
            self.assertEqual(image.getpixel((40, 8)), (0, 0, 0, 0))
            self.assertEqual(image.getpixel((63, 0)), (0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
