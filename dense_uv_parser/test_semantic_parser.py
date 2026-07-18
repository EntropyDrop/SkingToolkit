import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from SkingToolkit.dense_uv_parser.infer import save_parser_uv
from SkingToolkit.dense_uv_parser.losses import DenseUVParserLoss
from SkingToolkit.dense_uv_parser.model import DenseUVParserNet
from SkingToolkit.dense_uv_parser.utils import splat_to_uv_conditioning


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

        loss = outputs["layer"].mean() + outputs["outer_coverage"].mean()
        loss.backward()
        gradient = model.semantic_fusion.input_projection[1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

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


if __name__ == "__main__":
    unittest.main()
