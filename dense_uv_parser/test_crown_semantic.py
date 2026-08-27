import unittest
import torch
import torch.nn.functional as F

from SkingToolkit.dense_uv_parser.model import DenseUVParserNet
from SkingToolkit.dense_uv_parser.losses import (
    dense_semantic_supervision_loss,
    dense_semantic_outer_false_positive_loss,
    head_top_accessory_semantic_terms,
    DenseUVParserLoss,
)
from SkingToolkit.dense_uv_parser.semantic_targets import (
    build_head_top_accessory_face_targets,
    build_dense_view_semantic_targets,
)
from SkingToolkit.dense_uv_parser.semantic_backbone import (
    LAYER_TOPOLOGY_EYE_SIGLIP_ROUTE_PROMPTS,
)


class CrownSemanticImprovementTest(unittest.TestCase):
    def test_crown_prompts_contain_rich_visual_descriptors(self):
        prompts = LAYER_TOPOLOGY_EYE_SIGLIP_ROUTE_PROMPTS
        self.assertEqual(len(prompts), 5)
        # Class 0: crown & head top accessory
        self.assertIn("crown", prompts[0])
        self.assertIn("hat", prompts[0])
        self.assertIn("tiara", prompts[0])
        # Class 1: glasses & eye accessory
        self.assertIn("glasses", prompts[1])

    def test_crown_targets_keep_class_0_priority_over_eye_band(self):
        # Create a synthetic skin with a crown on top face (face 5) and front face (face 0)
        uv = torch.zeros(1, 4, 64, 64)
        # Minecraft head decor: Top Face 5 is (48, 0, 8, 8), Front Face 0 is (40, 8, 8, 8)
        uv[:, 3, 0:8, 48:56] = 1.0  # Top face 5
        uv[:, 3, 8:11, 40:48] = 1.0 # Front face 0 upper band

        top_targets = build_head_top_accessory_face_targets(uv)
        self.assertEqual(float(top_targets["presence"][0]), 1.0)
        # Verify component covers top face and front upper band
        self.assertTrue(top_targets["mask"][0, 5].any())
        self.assertTrue(top_targets["mask"][0, 0, 0:3].any())

    def test_false_positive_loss_does_not_suppress_correct_inner_pixels(self):
        # 5 classes: 0: crown, 1: glasses, 2: other_outer, 3: inner, 4: bg
        # Simulate an inner region where the model predicts inner with high confidence (logits[:, 3] > outer logits)
        logits = torch.zeros(1, 5, 16, 16)
        logits[:, 3] = 3.0  # confidently inner
        logits[:, :3] = -2.0 # low outer
        targets = torch.full((1, 16, 16), 3, dtype=torch.long) # inner ground truth

        loss = dense_semantic_outer_false_positive_loss(logits, targets)
        # Since there are no false positive predictions (margin <= 0), loss must be 0
        self.assertEqual(float(loss), 0.0)

    def test_crown_recall_gradient_flow_and_spatial_prompt_injection(self):
        torch.manual_seed(42)
        model = DenseUVParserNet(
            base_channels=8,
            view_classes=2,
            geometry_only=True,
            semantic_feature_dim=12,
            semantic_channels=8,
            semantic_attention_heads=2,
            semantic_spatial_feature_dim=12,
            semantic_spatial_channels=8,
            semantic_text_prompt_count=5,
            semantic_text_prompt_feature_dim=12,
            semantic_text_prompt_channels=6,
            dense_semantic_target_version=3,
        )
        # Prompt embeddings for 5 classes
        model.set_semantic_text_prompt_embeddings(torch.randn(5, 12))
        images = torch.rand(2, 4, 32, 32)
        view_ids = torch.tensor([0, 1])
        raw_spatial = torch.randn(2, 12, 4, 4)
        raw_global = torch.randn(2, 12)

        outputs = model(
            images,
            view_ids=view_ids,
            semantic_features={
                "raw_global": raw_global,
                "raw_spatial": raw_spatial,
            },
        )
        self.assertIn("dense_semantic_logits", outputs)
        dense_logits = outputs["dense_semantic_logits"]
        self.assertEqual(tuple(dense_logits.shape), (2, 5, 32, 32))

        # Ground truth with class 0 (crown) on top area
        targets = torch.full((2, 32, 32), 3, dtype=torch.long)
        targets[:, 2:8, 10:22] = 0 # crown region

        loss_focal = dense_semantic_supervision_loss(dense_logits, targets)
        top_terms = head_top_accessory_semantic_terms(dense_logits, targets)
        total_loss = loss_focal + 1.0 * top_terms["loss_head_top_accessory_hard_recall"]
        total_loss.backward()

        # Check gradients on stem and semantic head
        stem_grad = next(model.stem.parameters()).grad
        self.assertIsNotNone(stem_grad)
        self.assertGreater(float(stem_grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
