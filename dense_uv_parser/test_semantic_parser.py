import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
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
from SkingToolkit.dense_uv_parser.train import (
    _head_outer_route_connectivity_terms,
    head_outer_structure_losses,
    outer_uv_occupancy_losses,
)
from SkingToolkit.dense_uv_parser.utils import splat_to_uv_conditioning
from SkingToolkit.dense_uv_parser.semantic_targets import (
    build_head_outer_face_targets,
    build_part_layer_masks,
)
from SkingToolkit.dense_uv_parser.simple_inpainting import (
    simple_symmetry_nearest_inpaint,
)
from SkingToolkit.dense_uv_parser.uv_topology import (
    build_head_outer_face_graph,
    build_outer_uv_graph,
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

    def projected_occupancy(self, model, outputs):
        features = F.interpolate(
            outputs["outer_uv_features"],
            size=(64, 64),
            mode="bilinear",
            align_corners=False,
        )
        features = features.reshape(
            -1,
            model.view_classes,
            features.shape[1],
            64,
            64,
        ).mean(dim=1)
        atlas_features = torch.cat(
            [features, features.new_zeros(features.shape[0], 4, 64, 64)],
            dim=1,
        )
        return model.predict_projected_outer_uv_occupancy(
            atlas_features,
            outputs["outer_uv_global_context"],
        )

    def test_multiview_semantics_condition_dense_outputs(self):
        model = self.build_model()
        images = torch.rand(4, 4, 32, 32)
        view_ids = torch.tensor([0, 1, 0, 1])
        semantics = torch.rand(2, 2, 16)
        outputs = model(images, view_ids=view_ids, semantic_features=semantics)
        occupancy_logits = self.projected_occupancy(model, outputs)

        self.assertEqual(tuple(outputs["layer"].shape), (4, 3, 32, 32))
        self.assertEqual(tuple(outputs["route_confidence"].shape), (4, 1, 32, 32))
        self.assertEqual(tuple(outputs["outer_presence_logits"].shape), (2, 6))
        self.assertEqual(tuple(outputs["outer_coverage"].shape), (2, 6))
        self.assertEqual(
            tuple(occupancy_logits.shape),
            (2, 1, 64, 64),
        )

        loss = (
            outputs["layer"].mean()
            + outputs["outer_coverage"].mean()
            + occupancy_logits.mean()
        )
        loss.backward()
        gradient = model.semantic_fusion.input_projection[1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)
        occupancy_gradient = (
            model.outer_uv_occupancy_head.output[-1].weight.grad
        )
        self.assertIsNotNone(occupancy_gradient)
        self.assertGreater(float(occupancy_gradient.abs().sum()), 0.0)

    def test_privileged_semantics_form_independent_primary_role_groups(self):
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
        )
        images = torch.rand(4, 4, 32, 32)
        semantics = torch.rand(1, 4, 16)
        outputs = model(
            images,
            view_ids=torch.tensor([0, 1, 0, 1]),
            semantic_features=semantics,
        )

        self.assertEqual(tuple(outputs["layer"].shape), (4, 3, 32, 32))
        self.assertEqual(tuple(outputs["outer_presence_logits"].shape), (2, 6))

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

    def test_siglip_text_prompts_add_trainable_route_evidence(self):
        model = DenseUVParserNet(
            base_channels=8,
            view_classes=2,
            predict_affine=True,
            surface_classes=4,
            geometry_only=True,
            semantic_feature_dim=12,
            semantic_channels=8,
            semantic_attention_heads=2,
            semantic_layers=1,
            semantic_dropout=0.0,
            semantic_spatial_feature_dim=12,
            semantic_spatial_channels=8,
            semantic_text_prompt_count=3,
            semantic_text_prompt_feature_dim=12,
            semantic_text_prompt_channels=6,
            semantic_text_logit_scale=4.0,
            semantic_text_logit_bias=-1.0,
        )
        prompt_embeddings = torch.randn(3, 12)
        model.set_semantic_text_prompt_embeddings(prompt_embeddings)
        images = torch.rand(2, 4, 32, 32)
        semantic_features = {
            "raw_global": torch.rand(2, 12),
            "raw_spatial": torch.rand(2, 12, 7, 5),
        }

        outputs = model(
            images,
            view_ids=torch.tensor([0, 1]),
            semantic_features=semantic_features,
        )

        self.assertEqual(
            tuple(outputs["text_prompt_route_logits"].shape),
            (2, 3, 32, 32),
        )
        self.assertEqual(tuple(outputs["text_prompt_scores"].shape), (2, 3))
        self.assertFalse(
            model.semantic_text_prompt_fusion.prompt_embeddings.requires_grad
        )
        self.assertTrue(
            torch.allclose(
                model.semantic_text_prompt_fusion.prompt_embeddings.norm(
                    dim=-1
                ),
                torch.ones(3),
            )
        )
        outputs["layer"].mean().backward()
        gradient = (
            model.semantic_text_prompt_fusion.route_projection[-1].weight.grad
        )
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_text_prompt_route_has_independent_deep_supervision(self):
        model = DenseUVParserNet(
            base_channels=8,
            view_classes=2,
            geometry_only=True,
            semantic_feature_dim=12,
            semantic_channels=8,
            semantic_attention_heads=2,
            semantic_spatial_feature_dim=12,
            semantic_spatial_channels=8,
            semantic_text_prompt_count=3,
            semantic_text_prompt_feature_dim=12,
            semantic_text_prompt_channels=6,
        )
        model.set_semantic_text_prompt_embeddings(torch.randn(3, 12))
        outputs = model(
            torch.rand(2, 4, 16, 16),
            view_ids=torch.tensor([0, 1]),
            semantic_features={
                "raw_global": torch.rand(2, 12),
                "raw_spatial": torch.rand(2, 12, 5, 5),
            },
        )
        targets = {
            "foreground": torch.ones(2, 1, 16, 16),
            "route_role": torch.zeros(2, 16, 16, dtype=torch.long),
            "layer": torch.zeros(2, 16, 16, dtype=torch.long),
            "part": torch.zeros(2, 16, 16, dtype=torch.long),
            "face": torch.zeros(2, 16, 16, dtype=torch.long),
            "surface": torch.zeros(2, 16, 16, dtype=torch.long),
            "uv": torch.zeros(2, 2, 16, 16),
            "affine": torch.zeros(2, 3),
        }
        losses = DenseUVParserLoss(
            lambda_text_prompt_route=0.25,
            use_uv=False,
        )(outputs, targets)
        losses["loss_text_prompt_route_weighted"].backward()

        self.assertGreater(
            float(losses["loss_text_prompt_route"].detach()), 0.0
        )
        self.assertIsNotNone(
            model.semantic_text_prompt_fusion.route_projection[-1].weight.grad
        )
        self.assertGreater(
            float(
                model.semantic_text_prompt_fusion.route_projection[-1]
                .weight.grad.abs().sum()
            ),
            0.0,
        )

    def test_head_outer_structure_targets_and_auxiliary_gradients(self):
        target_uv = torch.zeros(1, 4, 64, 64)
        # Head outer front starts at (40, 8); form one connected component.
        target_uv[0, 3, 9:11, 41:44] = 1.0
        targets = build_head_outer_face_targets(target_uv)
        self.assertEqual(tuple(targets["occupancy"].shape), (1, 6, 8, 8))
        self.assertEqual(float(targets["presence"][0, 0]), 1.0)
        self.assertEqual(float(targets["presence"][0, 1:].sum()), 0.0)
        head_edges = build_head_outer_face_graph()
        self.assertTrue(
            ((head_edges[0] // 64) != (head_edges[1] // 64)).any()
        )

        model = DenseUVParserNet(
            base_channels=8,
            view_classes=2,
            geometry_only=True,
            semantic_feature_dim=12,
            semantic_channels=8,
            semantic_attention_heads=2,
            predict_head_outer_structure=True,
        )
        outputs = model(
            torch.rand(4, 4, 16, 16),
            view_ids=torch.tensor([0, 1, 0, 1]),
            semantic_features=torch.rand(1, 4, 12),
        )
        self.assertEqual(
            tuple(outputs["head_outer_face_occupancy_logits"].shape),
            (2, 6, 8, 8),
        )
        losses = head_outer_structure_losses(outputs, target_uv)
        total = (
            losses["loss_head_outer_occupancy_bce"]
            + losses["loss_head_outer_occupancy_dice"]
            + losses["loss_head_outer_presence"]
            + losses["loss_head_outer_coverage"]
            + losses["loss_head_outer_topology"]
            + losses["loss_head_outer_symmetry"]
        )
        total.backward()
        self.assertTrue(torch.isfinite(total))
        self.assertIsNotNone(
            model.head_outer_face_occupancy_head[-1].weight.grad
        )
        self.assertGreater(
            float(
                model.head_outer_face_occupancy_head[-1]
                .weight.grad.abs().sum()
            ),
            0.0,
        )
        self.assertIsNotNone(
            model.semantic_fusion.input_projection[1].weight.grad
        )

    def test_projected_head_outer_structure_uses_uv_features(self):
        for input_version, extra_channels in ((1, 2), (2, 9), (3, 9)):
            with self.subTest(input_version=input_version):
                model = DenseUVParserNet(
                    base_channels=8,
                    view_classes=2,
                    predict_affine=True,
                    surface_classes=4,
                    geometry_only=True,
                    semantic_feature_dim=12,
                    semantic_channels=8,
                    semantic_attention_heads=2,
                    predict_head_outer_structure=True,
                    head_outer_structure_mode="projected",
                    head_outer_projected_input_version=input_version,
                    outer_uv_feature_channels=4,
                    outer_uv_topology_channels=8,
                    outer_uv_topology_layers=1,
                )
                outputs = model(
                    torch.rand(4, 4, 16, 16),
                    view_ids=torch.tensor([0, 1, 0, 1]),
                    semantic_features=torch.rand(1, 4, 12),
                )
                features = F.interpolate(
                    outputs["head_outer_uv_features"],
                    size=(64, 64),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(2, 2, 4, 64, 64).mean(dim=1)
                atlas_features = torch.cat(
                    [
                        features,
                        features.new_zeros(
                            2, extra_channels, 64, 64
                        ),
                    ],
                    dim=1,
                )
                logits = model.predict_projected_head_outer_structure(
                    atlas_features,
                    outputs["head_outer_uv_global_context"],
                )
                self.assertEqual(tuple(logits.shape), (2, 6, 8, 8))
                self.assertEqual(
                    "head_outer_symmetry_logit" in outputs,
                    input_version >= 3,
                )
                logits.mean().backward()
                gradient = (
                    model.head_outer_projected_head.output[-1].weight.grad
                )
                self.assertIsNotNone(gradient)
                self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_head_outer_symmetry_penalizes_mirrored_positive_gap(self):
        target_uv = torch.zeros(1, 4, 64, 64)
        target_uv[0, 3, 8, 40] = 1.0
        target_uv[0, 3, 8, 47] = 1.0
        logits = torch.zeros(1, 6, 8, 8, requires_grad=True)
        with torch.no_grad():
            logits[0, 0, 0, 0] = 4.0
            logits[0, 0, 0, 7] = -4.0
        losses = head_outer_structure_losses(
            {
                "head_outer_face_occupancy_logits": logits,
                "head_outer_face_presence_logits": torch.zeros(1, 6),
                "head_outer_face_coverage": torch.zeros(1, 6),
            },
            target_uv,
        )
        self.assertGreater(
            float(losses["loss_head_outer_symmetry"].detach()), 0.0
        )

    def test_head_outer_symmetry_score_and_sparse_recall_are_supervised(self):
        target_uv = torch.zeros(1, 4, 64, 64)
        target_uv[0, 3, 8, 40] = 1.0
        target_uv[0, 3, 8, 47] = 1.0
        occupancy_logits = torch.full(
            (1, 6, 8, 8), -2.0, requires_grad=True
        )
        symmetry_logit = torch.zeros(1, requires_grad=True)
        losses = head_outer_structure_losses(
            {
                "head_outer_face_occupancy_logits": occupancy_logits,
                "head_outer_face_presence_logits": torch.zeros(1, 6),
                "head_outer_face_coverage": torch.zeros(1, 6),
                "head_outer_symmetry_logit": symmetry_logit,
            },
            target_uv,
        )
        total = (
            losses["loss_head_outer_component_hard_recall"]
            + losses["loss_head_outer_sparse_recall"]
            + losses["loss_head_outer_symmetry_score"]
        )
        total.backward()
        self.assertGreater(float(total.detach()), 0.0)
        self.assertIsNotNone(occupancy_logits.grad)
        self.assertIsNotNone(symmetry_logit.grad)
        self.assertEqual(
            float(losses["head_outer_symmetry_target"].detach()), 1.0
        )

    def test_v2_head_completion_fills_only_anchored_candidates(self):
        uv = torch.zeros(4, 64, 64)
        uv[:3, 8, 40:42] = torch.tensor([0.8, 0.1, 0.1]).view(3, 1)
        uv[3, 8, 40:42] = 1.0
        probability = torch.zeros(64, 64)
        probability[8, 40:43] = 0.95
        probability[9, 46] = 0.99

        repaired, stats = simple_symmetry_nearest_inpaint(
            uv,
            head_outer_probability=probability,
            head_outer_threshold=0.65,
            head_outer_min_component_seeds=2,
        )

        self.assertEqual(float(repaired[3, 8, 42]), 1.0)
        self.assertTrue(
            torch.allclose(repaired[:3, 8, 42], uv[:3, 8, 41])
        )
        self.assertEqual(float(repaired[3, 9, 46]), 0.0)
        self.assertGreaterEqual(
            stats["head_outer_topology_filled_texels"], 1
        )

    def test_v3_symmetric_head_completion_requires_global_confidence(self):
        uv = torch.zeros(4, 64, 64)
        uv[:3, 8, 40:42] = torch.tensor([0.8, 0.1, 0.1]).view(3, 1)
        uv[3, 8, 40:42] = 1.0
        probability = torch.zeros(64, 64)
        probability[8, 46:48] = 0.25

        repaired, stats = simple_symmetry_nearest_inpaint(
            uv,
            head_outer_probability=probability,
            head_outer_threshold=0.65,
            head_outer_min_component_seeds=2,
            head_outer_symmetry_probability=0.95,
            head_outer_symmetry_threshold=0.80,
            head_outer_symmetry_candidate_threshold=0.20,
        )
        self.assertEqual(float(repaired[3, 8, 46:48].sum()), 2.0)
        self.assertEqual(stats["head_outer_symmetry_filled_texels"], 2)

        rejected, rejected_stats = simple_symmetry_nearest_inpaint(
            uv,
            head_outer_probability=probability,
            head_outer_threshold=0.65,
            head_outer_min_component_seeds=2,
            head_outer_symmetry_probability=0.50,
            head_outer_symmetry_threshold=0.80,
            head_outer_symmetry_candidate_threshold=0.20,
        )
        self.assertEqual(float(rejected[3, 8, 46:48].sum()), 0.0)
        self.assertEqual(
            rejected_stats["head_outer_symmetry_filled_texels"], 0
        )

    def test_head_outer_route_connectivity_penalizes_brim_gap(self):
        edge = build_head_outer_face_graph()[:, 0]
        target = torch.zeros(1, 6 * 8 * 8, dtype=torch.bool)
        visible = torch.zeros_like(target)
        target[0, edge] = True
        visible[0, edge] = True
        logits = torch.full((1, 6 * 8 * 8), 2.2, requires_grad=True)
        with torch.no_grad():
            logits[0, edge[1]] = -2.2
        probability = torch.sigmoid(logits)

        broken = _head_outer_route_connectivity_terms(
            probability, target, visible
        )
        connected = _head_outer_route_connectivity_terms(
            torch.full_like(probability.detach(), 0.90),
            target,
            visible,
        )
        broken["loss_head_outer_route_connectivity"].backward()

        self.assertGreater(
            float(
                broken["loss_head_outer_route_connectivity"].detach()
            ),
            float(
                connected["loss_head_outer_route_connectivity"].detach()
            ),
        )
        self.assertLess(float(logits.grad[0, edge[1]]), 0.0)
        self.assertGreater(
            float(broken["count_head_outer_route_positive_edges"]),
            0.0,
        )

    def test_siglip_text_branch_preserves_seeded_base_initialization(self):
        common_arguments = {
            "base_channels": 8,
            "view_classes": 2,
            "geometry_only": True,
            "semantic_feature_dim": 12,
            "semantic_channels": 8,
            "semantic_attention_heads": 2,
            "semantic_spatial_feature_dim": 12,
            "semantic_spatial_channels": 8,
        }
        torch.manual_seed(31415)
        baseline = DenseUVParserNet(**common_arguments)
        torch.manual_seed(31415)
        prompted = DenseUVParserNet(
            **common_arguments,
            semantic_text_prompt_count=3,
            semantic_text_prompt_feature_dim=12,
            semantic_text_prompt_channels=6,
        )

        prompted_state = prompted.state_dict()
        for name, value in baseline.state_dict().items():
            self.assertTrue(
                torch.equal(value, prompted_state[name]),
                msg=f"Text branch changed seeded base parameter {name}.",
            )

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

    def test_balanced_occupancy_bce_does_not_prefer_all_transparent(self):
        logits = torch.zeros(1, 1, 64, 64, requires_grad=True)
        target_uv = torch.zeros(1, 4, 64, 64)
        _, outer_masks = build_part_layer_masks()
        occupied = outer_masks[:, 0].bool().any(dim=0)
        y, x = occupied.nonzero()[0]
        target_uv[0, 3, y, x] = 1.0

        losses = outer_uv_occupancy_losses(
            logits,
            target_uv,
            outer_masks,
            positive_balance=0.60,
            hard_positive_fraction=0.0,
            hard_negative_fraction=0.0,
        )
        losses["loss_outer_uv_occupancy_bce"].backward()

        valid_gradient = logits.grad[0, 0][occupied]
        self.assertLess(float(valid_gradient.sum()), 0.0)
        self.assertLess(float(logits.grad[0, 0, y, x]), 0.0)

    def test_occupancy_loss_ignores_unseen_outer_texels(self):
        logits = torch.zeros(1, 1, 64, 64, requires_grad=True)
        target_uv = torch.zeros(1, 4, 64, 64)
        _, outer_masks = build_part_layer_masks()
        occupied = outer_masks[:, 0].bool().any(dim=0)
        first, second = occupied.nonzero()[:2]
        target_uv[0, 3, first[0], first[1]] = 1.0
        support = torch.zeros_like(logits, dtype=torch.bool)
        support[0, 0, first[0], first[1]] = True

        losses = outer_uv_occupancy_losses(
            logits,
            target_uv,
            outer_masks,
            support_mask=support,
            hard_positive_fraction=0.0,
            hard_negative_fraction=0.0,
        )
        losses["loss_outer_uv_occupancy_bce"].backward()

        self.assertLess(float(logits.grad[0, 0, first[0], first[1]]), 0.0)
        self.assertEqual(float(logits.grad[0, 0, second[0], second[1]]), 0.0)
        self.assertEqual(
            int(losses["count_outer_occupancy_supervised_texels"]),
            1,
        )

    def test_outer_uv_occupancy_penalizes_coherent_false_component(self):
        flat_indices, edge_index = build_outer_uv_graph()
        source, target = edge_index[:, 0]
        false_flats = flat_indices[torch.tensor([source, target])]
        logits = torch.full((1, 1, 64, 64), -8.0, requires_grad=True)
        with torch.no_grad():
            logits.flatten()[false_flats] = 4.0
        target_uv = torch.zeros(1, 4, 64, 64)
        _, outer_masks = build_part_layer_masks()

        losses = outer_uv_occupancy_losses(
            logits,
            target_uv,
            outer_masks,
            hard_negative_fraction=0.10,
        )
        false_loss = (
            losses["loss_outer_uv_occupancy_hard_negative"]
            + losses["loss_outer_negative_topology"]
            + losses["loss_outer_component_false_positive"]
        )
        false_loss.backward()

        self.assertGreater(float(false_loss.detach()), 1.0)
        self.assertGreater(
            float(losses["loss_outer_component_false_positive"].detach()),
            0.9,
        )
        self.assertGreater(
            float(losses["loss_outer_negative_topology"].detach()),
            0.0,
        )
        self.assertGreater(float(logits.grad.flatten()[false_flats].min()), 0.0)

    def test_outer_uv_occupancy_head_does_not_shift_parser_trunk(self):
        model = self.build_model().train()
        outputs = model(
            torch.rand(2, 4, 32, 32),
            view_ids=torch.tensor([0, 1]),
            semantic_features=torch.rand(1, 2, 16),
        )
        occupancy_logits = self.projected_occupancy(model, outputs)

        occupancy_logits.mean().backward()

        self.assertIsNotNone(
            model.outer_uv_occupancy_head.output[-1].weight.grad
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
        conditioning = torch.zeros(1, 10, 64, 64)
        conditioning[0, 0:4, 8, 8] = torch.tensor([1.0, 0.0, 0.0, 1.0])
        conditioning[0, 4, 8, 8] = 1.0
        conditioning[0, 5:9, 8, 40] = torch.tensor(
            [0.0, 128.0 / 255.0, 1.0, 1.0]
        )
        conditioning[0, 9, 8, 40] = 1.0
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "simple.png"
            save_simple_inpaint_uv(conditioning, output)
            image = Image.open(output).convert("RGBA")

            self.assertEqual(image.size, (64, 64))
            self.assertEqual(image.getpixel((8, 8)), (255, 0, 0, 255))
            self.assertEqual(image.getpixel((20, 20)), (0, 0, 0, 0))
            self.assertEqual(image.getpixel((40, 8)), (0, 128, 255, 255))
            self.assertEqual(image.getpixel((63, 0)), (0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
