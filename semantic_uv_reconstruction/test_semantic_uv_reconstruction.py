import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

from SkingToolkit.semantic_uv_reconstruction.semantic_dataset import SemanticUVPairDataset
from SkingToolkit.semantic_uv_reconstruction.semantic_losses import (
    SemanticUVReconstructionLoss,
    build_part_layer_masks,
    build_semantic_attribute_targets,
)
from SkingToolkit.semantic_uv_reconstruction.semantic_backbone import (
    SigLIP2VisionBackbone,
    TIPSv2VisionBackbone,
)
from SkingToolkit.semantic_uv_reconstruction.semantic_model import SemanticUVReconstructor
from SkingToolkit.semantic_uv_reconstruction.train_semantic_uv_reconstruction import (
    build_arg_parser,
)


class FakeOpenSemanticBackbone(nn.Module):
    """Small differentiable stand-in; CI never downloads SigLIP2 weights."""

    raw_feature_dim = 6

    def __init__(self, token_channels):
        super().__init__()
        self.token_projection = nn.Linear(3, token_channels)
        self.global_projection = nn.Linear(6, token_channels)

    def forward(self, images):
        patch_rgb = F.adaptive_avg_pool2d(images, (2, 2)).flatten(2).transpose(1, 2)
        mean = images.mean(dim=(2, 3))
        std = images.var(dim=(2, 3), unbiased=False).add(1e-6).sqrt()
        raw_global = torch.cat([mean, std], dim=1)
        return {
            "tokens": self.token_projection(patch_rgb),
            "token_mask": torch.ones(
                images.shape[0], 4, dtype=torch.bool, device=images.device
            ),
            "global": self.global_projection(raw_global),
            "raw_global": raw_global,
        }

    def encode_global(self, images):
        outputs = self.forward(images)
        return {name: outputs[name] for name in ("global", "raw_global")}

    def project_global(self, raw_global):
        return self.global_projection(raw_global)


class FakeRenderer:
    def forward_view(self, uv, view):
        del view
        return F.interpolate(uv, size=(64, 32), mode="bilinear", align_corners=False)


class FakeHuggingFaceVisionTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(image_size=32, patch_size=8, hidden_size=12)
        self.projection = nn.Linear(3, 12)

    def forward(self, pixel_values):
        patches = F.avg_pool2d(pixel_values, kernel_size=8, stride=8)
        patches = patches.flatten(2).transpose(1, 2)
        tokens = self.projection(patches)
        return SimpleNamespace(last_hidden_state=tokens, pooler_output=tokens.mean(dim=1))


class SigLIP2AdapterTest(unittest.TestCase):
    def test_lazy_hugging_face_adapter_preserves_input_gradients(self):
        vision_tower = FakeHuggingFaceVisionTower()

        class FakeAutoImageProcessor:
            @staticmethod
            def from_pretrained(model_name, local_files_only=False, use_fast=True):
                del model_name, local_files_only
                if use_fast:
                    raise AssertionError("The adapter must preserve slow processor metadata.")
                return SimpleNamespace(
                    image_mean=(0.5, 0.5, 0.5),
                    image_std=(0.5, 0.5, 0.5),
                    size={"height": 32, "width": 32},
                )

        class FakeAutoModel:
            @staticmethod
            def from_pretrained(model_name, local_files_only=False):
                del model_name, local_files_only
                return SimpleNamespace(
                    vision_model=vision_tower,
                    config=SimpleNamespace(vision_config=vision_tower.config),
                )

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoImageProcessor = FakeAutoImageProcessor
        fake_transformers.AutoModel = FakeAutoModel
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            backbone = SigLIP2VisionBackbone("fake-fixres", token_channels=8)
        images = torch.rand(2, 3, 20, 40, requires_grad=True)
        outputs = backbone(images)
        self.assertEqual(outputs["tokens"].shape, (2, 8, 8))
        self.assertEqual(outputs["token_mask"].shape, (2, 8))
        self.assertTrue(outputs["token_mask"].all())
        self.assertTrue(outputs["tokens_compact"])
        self.assertEqual(outputs["raw_global"].shape, (2, 12))
        self.assertFalse(any(parameter.requires_grad for parameter in backbone.vision_model.parameters()))
        outputs["raw_global"].sum().backward()
        self.assertIsNotNone(images.grad)
        self.assertGreater(float(images.grad.abs().sum()), 0.0)


class TIPSv2AdapterTest(unittest.TestCase):
    def test_adapter_returns_letterbox_cropped_spatial_features(self):
        class FakeVisionBackbone(nn.Module):
            channels = [12]

            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(image_size=32, hidden_size=12)
                self.projection = nn.Conv2d(3, 12, kernel_size=1)

            @staticmethod
            def from_pretrained(
                model_name,
                local_files_only=False,
                out_indices=None,
            ):
                del model_name, local_files_only
                if out_indices != [-1]:
                    raise AssertionError("The adapter must request the final feature map.")
                return FakeVisionBackbone()

            def forward(self, pixel_values):
                pooled = F.adaptive_avg_pool2d(pixel_values, (4, 4))
                return SimpleNamespace(feature_maps=(self.projection(pooled),))

        class FakeAutoImageProcessor:
            @staticmethod
            def from_pretrained(model_name, local_files_only=False, use_fast=True):
                del model_name, local_files_only
                if use_fast:
                    raise AssertionError("The adapter must preserve processor metadata.")
                return SimpleNamespace(
                    image_mean=(0.5, 0.5, 0.5),
                    image_std=(0.5, 0.5, 0.5),
                    size={"height": 32, "width": 32},
                )

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoImageProcessor = FakeAutoImageProcessor
        fake_transformers.AutoBackbone = FakeVisionBackbone
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            backbone = TIPSv2VisionBackbone(
                "fake-tipsv2",
                inference_batch_size=1,
            )
        outputs = backbone.encode_dense(torch.rand(2, 3, 20, 40))

        self.assertEqual(tuple(outputs["raw_spatial"].shape), (2, 12, 2, 4))
        self.assertEqual(tuple(outputs["raw_global"].shape), (2, 12))
        self.assertFalse(
            any(parameter.requires_grad for parameter in backbone.vision_model.parameters())
        )


class SemanticUVModelTest(unittest.TestCase):
    def test_long_run_performance_defaults(self):
        args = build_arg_parser().parse_args([])
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.siglip_render_every, 4)
        self.assertEqual(args.siglip_render_warmup_epochs, 2)
        self.assertEqual(args.rgb_warmup_epochs, 2)
        self.assertEqual(args.rgb_warmup_multiplier, 2.0)
        self.assertEqual(args.memory_latents, 256)
        self.assertEqual(args.log_every, 50)
        self.assertTrue(args.fused_optimizer)
        self.assertTrue(args.compile)
        self.assertEqual(args.compile_mode, "max-autotune-no-cudagraphs")

    def test_dual_view_model_outputs_valid_skin_structure(self):
        model = SemanticUVReconstructor(
            view_count=2,
            base_channels=8,
            token_channels=32,
            query_size=8,
            attention_heads=4,
            attention_layers=1,
        )
        images = torch.rand(2, 2, 3, 64, 32)
        outputs = model(images)

        self.assertEqual(outputs["uv"].shape, (2, 4, 64, 64))
        self.assertEqual(outputs["semantic_uv_logits"].shape, (2, 13, 64, 64))
        self.assertEqual(outputs["outer_presence_logits"].shape, (2, 6))
        self.assertEqual(outputs["part_colors"].shape, (2, 12, 3))
        self.assertEqual(model.architecture_version, 3)
        self.assertEqual(model.memory_latent_count, 256)
        self.assertTrue(any(isinstance(layer, nn.PixelShuffle) for layer in model.decoder.modules()))
        base = model.base_mask.bool().expand(2, -1, -1, -1)
        invalid = ~(model.base_mask.bool() | model.decor_mask.bool())
        invalid = invalid.expand(2, -1, -1, -1)
        self.assertTrue(torch.equal(outputs["uv"][:, 3:4][base], torch.ones_like(outputs["uv"][:, 3:4][base])))
        self.assertTrue(torch.equal(outputs["uv"][:, 3:4][invalid], torch.zeros_like(outputs["uv"][:, 3:4][invalid])))

    def test_semantic_attribute_targets_capture_outer_head_color(self):
        inner_masks, outer_masks = build_part_layer_masks()
        uv = torch.zeros(1, 4, 64, 64)
        uv[:, 3:4] = inner_masks.sum(dim=0).clamp(0.0, 1.0)
        outer_head = outer_masks[0:1]
        uv[:, 0:1] = torch.maximum(uv[:, 0:1], outer_head)
        uv[:, 3:4] = torch.maximum(uv[:, 3:4], outer_head)

        targets = build_semantic_attribute_targets(uv, inner_masks, outer_masks)
        self.assertEqual(float(targets["outer_presence"][0, 0]), 1.0)
        self.assertEqual(float(targets["outer_presence"][0, 1:].sum()), 0.0)
        self.assertAlmostEqual(float(targets["outer_coverage"][0, 0]), 1.0, places=5)
        self.assertGreater(float(targets["part_colors"][0, 6, 0]), 0.99)
        self.assertLess(float(targets["part_colors"][0, 6, 1:].abs().sum()), 1e-6)

    def test_reconstruction_loss_backpropagates_without_render_branch(self):
        model = SemanticUVReconstructor(
            view_count=2,
            base_channels=8,
            token_channels=32,
            query_size=8,
            attention_heads=4,
            attention_layers=1,
        )
        criterion = SemanticUVReconstructionLoss(
            lambda_render_rgb=0.0,
            lambda_render_alpha=0.0,
            lambda_siglip_render=0.0,
        )
        outputs = model(torch.rand(1, 2, 3, 64, 32))
        target = outputs["uv"].detach().clone()
        target[:, :3] = 0.0
        metrics = criterion(outputs, target)
        metrics["loss_total"].backward()
        self.assertIsNotNone(model.rgb_head.weight.grad)
        self.assertGreater(float(model.rgb_head.weight.grad.abs().sum()), 0.0)
        detail_grad = model.detail_projection[0].weight.grad
        self.assertIsNotNone(detail_grad)
        self.assertGreater(float(detail_grad.abs().sum()), 0.0)
        self.assertIn("loss_uv_edge", metrics)
        self.assertIn("rgb_mae_255", metrics)
        self.assertIsNotNone(model.memory_latents.grad)
        self.assertGreater(float(model.memory_latents.grad.abs().sum()), 0.0)

    def test_rgb_warmup_scales_only_uv_detail_objective(self):
        model = SemanticUVReconstructor(
            view_count=2,
            base_channels=8,
            token_channels=32,
            query_size=8,
            attention_heads=4,
            attention_layers=1,
        )
        criterion = SemanticUVReconstructionLoss(
            lambda_uv_rgb=2.0,
            lambda_uv_edge=1.0,
            lambda_outer_alpha=0.0,
            lambda_outer_dice=0.0,
            lambda_semantic_uv=0.0,
            lambda_semantic_presence=0.0,
            lambda_semantic_coverage=0.0,
            lambda_semantic_color=0.0,
            lambda_render_rgb=0.0,
            lambda_render_alpha=0.0,
            lambda_siglip_render=0.0,
        )
        outputs = model(torch.rand(1, 2, 3, 64, 32))
        target = outputs["uv"].detach().clone()
        target[:, :3] = 1.0 - target[:, :3]
        normal = criterion(outputs, target, uv_detail_scale=1.0)
        warmup = criterion(outputs, target, uv_detail_scale=2.0)
        self.assertTrue(
            torch.allclose(warmup["loss_total"], normal["loss_total"] * 2.0)
        )
        self.assertEqual(float(warmup["uv_detail_scale"]), 2.0)

    def test_open_semantic_fusion_and_render_cycle_are_differentiable(self):
        backbone = FakeOpenSemanticBackbone(token_channels=32)
        model = SemanticUVReconstructor(
            view_count=2,
            base_channels=8,
            token_channels=32,
            query_size=8,
            attention_heads=4,
            attention_layers=1,
            open_semantic_backbone=backbone,
        )
        criterion = SemanticUVReconstructionLoss(
            lambda_uv_rgb=0.0,
            lambda_uv_edge=0.0,
            lambda_outer_alpha=0.0,
            lambda_outer_dice=0.0,
            lambda_semantic_uv=0.0,
            lambda_semantic_presence=0.0,
            lambda_semantic_coverage=0.0,
            lambda_semantic_color=0.0,
            lambda_render_rgb=0.0,
            lambda_render_alpha=0.0,
            lambda_siglip_render=1.0,
        )
        renderer = FakeRenderer()
        target_uv = torch.rand(1, 4, 64, 64)
        gt_renders = torch.stack(
            [renderer.forward_view(target_uv, view) for view in ("front", "back")], dim=1
        )
        outputs = model(gt_renders[:, :, :3])
        self.assertEqual(outputs["open_semantic_embedding"].shape, (1, 6))
        skipped_metrics = criterion(
            outputs,
            target_uv,
            compute_siglip_render=False,
        )
        self.assertEqual(float(skipped_metrics["loss_siglip_render"]), 0.0)
        metrics = criterion(
            outputs,
            target_uv,
            gt_renders=gt_renders,
            renderer=renderer,
            views=("front", "back"),
            semantic_encoder=model.encode_open_semantics,
        )
        metrics["loss_total"].backward()
        self.assertGreaterEqual(float(metrics["loss_siglip_render"].detach()), 0.0)
        self.assertIsNotNone(model.rgb_head.weight.grad)
        self.assertGreater(float(model.rgb_head.weight.grad.abs().sum()), 0.0)

    def test_cached_global_semantics_skip_source_vision_encoding(self):
        backbone = FakeOpenSemanticBackbone(token_channels=32)
        model = SemanticUVReconstructor(
            view_count=2,
            base_channels=8,
            token_channels=32,
            query_size=8,
            attention_heads=4,
            attention_layers=1,
            open_semantic_backbone=backbone,
        )
        images = torch.rand(1, 2, 3, 64, 32)
        cached = torch.rand(1, 2, backbone.raw_feature_dim)
        outputs = model(images, open_semantic_raw=cached)
        expected = F.normalize(cached.mean(dim=1), dim=-1)
        self.assertTrue(torch.allclose(outputs["open_semantic_embedding"], expected))


class SemanticAnnotationTest(unittest.TestCase):
    def test_dataset_needs_no_concept_vocabulary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename in ("one.png", "two.png"):
                Image.new("RGBA", (64, 64), (10, 20, 30, 255)).save(root / filename)
            dataset = SemanticUVPairDataset(root)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(set(dataset[0]), {"uv", "path"})

    def test_dataset_reads_memory_mapped_siglip_globals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            skins = root / "skins"
            cache = root / "cache"
            skins.mkdir()
            cache.mkdir()
            Image.new("RGBA", (64, 64), (10, 20, 30, 255)).save(skins / "one.png")
            values = np.arange(12, dtype=np.float16).reshape(1, 2, 6)
            np.save(cache / "embeddings.npy", values)
            (cache / "metadata.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "filenames": ["one.png"],
                        "views": ["front", "back"],
                        "siglip_model": "fake",
                        "feature_dim": 6,
                    }
                ),
                encoding="utf-8",
            )
            dataset = SemanticUVPairDataset(
                skins,
                siglip_cache_dir=cache,
                siglip_cache_views=["front", "back"],
                siglip_cache_model="fake",
            )
            sample = dataset[0]
            self.assertEqual(sample["siglip_raw_global"].shape, (2, 6))
            self.assertTrue(
                torch.equal(sample["siglip_raw_global"], torch.from_numpy(values[0]).float())
            )


if __name__ == "__main__":
    unittest.main()
