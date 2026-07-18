import io
import tempfile
import unittest
from pathlib import Path

import torch

from SkingToolkit.dense_uv_parser.infer import (
    generate_topology_completion,
    load_inpaint,
    lock_completed_parser_evidence,
    propagate_completed_unknown_colors,
)
from SkingToolkit.semantic_uv_reconstruction.model import UVInpaintingNet
from SkingToolkit.semantic_uv_reconstruction.topology_model import (
    TopologyAwareUVCompletionNet,
)
from SkingToolkit.semantic_uv_reconstruction.topology import build_uv_topology
from SkingToolkit.semantic_uv_reconstruction.train import Logger


class InpaintCheckpointCompatibilityTest(unittest.TestCase):
    def test_low_confidence_context_is_copied_only_at_the_same_uv(self):
        topology = build_uv_topology()
        indices = (topology.surface.reshape(-1) == 0).nonzero(
            as_tuple=False
        ).flatten()[:2]
        context_y, context_x = divmod(int(indices[0]), 64)
        other_y, other_x = divmod(int(indices[1]), 64)
        context_color = torch.tensor([0.55, 0.08, 0.12, 1.0])
        generated_color = torch.tensor([0.0, 1.0, 1.0, 1.0])
        conditioning = torch.zeros(1, 12, 64, 64)
        conditioning[0, 0:4, context_y, context_x] = context_color
        conditioning[0, 4, context_y, context_x] = 0.0
        conditioning[0, 5, context_y, context_x] = 0.4
        completed = torch.zeros(1, 4, 64, 64)
        completed[0, :, context_y, context_x] = generated_color
        completed[0, :, other_y, other_x] = generated_color

        propagated, stats = propagate_completed_unknown_colors(
            completed,
            conditioning,
            min_confidence=0.75,
            context_min_confidence=0.35,
        )

        self.assertTrue(
            torch.equal(propagated[0, :, context_y, context_x], context_color)
        )
        self.assertTrue(
            torch.equal(propagated[0, :, other_y, other_x], generated_color)
        )
        self.assertEqual(stats["available_context_texels"], 1)
        self.assertEqual(stats["palette_context_texels"], 0)
        self.assertEqual(stats["direct_context_texels"], 1)
        self.assertEqual(stats["topology_color_propagated_texels"], 0)
        self.assertEqual(stats["uncolored_generated_texels"], 1)

    def test_rejected_context_guides_color_without_becoming_locked_evidence(self):
        topology = build_uv_topology()
        indices = (topology.surface.reshape(-1) == 0).nonzero(
            as_tuple=False
        ).flatten()[:3]
        context_color = torch.tensor([0.55, 0.08, 0.12, 1.0])
        conditioning = torch.zeros(1, 12, 64, 64)
        for source_index in indices[:2]:
            y, x = divmod(int(source_index), 64)
            conditioning[0, 0:4, y, x] = context_color
            conditioning[0, 4, y, x] = 0.0
            conditioning[0, 5, y, x] = 0.9
        target_y, target_x = divmod(int(indices[2]), 64)
        completed = torch.zeros(1, 4, 64, 64)
        completed[0, :, target_y, target_x] = torch.tensor(
            [0.0, 1.0, 1.0, 1.0]
        )

        propagated, stats = propagate_completed_unknown_colors(
            completed,
            conditioning,
            min_confidence=0.75,
        )
        locked, lock_stats = lock_completed_parser_evidence(
            propagated,
            conditioning,
            confidence_threshold=0.0,
        )

        self.assertTrue(
            torch.equal(propagated[0, :, target_y, target_x], context_color)
        )
        self.assertTrue(torch.equal(locked, propagated))
        self.assertEqual(stats["topology_color_propagated_texels"], 1)
        self.assertEqual(lock_stats["locked_evidence_texels"], 0)

    def test_final_color_propagation_ignores_single_observed_outlier(self):
        topology = build_uv_topology()
        indices = (topology.surface.reshape(-1) == 0).nonzero(
            as_tuple=False
        ).flatten()[:4]
        stable_color = torch.tensor([0.25, 0.10, 0.55, 1.0])
        outlier_color = torch.tensor([1.0, 0.0, 1.0, 1.0])
        conditioning = torch.zeros(1, 12, 64, 64)
        for source_index in indices[:2]:
            y, x = divmod(int(source_index), 64)
            conditioning[0, 0:4, y, x] = stable_color
            conditioning[0, 4, y, x] = 1.0
            conditioning[0, 5, y, x] = 1.0
        outlier_y, outlier_x = divmod(int(indices[2]), 64)
        conditioning[0, 0:4, outlier_y, outlier_x] = outlier_color
        conditioning[0, 4, outlier_y, outlier_x] = 1.0
        conditioning[0, 5, outlier_y, outlier_x] = 1.0
        target_y, target_x = divmod(int(indices[3]), 64)
        completed = torch.zeros(1, 4, 64, 64)
        completed[0, :, target_y, target_x] = outlier_color

        propagated, stats = propagate_completed_unknown_colors(
            completed,
            conditioning,
            min_confidence=0.75,
        )

        self.assertTrue(
            torch.equal(propagated[0, :, target_y, target_x], stable_color)
        )
        self.assertEqual(stats["generated_opaque_texels"], 1)
        self.assertEqual(stats["topology_color_propagated_texels"], 1)
        self.assertEqual(stats["uncolored_generated_texels"], 0)

    def test_post_generation_lock_restores_all_routed_evidence(self):
        conditioning = torch.zeros(1, 12, 64, 64)
        conditioning[0, 0:4, 8, 8] = torch.tensor([0.2, 0.4, 0.6, 1.0])
        conditioning[0, 4, 8, 8] = 1.0
        conditioning[0, 5, 8, 8] = 0.55
        conditioning[0, 6:10, 16, 16] = torch.tensor([0.9, 0.3, 0.1, 1.0])
        conditioning[0, 10, 16, 16] = 1.0
        conditioning[0, 11, 16, 16] = 0.60
        completed = torch.ones(1, 4, 64, 64)

        locked, stats = lock_completed_parser_evidence(
            completed,
            conditioning,
            confidence_threshold=0.0,
        )

        self.assertTrue(
            torch.equal(locked[0, :, 8, 8], conditioning[0, 0:4, 8, 8])
        )
        self.assertTrue(
            torch.equal(locked[0, :, 16, 16], conditioning[0, 6:10, 16, 16])
        )
        self.assertEqual(stats["locked_evidence_texels"], 2)
        self.assertEqual(stats["model_overwrote_locked_texels"], 2)

    def test_legacy_topology_generate_signature_still_runs(self):
        class LegacyGenerator:
            def generate(self, conditioning, steps, temperature, seed):
                self.arguments = (steps, temperature, seed)
                return conditioning[:, :4]

        generator = LegacyGenerator()
        conditioning = torch.zeros(1, 12, 64, 64)
        result = generate_topology_completion(
            generator,
            conditioning,
            steps=4,
            temperature=0.0,
            seed=1234,
            palette_snap=True,
        )
        self.assertEqual(tuple(result.shape), (1, 4, 64, 64))
        self.assertEqual(generator.arguments, (4, 0.0, 1234))

    def test_training_logger_preserves_terminal_capabilities(self):
        stream = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            logger = Logger(Path(directory) / "train.log", stream)
            try:
                self.assertEqual(logger.isatty(), stream.isatty())
                self.assertEqual(logger.encoding, "utf-8")
                logger.write("loading model\n")
                logger.flush()
                self.assertEqual(stream.getvalue(), "loading model\n")
            finally:
                logger.log.close()

    def test_loads_topology_checkpoint_from_embedded_model_config(self):
        source = TopologyAwareUVCompletionNet(
            input_channels=12,
            hidden_channels=32,
            layers=1,
            attention_heads=4,
            dropout=0.0,
            hard_lock_threshold=0.77,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "topology.pt"
            torch.save(
                {
                    "model": source.state_dict(),
                    "model_config": source.checkpoint_config(),
                    "args": {},
                },
                checkpoint_path,
            )
            loaded, args = load_inpaint(checkpoint_path, torch.device("cpu"))

        self.assertIsInstance(loaded, TopologyAwareUVCompletionNet)
        self.assertEqual(args["completion_model"], "topology_maskgit")
        self.assertEqual(loaded.hidden_channels, 32)
        self.assertEqual(loaded.layers, 1)
        self.assertEqual(loaded.input_channels, 12)
        self.assertAlmostEqual(loaded.hard_lock_threshold, 0.77)

    def test_loads_legacy_unet_checkpoint_without_model_config(self):
        source = UVInpaintingNet(input_channels=10, base_channels=8)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "legacy_unet.pt"
            torch.save(
                {
                    "model": source.state_dict(),
                    "input_channels": 10,
                    "args": {"base_channels": 8, "preserve_known": True},
                },
                checkpoint_path,
            )
            loaded, args = load_inpaint(checkpoint_path, torch.device("cpu"))

        self.assertIsInstance(loaded, UVInpaintingNet)
        self.assertEqual(args["completion_model"], "unet")


if __name__ == "__main__":
    unittest.main()
