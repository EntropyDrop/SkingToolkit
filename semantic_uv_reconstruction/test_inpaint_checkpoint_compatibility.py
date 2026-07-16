import io
import tempfile
import unittest
from pathlib import Path

import torch

from SkingToolkit.dense_uv_parser.infer import load_inpaint
from SkingToolkit.semantic_uv_reconstruction.model import UVInpaintingNet
from SkingToolkit.semantic_uv_reconstruction.topology_model import (
    TopologyAwareUVCompletionNet,
)
from SkingToolkit.semantic_uv_reconstruction.train import Logger


class InpaintCheckpointCompatibilityTest(unittest.TestCase):
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
