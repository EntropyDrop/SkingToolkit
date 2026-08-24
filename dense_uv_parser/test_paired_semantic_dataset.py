from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from SkingToolkit.dense_uv_parser.losses import (
    dense_semantic_outer_false_positive_loss,
)
from SkingToolkit.dense_uv_parser.skin_dataset import PairedRenderSkinDataset


def _write_pair(root: Path, name="sample"):
    render = np.zeros((32, 64, 3), dtype=np.uint8)
    render[:, :32] = (20, 40, 60)
    render[:, 32:] = (80, 100, 120)
    Image.fromarray(render, mode="RGB").save(root / f"{name}_edited.png")

    uv = np.zeros((64, 64, 4), dtype=np.uint8)
    uv[..., :3] = (12, 34, 56)
    uv[..., 3] = 255
    Image.fromarray(uv, mode="RGBA").save(root / f"{name}_result.png")


class PairedSemanticDatasetTest(unittest.TestCase):
    def test_loads_two_views_and_uv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_pair(root)
            dataset = PairedRenderSkinDataset(
                root,
                view_size=(16, 16),
            )

            sample = dataset[0]

            self.assertEqual(sample["uv"].shape, (4, 64, 64))
            self.assertEqual(sample["rendered"].shape, (2, 4, 16, 16))
            self.assertLess(
                sample["rendered"][0, :3].mean().item(),
                sample["rendered"][1, :3].mean().item(),
            )
            self.assertTrue(sample["path"].endswith("sample_edited.png"))
            self.assertTrue(sample["uv_path"].endswith("sample_result.png"))

    def test_skips_non_skin_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            render = Image.new("RGB", (64, 32), (0, 0, 0))
            render.save(root / "bad_edited.png")
            Image.new("RGBA", (128, 128), (0, 0, 0, 0)).save(
                root / "bad_result.png"
            )

            with self.assertRaisesRegex(
                ValueError, r"No \*_edited / 64x64 \*_result pairs"
            ):
                PairedRenderSkinDataset(root)

    def test_manifest_selects_only_quality_gated_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_pair(root, "keep")
            _write_pair(root, "drop")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "data_dir": str(root.resolve()),
                        "pairs": [
                            {
                                "edited": "keep_edited.png",
                                "result": "keep_result.png",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            dataset = PairedRenderSkinDataset(
                root,
                manifest_path=manifest_path,
            )

            self.assertEqual(len(dataset), 1)
            self.assertTrue(dataset[0]["path"].endswith("keep_edited.png"))

    def test_versioned_result_suffix_never_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_pair(root, "sample")
            with self.assertRaisesRegex(ValueError, "_v94_result"):
                PairedRenderSkinDataset(
                    root,
                    result_suffix="_v94_result",
                )

            (root / "sample_result.png").rename(
                root / "sample_v94_result.png"
            )
            dataset = PairedRenderSkinDataset(
                root,
                result_suffix="_v94_result",
            )
            self.assertTrue(
                dataset[0]["uv_path"].endswith("sample_v94_result.png")
            )

    def test_hard_negative_loss_penalizes_sparse_inner_to_outer_error(self):
        targets = torch.full((1, 8, 8), 3, dtype=torch.long)
        correct = torch.zeros(1, 5, 8, 8)
        correct[:, 3] = 5.0
        incorrect = correct.clone()
        incorrect[:, 2, 3, 4] = 10.0

        correct_loss = dense_semantic_outer_false_positive_loss(
            correct, targets
        )
        incorrect_loss = dense_semantic_outer_false_positive_loss(
            incorrect, targets
        )

        self.assertGreater(incorrect_loss.item(), correct_loss.item())


if __name__ == "__main__":
    unittest.main()
