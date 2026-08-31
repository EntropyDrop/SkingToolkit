from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from SkingToolkit.dense_uv_parser.losses import (
    dense_semantic_outer_false_positive_loss,
    dense_semantic_outer_union_terms,
    hierarchical_dense_semantic_terms,
    head_top_accessory_semantic_terms,
)
from SkingToolkit.dense_uv_parser.skin_dataset import PairedRenderSkinDataset
from SkingToolkit.dense_uv_parser.train import (
    stratified_semantic_sample_weights,
    stratified_semantic_split,
)


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

    def test_accessory_hard_recall_requires_argmax_margin(self):
        targets = torch.full((1, 2, 2), 3, dtype=torch.long)
        targets[:, 0, 0] = 0
        losing = torch.zeros(1, 5, 2, 2)
        losing[:, 0, 0, 0] = 2.0
        losing[:, 3, 0, 0] = 4.0
        winning = losing.clone()
        winning[:, 0, 0, 0] = 6.0

        losing_terms = head_top_accessory_semantic_terms(losing, targets)
        winning_terms = head_top_accessory_semantic_terms(winning, targets)
        shifted_terms = head_top_accessory_semantic_terms(
            losing + 17.0,
            targets,
        )

        self.assertGreater(
            losing_terms["loss_head_top_accessory_hard_recall"].item(),
            winning_terms["loss_head_top_accessory_hard_recall"].item(),
        )
        self.assertAlmostEqual(
            losing_terms["loss_head_top_accessory_hard_recall"].item(),
            shifted_terms["loss_head_top_accessory_hard_recall"].item(),
            places=5,
        )

    def test_outer_union_loss_penalizes_outer_pixels_routed_to_inner(self):
        targets = torch.full((1, 4, 4), 3, dtype=torch.long)
        targets[:, :2] = 0
        correct = torch.zeros(1, 5, 4, 4)
        correct[:, 0, :2] = 6.0
        correct[:, 3, 2:] = 6.0
        incorrect = correct.clone()
        incorrect[:, 0, :2] = 0.0
        incorrect[:, 3, :2] = 6.0

        correct_terms = dense_semantic_outer_union_terms(correct, targets)
        incorrect_terms = dense_semantic_outer_union_terms(
            incorrect,
            targets,
        )

        self.assertGreater(
            incorrect_terms[
                "loss_dense_semantic_outer_union_hard_recall"
            ].item(),
            correct_terms[
                "loss_dense_semantic_outer_union_hard_recall"
            ].item(),
        )
        self.assertEqual(
            incorrect_terms["count_dense_semantic_outer_union_fn"].item(),
            8.0,
        )

    def test_hierarchical_attributes_can_overlap(self):
        outer_logit = torch.full((1, 1, 2, 2), 4.0)
        attributes = torch.full((1, 3, 2, 2), -4.0)
        attributes[:, 0, 0, 0] = 4.0
        attributes[:, 1, 0, 0] = 4.0
        outer_target = torch.ones(1, 2, 2, dtype=torch.long)
        attribute_target = torch.zeros(1, 3, 2, 2)
        attribute_target[:, 0, 0, 0] = 1.0
        attribute_target[:, 1, 0, 0] = 1.0
        attribute_target[:, 2, 0, 1:] = 1.0
        attribute_target[:, 2, 1] = 1.0

        terms = hierarchical_dense_semantic_terms(
            outer_logit,
            attributes,
            outer_target,
            attribute_target,
        )

        self.assertEqual(
            terms["count_dense_semantic_attribute_0_tp"].item(), 1.0
        )
        self.assertEqual(
            terms["count_dense_semantic_attribute_1_tp"].item(), 1.0
        )

    def test_semantic_split_and_sampling_are_stratified(self):
        strata = [1] * 8 + [2] * 4 + [3] * 2
        train, validation = stratified_semantic_split(
            strata, val_split=0.25, seed=1234
        )
        self.assertTrue({strata[index] for index in validation} >= {1, 2, 3})
        weights = stratified_semantic_sample_weights(strata, train)
        self.assertEqual(weights.numel(), len(train))
        self.assertGreater(weights.max().item(), weights.min().item())


if __name__ == "__main__":
    unittest.main()
