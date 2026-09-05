"""Contract tests for training changes, using the local v61 package."""
import unittest
from types import SimpleNamespace

import torch

from SkingToolkit.dense_uv_parser.semantic_generalization import (
    appearance_augment, build_split, eligible, flat_route_loss,
    head_outer_hard_positive_loss, semantics_disabled, skip_regularizer,
)
from SkingToolkit.dense_uv_parser.model import DenseUVParserNet


class SemanticGeneralizationTests(unittest.TestCase):
    def test_appearance_preserves_alpha_background_labels_and_seed(self):
        image = torch.rand(4, 4, 24, 16)
        fg = torch.zeros(4, 1, 24, 16)
        fg[:, :, 4:20, 3:13] = 1
        out = appearance_augment(image, fg, generator=torch.Generator().manual_seed(5))
        repeat = appearance_augment(image, fg, generator=torch.Generator().manual_seed(5))
        self.assertTrue(torch.equal(out, repeat))
        self.assertTrue(torch.equal(out[:, 3], image[:, 3]))
        background = (~fg.bool()).expand(-1, 3, -1, -1)
        self.assertTrue(torch.equal(out[:, :3][background], image[:, :3][background]))
        self.assertGreater(float((out - image).abs().sum()), 0)
        self.assertTrue(torch.equal(appearance_augment(image, fg, 0), image))

    def test_pilot_split_is_subset_of_original_training_split(self):
        dataset = list(range(1800))
        train, val = build_split(dataset, 1234, 0.1)
        pilot = set(train.indices[:128])
        self.assertFalse(pilot & set(val.indices))
        self.assertEqual(build_split(dataset, 1234, 0.1)[0].indices, train.indices)

    def test_context_dropout_does_not_change_inference_or_checkpoint(self):
        model = DenseUVParserNet(base_channels=8, geometry_only=True, view_classes=2)
        before = set(model.state_dict())
        image = torch.rand(2, 4, 32, 16)
        ids = torch.tensor([0, 1])
        model.eval()
        with torch.no_grad():
            expected = model(image, view_ids=ids)["layer"]
            handles = skip_regularizer(model, 0.75)
            actual = model(image, view_ids=ids)["layer"]
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(before, set(model.state_dict()))
        for handle in handles:
            handle.remove()

    def test_hard_positive_loss_pushes_missed_head_outer_towards_outer(self):
        logits = torch.tensor([[[[3.0]], [[-2.0]], [[0.0]]]], requires_grad=True)
        target = {"route_role": torch.ones(1, 1, 1, dtype=torch.long),
                  "part": torch.zeros(1, 1, 1, dtype=torch.long)}
        head_outer_hard_positive_loss(logits, target).backward()
        self.assertLess(float(logits.grad[0, 1, 0, 0]), 0)
        target["part"].fill_(1)
        self.assertEqual(float(head_outer_hard_positive_loss(logits, target).detach()), 0)

    def test_regression_guard_rejects_recall_collapse(self):
        baseline = {section: {key: 0.9 for key in
                    ("inner_iou", "outer_iou", "outer_precision", "outer_recall")}
                    for section in ("clean", "hard_uv")}
        metrics = {k: dict(v) for k, v in baseline.items()}
        metrics["hard_uv"]["outer_recall"] = 0.8
        self.assertFalse(eligible(metrics, baseline, 0.015))
        self.assertTrue(eligible(baseline, baseline, 0.015))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
