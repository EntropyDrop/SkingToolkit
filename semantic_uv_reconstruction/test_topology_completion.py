import unittest

import torch

from SkingToolkit.semantic_uv_reconstruction.topology import (
    INVALID_SURFACE,
    SURFACE_COUNT,
    build_uv_topology,
)
from SkingToolkit.semantic_uv_reconstruction.topology_model import (
    TopologyAwareUVCompletionNet,
)


class UVTopologyTest(unittest.TestCase):
    def test_topology_covers_the_two_valid_atlas_layers(self):
        topology = build_uv_topology()
        self.assertEqual(int(topology.valid.sum()), 3264)
        self.assertEqual(tuple(topology.surface_pool.shape), (SURFACE_COUNT, 4096))
        self.assertTrue(topology.neighbour_valid[topology.valid.reshape(-1)].all())
        self.assertTrue(
            torch.all(topology.surface[~topology.valid] == INVALID_SURFACE)
        )

    def test_surface_neighbours_stay_on_the_same_part_and_layer_and_cross_seams(self):
        topology = build_uv_topology()
        valid = topology.valid.reshape(-1)
        layer = topology.layer.reshape(-1)
        part = topology.part.reshape(-1)
        face = topology.face.reshape(-1)
        indices = valid.nonzero(as_tuple=False).flatten()
        neighbours = topology.neighbours[indices]
        self.assertTrue(torch.equal(layer[neighbours], layer[indices].unsqueeze(1).expand_as(neighbours)))
        self.assertTrue(torch.equal(part[neighbours], part[indices].unsqueeze(1).expand_as(neighbours)))
        self.assertTrue((face[neighbours] != face[indices].unsqueeze(1)).any())

    def test_inner_outer_pairing_is_symmetric(self):
        topology = build_uv_topology()
        paired = topology.paired_layer_texel.reshape(-1)
        valid_indices = topology.valid.reshape(-1).nonzero(as_tuple=False).flatten()
        self.assertTrue(torch.equal(paired[paired[valid_indices]], valid_indices))
        layer = topology.layer.reshape(-1)
        self.assertTrue(torch.equal(layer[paired[valid_indices]], 1 - layer[valid_indices]))


class TopologyAwareCompletionTest(unittest.TestCase):
    def build_model(self):
        torch.manual_seed(7)
        return TopologyAwareUVCompletionNet(
            hidden_channels=32,
            layers=1,
            attention_heads=4,
            dropout=0.0,
        )

    def test_forward_preserves_observed_texels_exactly(self):
        model = self.build_model()
        self.assertEqual(model.blocks[0].neighbours.shape[1], 5)
        conditioning = torch.zeros(1, 10, 64, 64)
        topology = build_uv_topology()
        inner_flat = ((topology.layer == 0) & topology.valid).reshape(-1).nonzero()[0, 0]
        outer_flat = ((topology.layer == 1) & topology.valid).reshape(-1).nonzero()[0, 0]
        inner_y, inner_x = divmod(int(inner_flat), 64)
        outer_y, outer_x = divmod(int(outer_flat), 64)
        inner_rgba = torch.tensor([0.1, 0.2, 0.3, 1.0])
        outer_rgba = torch.tensor([0.9, 0.7, 0.2, 1.0])
        conditioning[0, 0:4, inner_y, inner_x] = inner_rgba
        conditioning[0, 4, inner_y, inner_x] = 1.0
        conditioning[0, 5:9, outer_y, outer_x] = outer_rgba
        conditioning[0, 9, outer_y, outer_x] = 1.0

        output = model(conditioning)
        self.assertTrue(torch.equal(output[0, :, inner_y, inner_x], inner_rgba))
        self.assertTrue(torch.equal(output[0, :, outer_y, outer_x], outer_rgba))
        self.assertTrue(torch.equal(output[0, 3][topology.layer == 0], torch.ones_like(output[0, 3][topology.layer == 0])))
        self.assertTrue(torch.equal(output[0, 3][~topology.valid], torch.zeros_like(output[0, 3][~topology.valid])))

    def test_discrete_unknown_loss_backpropagates(self):
        model = self.build_model()
        conditioning = torch.zeros(1, 10, 64, 64)
        target = torch.rand(1, 4, 64, 64)
        topology = build_uv_topology()
        target[:, 3] = topology.valid.float()
        outputs = model(conditioning, return_logits=True)
        losses = model.masked_token_loss(outputs, target)
        self.assertTrue(torch.isfinite(losses["loss_token"]))
        losses["loss_token"].backward()
        self.assertIsNotNone(model.rgb_head.weight.grad)
        self.assertGreater(float(model.rgb_head.weight.grad.abs().sum()), 0.0)

    def test_iterative_generation_is_deterministic_and_preserves_known(self):
        model = self.build_model().eval()
        topology = build_uv_topology()
        conditioning = torch.zeros(1, 10, 64, 64)
        flat = ((topology.layer == 0) & topology.valid).reshape(-1).nonzero()[0, 0]
        y, x = divmod(int(flat), 64)
        known = torch.tensor([12, 34, 56, 255], dtype=torch.float32) / 255.0
        conditioning[0, 0:4, y, x] = known
        conditioning[0, 4, y, x] = 1.0

        first = model.generate(conditioning, steps=2, temperature=0.0, seed=9)
        second = model.generate(conditioning, steps=2, temperature=0.0, seed=999)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first[0, :, y, x], known))
        quantized = first[:, :3] * 255.0
        self.assertTrue(torch.equal(quantized, quantized.round()))

    def test_confidence_aware_input_only_locks_high_confidence_evidence(self):
        model = TopologyAwareUVCompletionNet(
            input_channels=12,
            hidden_channels=32,
            layers=1,
            attention_heads=4,
            dropout=0.0,
            hard_lock_threshold=0.8,
        )
        topology = build_uv_topology()
        conditioning = torch.zeros(1, 12, 64, 64)
        inner_indices = ((topology.layer == 0) & topology.valid).reshape(-1).nonzero()
        high_y, high_x = divmod(int(inner_indices[0, 0]), 64)
        soft_y, soft_x = divmod(int(inner_indices[1, 0]), 64)
        conditioning[0, 0:4, high_y, high_x] = torch.tensor([0.1, 0.2, 0.3, 1.0])
        conditioning[0, 4, high_y, high_x] = 1.0
        conditioning[0, 5, high_y, high_x] = 0.9
        conditioning[0, 0:4, soft_y, soft_x] = torch.tensor([0.7, 0.6, 0.5, 1.0])
        conditioning[0, 4, soft_y, soft_x] = 1.0
        conditioning[0, 5, soft_y, soft_x] = 0.6

        outputs = model(conditioning, return_logits=True)
        high_flat = high_y * 64 + high_x
        soft_flat = soft_y * 64 + soft_x
        self.assertEqual(float(outputs["known"][0, high_flat, 0]), 1.0)
        self.assertEqual(float(outputs["known"][0, soft_flat, 0]), 0.0)
        self.assertEqual(float(outputs["evidence"][0, soft_flat, 0]), 1.0)
        self.assertTrue(
            torch.equal(
                outputs["uv"][0, :, high_y, high_x],
                conditioning[0, 0:4, high_y, high_x],
            )
        )


if __name__ == "__main__":
    unittest.main()
