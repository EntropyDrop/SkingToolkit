import unittest

import torch

from SkingToolkit.semantic_uv_reconstruction.topology import (
    INVALID_SURFACE,
    MIRRORED_PART,
    SURFACE_COUNT,
    build_uv_topology,
    simple_symmetry_nearest_inpaint,
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

    def test_character_space_mirror_is_exact_and_involutive(self):
        topology = build_uv_topology()
        valid = topology.valid.reshape(-1)
        indices = valid.nonzero(as_tuple=False).flatten()
        mirrored = topology.mirrored_texel.reshape(-1)
        layer = topology.layer.reshape(-1)
        part = topology.part.reshape(-1)
        positions = topology.world_position.reshape(-1, 3)
        expected_positions = positions[indices].clone()
        expected_positions[:, 0] = -expected_positions[:, 0]
        mirrored_parts = torch.tensor(MIRRORED_PART, dtype=torch.long)

        self.assertTrue(torch.equal(mirrored[mirrored[indices]], indices))
        self.assertTrue(torch.equal(layer[mirrored[indices]], layer[indices]))
        self.assertTrue(
            torch.equal(part[mirrored[indices]], mirrored_parts[part[indices]])
        )
        self.assertTrue(
            torch.equal(positions[mirrored[indices]], expected_positions)
        )

    def test_bottom_face_uv_depth_runs_from_back_edge_to_front_edge(self):
        topology = build_uv_topology()
        valid = topology.valid & (topology.layer == 0)

        for part in range(6):
            bottom = valid & (topology.part == part) & (topology.face == 5)
            front = valid & (topology.part == part) & (topology.face == 0)
            back = valid & (topology.part == part) & (topology.face == 1)
            bottom_indices = bottom.reshape(-1).nonzero(as_tuple=False).flatten()
            local_v = topology.local_uv.reshape(-1, 2)[bottom_indices, 1]
            positions = topology.world_position.reshape(-1, 3)

            back_edge = bottom_indices[local_v == local_v.min()]
            front_edge = bottom_indices[local_v == local_v.max()]
            front_positions = positions[front.reshape(-1)]
            back_positions = positions[back.reshape(-1)]
            back_edge_to_back = torch.cdist(
                positions[back_edge], back_positions
            ).min(dim=1).values
            back_edge_to_front = torch.cdist(
                positions[back_edge], front_positions
            ).min(dim=1).values
            front_edge_to_front = torch.cdist(
                positions[front_edge], front_positions
            ).min(dim=1).values
            front_edge_to_back = torch.cdist(
                positions[front_edge], back_positions
            ).min(dim=1).values

            self.assertLess(
                float(back_edge_to_back.max()),
                float(back_edge_to_front.min()),
            )
            self.assertLess(
                float(front_edge_to_front.max()),
                float(front_edge_to_back.min()),
            )

    def test_inner_fill_order_is_face_outer_to_inner_clockwise(self):
        topology = build_uv_topology()
        # Head-front is the first part/face and occupies x=8..15, y=8..15.
        expected_outer_ring = torch.tensor(
            [8 * 64 + x for x in range(8, 16)]
            + [y * 64 + 15 for y in range(9, 16)]
            + [15 * 64 + x for x in range(14, 7, -1)]
            + [y * 64 + 8 for y in range(14, 8, -1)],
            dtype=torch.long,
        )
        expected_second_ring = torch.tensor(
            [9 * 64 + x for x in range(9, 15)]
            + [y * 64 + 14 for y in range(10, 15)]
            + [14 * 64 + x for x in range(13, 8, -1)]
            + [y * 64 + 9 for y in range(13, 9, -1)],
            dtype=torch.long,
        )
        outer_count = len(expected_outer_ring)
        second_count = len(expected_second_ring)

        self.assertTrue(
            torch.equal(
                topology.inner_fill_order[:outer_count],
                expected_outer_ring,
            )
        )
        self.assertTrue(
            torch.equal(
                topology.inner_fill_order[
                    outer_count : outer_count + second_count
                ],
                expected_second_ring,
            )
        )

    def test_every_inner_face_finishes_each_outer_ring_before_moving_inward(self):
        topology = build_uv_topology()
        order = topology.inner_fill_order
        ordered_surfaces = topology.surface.reshape(-1)[order]

        for surface in range(SURFACE_COUNT // 2):
            face_order = order[ordered_surfaces == surface]
            x = face_order % 64
            y = torch.div(face_order, 64, rounding_mode="floor")
            ring = torch.minimum(
                torch.minimum(x - x.min(), x.max() - x),
                torch.minimum(y - y.min(), y.max() - y),
            )
            self.assertTrue(torch.all(ring[1:] >= ring[:-1]))

    def test_simple_inpaint_prefers_known_symmetry_before_nearest_3d(self):
        topology = build_uv_topology()
        valid_inner = (
            topology.valid & (topology.layer == 0)
        ).reshape(-1).nonzero(as_tuple=False).flatten()
        target = int(valid_inner[len(valid_inner) // 3])
        mirrored = int(topology.mirrored_texel.reshape(-1)[target])
        paired = int(topology.paired_layer_texel.reshape(-1)[target])
        uv = torch.zeros(4, 64, 64)
        flat = uv.reshape(4, -1)
        mirror_rgba = torch.tensor([0.9, 0.1, 0.2, 1.0])
        nearer_rgba = torch.tensor([0.1, 0.8, 0.3, 1.0])
        flat[:, mirrored] = mirror_rgba
        flat[:, paired] = nearer_rgba

        repaired, stats = simple_symmetry_nearest_inpaint(uv)

        self.assertTrue(torch.equal(repaired.reshape(4, -1)[:, target], mirror_rgba))
        self.assertTrue(torch.equal(repaired.reshape(4, -1)[:, mirrored], mirror_rgba))
        self.assertGreater(stats["symmetry_filled_texels"], 0)
        self.assertGreater(stats["unresolved_texels"], 0)

    def test_simple_inpaint_nearest_3d_can_copy_from_outer_layer(self):
        topology = build_uv_topology()
        valid_inner = (
            topology.valid & (topology.layer == 0)
        ).reshape(-1).nonzero(as_tuple=False).flatten()
        target = int(valid_inner[len(valid_inner) // 2])
        outer_source = int(topology.paired_layer_texel.reshape(-1)[target])
        uv = torch.zeros(4, 64, 64)
        outer_rgba = torch.tensor([0.2, 0.4, 0.95, 1.0])
        uv.reshape(4, -1)[:, outer_source] = outer_rgba

        repaired, stats = simple_symmetry_nearest_inpaint(uv)

        self.assertTrue(torch.equal(repaired.reshape(4, -1)[:, target], outer_rgba))
        self.assertEqual(stats["known_outer_texels"], 1)
        self.assertGreater(stats["nearest_3d_filled_texels"], 0)
        self.assertGreater(stats["unresolved_texels"], 0)

    def test_simple_inpaint_preserves_outer_layer_exactly(self):
        topology = build_uv_topology()
        outer_indices = (
            topology.valid & (topology.layer == 1)
        ).reshape(-1).nonzero(as_tuple=False).flatten()
        inner_source = int(
            (
                topology.valid & (topology.layer == 0)
            ).reshape(-1).nonzero(as_tuple=False).flatten()[0]
        )
        uv = torch.zeros(4, 64, 64)
        flat = uv.reshape(4, -1)
        flat[:, inner_source] = torch.tensor([0.4, 0.5, 0.6, 1.0])
        flat[:, int(outer_indices[0])] = torch.tensor([0.9, 0.2, 0.1, 1.0])
        flat[:, int(outer_indices[1])] = torch.tensor([0.3, 0.7, 0.8, 0.0])
        outer_before = flat[:, outer_indices].clone()

        repaired, stats = simple_symmetry_nearest_inpaint(uv)

        self.assertTrue(
            torch.equal(repaired.reshape(4, -1)[:, outer_indices], outer_before)
        )
        self.assertEqual(stats["preserved_outer_texels"], int(outer_indices.numel()))
        self.assertGreater(stats["unresolved_texels"], 0)

    def test_simple_inpaint_does_not_copy_nearest_color_across_parts(self):
        topology = build_uv_topology()
        flat_part = topology.part.reshape(-1)
        flat_layer = topology.layer.reshape(-1)
        flat_valid = topology.valid.reshape(-1)
        head_source = int(
            (flat_valid & (flat_layer == 0) & (flat_part == 0))
            .nonzero(as_tuple=False)
            .flatten()[0]
        )
        body_target = int(
            (flat_valid & (flat_layer == 0) & (flat_part == 1))
            .nonzero(as_tuple=False)
            .flatten()[0]
        )
        uv = torch.zeros(4, 64, 64)
        uv.reshape(4, -1)[:, head_source] = torch.tensor(
            [0.8, 0.3, 0.1, 1.0]
        )

        repaired, stats = simple_symmetry_nearest_inpaint(uv)

        self.assertTrue(
            torch.equal(
                repaired.reshape(4, -1)[:, body_target], torch.zeros(4)
            )
        )
        self.assertGreater(stats["unresolved_texels"], 0)


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

    def test_mean_decode_avoids_uniform_logit_argmax_extreme(self):
        model = self.build_model().eval()
        with torch.no_grad():
            model.rgb_head.weight.zero_()
            model.rgb_head.bias.zero_()
        conditioning = torch.zeros(1, 10, 64, 64)
        topology = build_uv_topology()
        inner_flat = ((topology.layer == 0) & topology.valid).reshape(-1).nonzero()[0, 0]
        y, x = divmod(int(inner_flat), 64)

        mean = model.generate(
            conditioning,
            steps=1,
            temperature=0.0,
            rgb_decode="mean",
        )
        legacy = model.generate(
            conditioning,
            steps=1,
            temperature=0.0,
            rgb_decode="argmax",
        )

        self.assertTrue(torch.equal(mean[0, :3, y, x], torch.full((3,), 128.0 / 255.0)))
        self.assertTrue(torch.equal(legacy[0, :3, y, x], torch.zeros(3)))

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

    def test_palette_snap_prevents_generated_unobserved_colors(self):
        model = TopologyAwareUVCompletionNet(
            input_channels=12,
            hidden_channels=32,
            layers=1,
            attention_heads=4,
            dropout=0.0,
        ).eval()
        topology = build_uv_topology()
        conditioning = torch.zeros(1, 12, 64, 64)
        reference_flat = (
            ((topology.layer == 0) & topology.valid)
            .reshape(-1)
            .nonzero(as_tuple=False)[0, 0]
        )
        reference_y, reference_x = divmod(int(reference_flat), 64)
        reference_rgb = torch.tensor([17.0, 93.0, 201.0]) / 255.0
        conditioning[0, 0:3, reference_y, reference_x] = reference_rgb
        conditioning[0, 3, reference_y, reference_x] = 1.0
        conditioning[0, 4, reference_y, reference_x] = 1.0
        conditioning[0, 5, reference_y, reference_x] = 1.0

        generated = model.generate(
            conditioning,
            steps=1,
            temperature=0.0,
            palette_snap=True,
        )
        generated_rgb = generated[0, :3].permute(1, 2, 0)
        opaque_inner = (topology.layer == 0) & topology.valid
        self.assertTrue(
            torch.equal(
                generated_rgb[opaque_inner],
                reference_rgb.expand(int(opaque_inner.sum()), -1),
            )
        )

    def test_palette_snap_copies_nearest_same_surface_evidence(self):
        model = TopologyAwareUVCompletionNet(
            input_channels=12,
            hidden_channels=32,
            layers=1,
            attention_heads=4,
            dropout=0.0,
        ).eval()
        topology = build_uv_topology()
        surface_indices = (topology.surface.reshape(-1) == 0).nonzero(
            as_tuple=False
        ).flatten()
        target_index = surface_indices[len(surface_indices) // 2]
        distances = torch.cdist(
            topology.local_uv.reshape(-1, 2)[target_index].view(1, 2),
            topology.local_uv.reshape(-1, 2)[surface_indices],
        )[0]
        order = distances.argsort()
        near_index = surface_indices[order[1]]
        far_index = surface_indices[order[-1]]

        result = torch.zeros(1, 64 * 64, 4)
        result[0, target_index, 3] = 1.0
        observed = torch.zeros_like(result)
        near_color = torch.tensor([0.1, 0.2, 0.3])
        far_color = torch.tensor([0.8, 0.7, 0.6])
        observed[0, near_index, :3] = near_color
        observed[0, far_index, :3] = far_color
        observed[0, near_index, 3] = 1.0
        observed[0, far_index, 3] = 1.0
        evidence = torch.zeros(1, 64 * 64, 1)
        confidence = torch.zeros_like(evidence)
        evidence[0, near_index] = 1.0
        evidence[0, far_index] = 1.0
        confidence[0, near_index] = 1.0
        confidence[0, far_index] = 1.0
        generated_mask = torch.zeros(1, 64 * 64, dtype=torch.bool)
        generated_mask[0, target_index] = True

        snapped = model._snap_generated_rgb_to_evidence_palette(
            result,
            observed,
            evidence,
            confidence,
            generated_mask,
            min_confidence=0.75,
        )

        self.assertTrue(torch.equal(snapped[0, target_index, :3], near_color))

    def test_palette_snap_keeps_joint_observed_rgb_triplets(self):
        model = TopologyAwareUVCompletionNet(
            input_channels=12,
            hidden_channels=32,
            layers=1,
            attention_heads=4,
            dropout=0.0,
        ).eval()
        topology = build_uv_topology()
        surface_indices = (topology.surface.reshape(-1) == 0).nonzero(
            as_tuple=False
        ).flatten()
        red_index, green_index, target_index = surface_indices[:3]
        result = torch.zeros(1, 64 * 64, 4)
        result[0, target_index] = torch.tensor([1.0, 1.0, 0.0, 1.0])
        observed = torch.zeros_like(result)
        observed[0, red_index] = torch.tensor([1.0, 0.0, 0.0, 1.0])
        observed[0, green_index] = torch.tensor([0.0, 1.0, 0.0, 1.0])
        evidence = torch.zeros(1, 64 * 64, 1)
        confidence = torch.zeros_like(evidence)
        evidence[0, red_index] = 1.0
        evidence[0, green_index] = 1.0
        confidence[0, red_index] = 1.0
        confidence[0, green_index] = 1.0
        generated = torch.zeros(1, 64 * 64, dtype=torch.bool)
        generated[0, target_index] = True

        snapped = model._snap_generated_rgb_to_evidence_palette(
            result,
            observed,
            evidence,
            confidence,
            generated,
            min_confidence=0.75,
        )

        snapped_rgb = snapped[0, target_index, :3]
        candidates = torch.stack(
            [observed[0, red_index, :3], observed[0, green_index, :3]]
        )
        self.assertTrue((candidates == snapped_rgb).all(dim=1).any())
        self.assertFalse(torch.equal(snapped_rgb, torch.tensor([1.0, 1.0, 0.0])))


if __name__ == "__main__":
    unittest.main()
