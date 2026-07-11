import unittest

import torch

from SkingToolkit.inverse_uv.model import InverseUVNet


class InverseUVKnownPixelTest(unittest.TestCase):
    def test_preserve_known_mode_controls_hard_copy(self):
        conditioning = torch.zeros(1, 10, 64, 64)
        conditioning[0, 0, 8, 8] = 1.0
        conditioning[0, 3, 8, 8] = 1.0
        conditioning[0, 4, 8, 8] = 1.0

        model = InverseUVNet(input_channels=10, base_channels=8, preserve_known=True)
        for parameter in model.parameters():
            parameter.data.zero_()

        preserved = model(conditioning)
        model.preserve_known = False
        editable = model(conditioning)

        self.assertTrue(
            torch.equal(preserved[0, :, 8, 8], torch.tensor([1.0, 0.0, 0.0, 1.0]))
        )
        self.assertTrue(
            torch.equal(editable[0, :, 8, 8], torch.full((4,), 0.5))
        )
        editable[0, :, 8, 8].sum().backward()
        self.assertGreater(model.head[-1].bias.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
