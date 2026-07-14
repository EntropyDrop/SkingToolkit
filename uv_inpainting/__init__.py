"""Parser-conditioned Minecraft skin UV inpainting utilities."""

from .model import UVInpaintingNet
LightUVInpaintingNet = UVInpaintingNet
from .losses import UVInpaintingLoss
from .dataset import UVInpaintingDataset

__all__ = ["UVInpaintingNet", "LightUVInpaintingNet", "UVInpaintingLoss", "UVInpaintingDataset"]
