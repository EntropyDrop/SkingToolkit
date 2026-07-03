"""Supervised front/back render to Minecraft UV training utilities."""

from .model import InverseUVNet
LightInverseUVNet = InverseUVNet
from .losses import InverseUVLoss
from .dataset import InverseUVDataset

__all__ = ["InverseUVNet", "LightInverseUVNet", "InverseUVLoss", "InverseUVDataset"]
