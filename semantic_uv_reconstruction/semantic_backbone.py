"""Compatibility imports for legacy semantic reconstruction tooling."""

from SkingToolkit.dense_uv_parser.semantic_backbone import (  # noqa: F401
    SigLIP2VisionBackbone,
    TIPSv2VisionBackbone,
)


__all__ = ["SigLIP2VisionBackbone", "TIPSv2VisionBackbone"]
