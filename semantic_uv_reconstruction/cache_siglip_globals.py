"""Compatibility entry point for the relocated Dense UV Parser cache builder."""

import sys
from pathlib import Path


TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TOOLKIT_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from SkingToolkit.dense_uv_parser.cache_semantic_features import main


if __name__ == "__main__":
    main()
