import sys
from pathlib import Path

# Inject workspace root into sys.path to allow absolute imports
TOOLKIT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = TOOLKIT_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

# Import from mc_skin_utils library
from mc_skin_utils.alice_to_steve import alice_to_steve
from mc_skin_utils.mc_voxel_texture_resolver import resolve_voxel_consistency

__all__ = ["alice_to_steve", "resolve_voxel_consistency"]


