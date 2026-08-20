import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from SkingToolkit.dense_uv_parser.cache_semantic_features import (
    cache_is_reusable,
)
from SkingToolkit.dense_uv_parser.semantic_cache import SIGLIP_CACHE_VERSION


class SemanticCacheTest(unittest.TestCase):
    def test_larger_cache_is_reused_for_deterministic_dataset_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            filenames = ["a.png", "b.png", "c.png"]
            metadata = {
                "version": SIGLIP_CACHE_VERSION,
                "data_dir": str(root.resolve()),
                "filenames": filenames,
                "views": ["front_left", "back_left"],
                "siglip_model": "test/siglip",
                "feature_dim": 2,
                "dtype": "float16",
            }
            (cache_dir / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            np.save(
                cache_dir / "embeddings.npy",
                np.zeros((3, 2, 2), dtype=np.float16),
            )
            prefix_dataset = SimpleNamespace(
                data_dir=root,
                skin_paths=[root / "a.png", root / "b.png"],
            )
            self.assertTrue(
                cache_is_reusable(
                    cache_dir,
                    prefix_dataset,
                    ["front_left", "back_left"],
                    "test/siglip",
                )
            )

            non_prefix_dataset = SimpleNamespace(
                data_dir=root,
                skin_paths=[root / "a.png", root / "c.png"],
            )
            self.assertFalse(
                cache_is_reusable(
                    cache_dir,
                    non_prefix_dataset,
                    ["front_left", "back_left"],
                    "test/siglip",
                )
            )


if __name__ == "__main__":
    unittest.main()
