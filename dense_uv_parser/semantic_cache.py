"""Memory-mapped frozen semantic features used by Dense UV Parser."""

import json
from pathlib import Path

import numpy as np
import torch


SIGLIP_CACHE_VERSION = 2
SUPPORTED_SIGLIP_CACHE_VERSIONS = (1, 2)


class SigLIPGlobalCache:
    def __init__(
        self,
        cache_dir,
        expected_views=None,
        expected_model=None,
        expected_data_dir=None,
        require_spatial=False,
    ):
        self.cache_dir = Path(cache_dir)
        metadata_path = self.cache_dir / "metadata.json"
        embeddings_path = self.cache_dir / "embeddings.npy"
        if not metadata_path.is_file() or not embeddings_path.is_file():
            raise FileNotFoundError(
                f"Incomplete SigLIP cache in {self.cache_dir}; expected metadata.json "
                "and embeddings.npy."
            )
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("version") not in SUPPORTED_SIGLIP_CACHE_VERSIONS:
            raise ValueError(
                f"Unsupported SigLIP cache version {self.metadata.get('version')!r}."
            )
        if expected_views is not None and self.metadata.get("views") != list(
            expected_views
        ):
            raise ValueError(
                f"SigLIP cache views={self.metadata.get('views')!r}, "
                f"expected {list(expected_views)!r}."
            )
        if (
            expected_model is not None
            and self.metadata.get("siglip_model") != expected_model
        ):
            raise ValueError(
                f"SigLIP cache model={self.metadata.get('siglip_model')!r}, "
                f"expected {expected_model!r}."
            )
        if expected_data_dir is not None:
            cached_data_dir = self.metadata.get("data_dir")
            requested_data_dir = Path(expected_data_dir).resolve()
            if cached_data_dir and Path(cached_data_dir).resolve() != requested_data_dir:
                raise ValueError(
                    f"SigLIP cache data_dir={str(Path(cached_data_dir).resolve())!r}, "
                    f"expected {str(requested_data_dir)!r}."
                )

        filenames = self.metadata.get("filenames")
        if not isinstance(filenames, list) or len(set(filenames)) != len(filenames):
            raise ValueError("SigLIP cache filenames must be a unique list.")
        self.filename_to_index = {
            filename: index for index, filename in enumerate(filenames)
        }
        self.embeddings = np.load(embeddings_path, mmap_mode="r")
        expected_shape = (
            len(filenames),
            len(self.metadata["views"]),
            int(self.metadata["feature_dim"]),
        )
        if self.embeddings.shape != expected_shape:
            raise ValueError(
                f"SigLIP cache shape={self.embeddings.shape}, expected {expected_shape}."
            )

        spatial_metadata = (
            self.metadata.get("spatial_feature_dim"),
            self.metadata.get("spatial_height"),
            self.metadata.get("spatial_width"),
        )
        self.has_spatial = all(value is not None for value in spatial_metadata)
        self.spatial_embeddings = None
        if self.has_spatial:
            spatial_path = self.cache_dir / "spatial_embeddings.npy"
            if not spatial_path.is_file():
                raise FileNotFoundError(
                    "SigLIP cache metadata declares spatial features but "
                    f"{spatial_path} is missing."
                )
            self.spatial_embeddings = np.load(spatial_path, mmap_mode="r")
            expected_spatial_shape = (
                len(filenames),
                len(self.metadata["views"]),
                int(self.metadata["spatial_feature_dim"]),
                int(self.metadata["spatial_height"]),
                int(self.metadata["spatial_width"]),
            )
            if self.spatial_embeddings.shape != expected_spatial_shape:
                raise ValueError(
                    "SigLIP spatial cache shape="
                    f"{self.spatial_embeddings.shape}, expected "
                    f"{expected_spatial_shape}."
                )
        elif any(value is not None for value in spatial_metadata):
            raise ValueError("SigLIP spatial cache metadata is incomplete.")
        if require_spatial and not self.has_spatial:
            raise ValueError(
                f"SigLIP cache {self.cache_dir} contains only global features; "
                "rebuild it with spatial caching enabled."
            )

    def get(self, filename):
        try:
            index = self.filename_to_index[filename]
        except KeyError as error:
            raise KeyError(f"{filename} is missing from the SigLIP cache.") from error
        return torch.from_numpy(
            np.array(self.embeddings[index], dtype=np.float32, copy=True)
        )

    def get_spatial(self, filename):
        if not self.has_spatial:
            raise ValueError("This SigLIP cache has no spatial features.")
        try:
            index = self.filename_to_index[filename]
        except KeyError as error:
            raise KeyError(f"{filename} is missing from the SigLIP cache.") from error
        return torch.from_numpy(
            np.array(
                self.spatial_embeddings[index],
                dtype=np.float16,
                copy=True,
            )
        )
