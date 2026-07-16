import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from SkingToolkit.semantic_uv_reconstruction.dataset import IMAGE_EXTENSIONS, load_skin


IGNORE_INDEX = 255
SIGLIP_CACHE_VERSION = 1


class SigLIPGlobalCache:
    """Memory-mapped frozen SigLIP globals keyed by source skin filename."""

    def __init__(
        self,
        cache_dir,
        expected_views=None,
        expected_model=None,
        expected_data_dir=None,
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
        if self.metadata.get("version") != SIGLIP_CACHE_VERSION:
            raise ValueError(
                f"Unsupported SigLIP cache version {self.metadata.get('version')!r}."
            )
        if expected_views is not None and self.metadata.get("views") != list(expected_views):
            raise ValueError(
                f"SigLIP cache views={self.metadata.get('views')!r}, "
                f"expected {list(expected_views)!r}."
            )
        if expected_model is not None and self.metadata.get("siglip_model") != expected_model:
            raise ValueError(
                f"SigLIP cache model={self.metadata.get('siglip_model')!r}, "
                f"expected {expected_model!r}."
            )
        if expected_data_dir is not None:
            cached_data_dir_value = self.metadata.get("data_dir")
            requested_data_dir = Path(expected_data_dir).resolve()
            if (
                cached_data_dir_value
                and Path(cached_data_dir_value).resolve() != requested_data_dir
            ):
                raise ValueError(
                    f"SigLIP cache data_dir={str(Path(cached_data_dir_value).resolve())!r}, "
                    f"expected {str(requested_data_dir)!r}."
                )
        filenames = self.metadata.get("filenames")
        if not isinstance(filenames, list) or len(set(filenames)) != len(filenames):
            raise ValueError("SigLIP cache filenames must be a unique list.")
        self.filename_to_index = {name: index for index, name in enumerate(filenames)}
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

    def get(self, filename):
        try:
            index = self.filename_to_index[filename]
        except KeyError as error:
            raise KeyError(f"{filename} is missing from the SigLIP cache.") from error
        # Copy the tiny row so DataLoader collation never holds a read-only mmap view.
        return torch.from_numpy(np.array(self.embeddings[index], dtype=np.float32, copy=True))


def load_semantic_uv_label(path):
    """Load a single-channel 64x64 class-id image; 255 means ignore."""
    label = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    if label.shape != (64, 64):
        raise ValueError(f"Expected a 64x64 semantic UV label at {path}, got {label.shape}.")
    return torch.from_numpy(label.astype(np.int64))


class SemanticUVPairDataset(Dataset):
    """Lightweight skin source for fixed-render semantic UV reconstruction.

    Rendering deliberately happens in the training process on the selected
    device. DataLoader workers only read the source 64x64 skin and an optional
    dense structural label. Open-world appearance semantics come from the
    frozen vision-language backbone and require no finite tag vocabulary.
    """

    def __init__(
        self,
        data_dir,
        max_samples=None,
        bg_color=(128, 128, 128),
        normalize_model=True,
        semantic_labels_dir=None,
        siglip_cache_dir=None,
        siglip_cache_views=None,
        siglip_cache_model=None,
    ):
        self.data_dir = Path(data_dir)
        self.bg_color = bg_color
        self.normalize_model = normalize_model
        self.semantic_labels_dir = Path(semantic_labels_dir) if semantic_labels_dir else None
        self.siglip_cache = (
            SigLIPGlobalCache(
                siglip_cache_dir,
                expected_views=siglip_cache_views,
                expected_model=siglip_cache_model,
                expected_data_dir=self.data_dir,
            )
            if siglip_cache_dir
            else None
        )

        self.skin_paths = sorted(
            self.data_dir / filename
            for filename in os.listdir(self.data_dir)
            if filename.lower().endswith(IMAGE_EXTENSIONS) and not filename.startswith("half_")
        )
        if max_samples is not None:
            self.skin_paths = self.skin_paths[:max_samples]
        if not self.skin_paths:
            raise ValueError(f"No skin images found in {self.data_dir}")

        if self.semantic_labels_dir is not None:
            missing = [path.name for path in self.skin_paths if not (self.semantic_labels_dir / path.name).is_file()]
            if missing:
                preview = ", ".join(missing[:5])
                raise ValueError(
                    f"Missing semantic UV labels for {len(missing)} skins in "
                    f"{self.semantic_labels_dir}; first missing: {preview}"
                )
        if self.siglip_cache is not None:
            missing = [
                path.name
                for path in self.skin_paths
                if path.name not in self.siglip_cache.filename_to_index
            ]
            if missing:
                raise ValueError(
                    f"SigLIP cache is missing {len(missing)} selected skins; "
                    f"first missing: {', '.join(missing[:5])}."
                )

    def __len__(self):
        return len(self.skin_paths)

    def __getitem__(self, index):
        skin_path = self.skin_paths[index]
        sample = {
            "uv": load_skin(
                skin_path,
                bg_color=self.bg_color,
                normalize_model=self.normalize_model,
            ),
            "path": str(skin_path),
        }
        if self.semantic_labels_dir is not None:
            sample["semantic_uv"] = load_semantic_uv_label(self.semantic_labels_dir / skin_path.name)
        if self.siglip_cache is not None:
            sample["siglip_raw_global"] = self.siglip_cache.get(skin_path.name)
        return sample
