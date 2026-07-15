import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from SkingToolkit.semantic_uv_reconstruction.dataset import IMAGE_EXTENSIONS, load_skin


IGNORE_INDEX = 255


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
    ):
        self.data_dir = Path(data_dir)
        self.bg_color = bg_color
        self.normalize_model = normalize_model
        self.semantic_labels_dir = Path(semantic_labels_dir) if semantic_labels_dir else None

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
        return sample
