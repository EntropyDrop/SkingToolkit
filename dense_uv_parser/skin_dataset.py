"""Minimal 64x64 Minecraft skin dataset for Dense UV Parser training."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from mc_skin_utils.alice_to_steve import alice_to_steve


IMAGE_EXTENSIONS = (".png", ".webp", ".jpg", ".jpeg")


def load_skin(path, bg_color=(128, 128, 128), normalize_model=True):
    skin = Image.open(path).convert("RGBA")
    if skin.size != (64, 64):
        raise ValueError(f"Expected a 64x64 Minecraft skin at {path}, got {skin.size}.")
    if normalize_model and skin.getpixel((47, 52))[3] == 0:
        skin = alice_to_steve(skin)

    skin_np = np.array(skin, dtype=np.uint8)
    alpha = skin_np[..., 3]
    skin_np[(alpha > 0) & (alpha < 255), 3] = 255
    rgba = torch.from_numpy(skin_np.astype(np.float32) / 255.0).permute(2, 0, 1)
    alpha = rgba[3:4]
    background = torch.tensor(bg_color, dtype=rgba.dtype).view(3, 1, 1) / 255.0
    rgba[:3] = torch.where(alpha > 0, rgba[:3], background)
    return rgba.clamp(0.0, 1.0)


class SkinUVDataset(Dataset):
    def __init__(
        self,
        data_dir,
        mappings_dir=None,
        views=None,
        max_samples=None,
        bg_color=(128, 128, 128),
        normalize_model=True,
        **_ignored,
    ):
        # mappings_dir/views remain accepted so old training invocations keep a
        # stable constructor while rendering stays in dense_uv_parser/train.py.
        del mappings_dir, views
        self.data_dir = Path(data_dir)
        self.bg_color = bg_color
        self.normalize_model = bool(normalize_model)
        self.skin_paths = sorted(
            path
            for path in self.data_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and not path.name.startswith("half_")
        )
        if max_samples is not None:
            self.skin_paths = self.skin_paths[:max_samples]
        if not self.skin_paths:
            raise ValueError(f"No skin images found in {self.data_dir}")

    def __len__(self):
        return len(self.skin_paths)

    def __getitem__(self, index):
        skin_path = self.skin_paths[index]
        return {
            "uv": load_skin(
                skin_path,
                bg_color=self.bg_color,
                normalize_model=self.normalize_model,
            ),
            "path": str(skin_path),
        }
