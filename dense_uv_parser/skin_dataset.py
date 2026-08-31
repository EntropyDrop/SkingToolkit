"""Minimal 64x64 Minecraft skin dataset for Dense UV Parser training."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

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


class PairedRenderSkinDataset(Dataset):
    """Stage-one render / 64x64 UV pairs for real-domain semantic training.

    Website generation archives store a normalized two-view render as
    ``*_edited`` beside a versioned result skin such as ``*_v94_result``.
    These pairs are the only validation source that measures the domain used
    by inference; rendering a random UV with the differentiable renderer cannot expose
    failures caused by stage-one shading, antialiasing, or color statistics.
    """

    def __init__(
        self,
        data_dir,
        views=("front_left", "back_left"),
        view_size=(512, 256),
        max_samples=None,
        bg_color=(128, 128, 128),
        normalize_model=True,
        manifest_path=None,
        result_suffix="_result",
        **_ignored,
    ):
        self.data_dir = Path(data_dir)
        self.views = tuple(views)
        self.view_size = tuple(int(value) for value in view_size)
        self.bg_color = bg_color
        self.normalize_model = bool(normalize_model)
        self.result_suffix = str(result_suffix)
        if (
            not self.result_suffix.startswith("_")
            or "/" in self.result_suffix
            or "\\" in self.result_suffix
        ):
            raise ValueError(
                "result_suffix must start with '_' and contain no path separators."
            )
        if len(self.views) != 2:
            raise ValueError(
                "Paired render training currently requires exactly the "
                "front_left and back_left inference views."
            )
        if min(self.view_size) < 1:
            raise ValueError("view_size must contain positive dimensions.")

        pairs = []
        if manifest_path is not None:
            manifest_path = Path(manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_root = Path(manifest["data_dir"]).resolve()
            if manifest_root != self.data_dir.resolve():
                raise ValueError(
                    f"Paired manifest data_dir={manifest_root} does not match "
                    f"requested {self.data_dir.resolve()}."
                )
            manifest_suffix = manifest.get("result_suffix")
            if (
                manifest_suffix is not None
                and manifest_suffix != self.result_suffix
            ):
                raise ValueError(
                    f"Paired manifest result_suffix={manifest_suffix!r} does "
                    f"not match requested {self.result_suffix!r}."
                )
            for item in manifest.get("pairs", []):
                edited_path = self.data_dir / item["edited"]
                result_path = self.data_dir / item["result"]
                if not edited_path.is_file() or not result_path.is_file():
                    raise FileNotFoundError(
                        "Paired semantic manifest references a missing file: "
                        f"{edited_path} or {result_path}."
                    )
                pairs.append((edited_path, result_path))
        else:
            for edited_path in sorted(self.data_dir.rglob("*_edited.*")):
                if edited_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                sample_stem = edited_path.stem[: -len("_edited")]
                result_paths = [
                    path
                    for path in sorted(
                        edited_path.parent.glob(
                            f"{sample_stem}{self.result_suffix}.*"
                        )
                    )
                    if path.suffix.lower() in IMAGE_EXTENSIONS
                ]
                if not result_paths:
                    continue
                result_path = result_paths[0]
                try:
                    with Image.open(result_path) as result_image:
                        result_size = result_image.size
                except (OSError, ValueError):
                    continue
                if result_size != (64, 64):
                    continue
                pairs.append((edited_path, result_path))
        if max_samples is not None:
            pairs = pairs[: int(max_samples)]
        if not pairs:
            raise ValueError(
                "No *_edited / 64x64 "
                f"*{self.result_suffix} pairs found under {self.data_dir}."
            )
        self.pairs = pairs
        # Keep the established cache/trainer interface.  Cache keys use the
        # edited input filename because its pixels, rather than the UV result,
        # are encoded by SigLIP2.
        self.skin_paths = [edited_path for edited_path, _ in pairs]
        self._semantic_strata = None

    def __len__(self):
        return len(self.pairs)

    def semantic_strata(self, alpha_threshold=0.5):
        """Return deterministic top/eye/other presence bitmasks per pair."""
        if self._semantic_strata is not None:
            return list(self._semantic_strata)
        from SkingToolkit.dense_uv_parser.semantic_targets import (
            build_head_eye_accessory_face_targets,
            build_head_top_accessory_face_targets,
            head_outer_face_values_to_uv,
        )
        from SkingToolkit.dense_uv_parser.uv_layout import (
            build_part_layer_masks,
        )

        _, outer_part_masks = build_part_layer_masks()
        outer_atlas = outer_part_masks[:, 0].bool().any(dim=0)
        strata = []
        for _, result_path in self.pairs:
            uv = load_skin(
                result_path,
                bg_color=self.bg_color,
                normalize_model=self.normalize_model,
            ).unsqueeze(0)
            top_faces = build_head_top_accessory_face_targets(
                uv, alpha_threshold=alpha_threshold
            )["mask"]
            eye_faces = build_head_eye_accessory_face_targets(
                uv, alpha_threshold=alpha_threshold
            )["mask"]
            top_uv = head_outer_face_values_to_uv(top_faces)[0, 0] > 0.5
            eye_uv = head_outer_face_values_to_uv(eye_faces)[0, 0] > 0.5
            occupied_outer = (uv[0, 3] > float(alpha_threshold)) & outer_atlas
            other = occupied_outer & ~top_uv & ~eye_uv
            stratum = (
                int(top_uv.any().item())
                | (int(eye_uv.any().item()) << 1)
                | (int(other.any().item()) << 2)
            )
            strata.append(stratum)
        self._semantic_strata = tuple(strata)
        return list(self._semantic_strata)

    def _load_views(self, path):
        with Image.open(path) as source_image:
            image = source_image.convert("RGB")
        width, height = image.size
        view_count = len(self.views)
        if width % view_count != 0:
            raise ValueError(
                f"Paired render width {width} at {path} is not divisible by "
                f"{view_count} views."
            )
        view_width = width // view_count
        tensors = []
        for index in range(view_count):
            view = image.crop(
                (index * view_width, 0, (index + 1) * view_width, height)
            )
            tensor = TF.to_tensor(view)
            if tuple(tensor.shape[-2:]) != self.view_size:
                tensor = F.interpolate(
                    tensor.unsqueeze(0),
                    size=self.view_size,
                    mode="nearest-exact",
                ).squeeze(0)
            tensors.append(
                torch.cat([tensor, torch.ones_like(tensor[:1])], dim=0)
            )
        return torch.stack(tensors, dim=0).clamp(0.0, 1.0)

    def __getitem__(self, index):
        edited_path, result_path = self.pairs[index]
        return {
            "uv": load_skin(
                result_path,
                bg_color=self.bg_color,
                normalize_model=self.normalize_model,
            ),
            "rendered": self._load_views(edited_path),
            "path": str(edited_path),
            "uv_path": str(result_path),
        }
