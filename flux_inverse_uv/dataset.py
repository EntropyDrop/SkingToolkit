import os
import sys
from pathlib import Path

FLUX_INVERSE_UV_DIR = Path(__file__).resolve().parent
TOOLKIT_ROOT = FLUX_INVERSE_UV_DIR.parent
WORKSPACE_ROOT = TOOLKIT_ROOT.parent

for p in [str(WORKSPACE_ROOT), str(TOOLKIT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class FluxInverseUVDataset(Dataset):
    def __init__(
        self,
        control_imgs_dir=None,
        photos_dir=None,
        target_imgs_dir=None,
        data_dir=None,
        captions_dir=None,
        cond_size=512,
        is_square=True,
        bg_color=(128, 128, 128),
        alpha_threshold=128,
        default_caption=""
    ):
        """
        PyTorch Dataset for Flux Inverse UV Fine-tuning.
        Args:
            control_imgs_dir: Path to conditioning control_imgs folder.
            photos_dir: Alias for control_imgs_dir.
            target_imgs_dir: Path to pre-built target_imgs folder (512x512 target skin images).
            data_dir: Optional path to skins folder containing 64x64 skin PNGs.
            captions_dir: Optional path to captions folder.
            cond_size: Target/control image resolution (e.g. 512).
            is_square: If True, target_width == cond_size (512x512). Otherwise cond_size // 2.
            bg_color: Solid gray color (128,128,128) for matte background.
            alpha_threshold: Source skin alpha below this value is treated as transparent.
            default_caption: Caption used when no .txt caption exists.
        """
        self.control_imgs_dir = control_imgs_dir or photos_dir or str(FLUX_INVERSE_UV_DIR / "control_imgs")
        self.photos_dir = self.control_imgs_dir
        self.target_imgs_dir = target_imgs_dir or str(FLUX_INVERSE_UV_DIR / "target_imgs")
        self.data_dir = data_dir
        self.captions_dir = captions_dir or self.control_imgs_dir
        self.cond_size = cond_size
        self.is_square = is_square
        self.target_height = cond_size
        self.target_width = cond_size if is_square else cond_size // 2
        self.bg_color = bg_color
        self.alpha_threshold = alpha_threshold
        self.default_caption = default_caption
        self.target_resample = Image.Resampling.NEAREST

        mask_np = np.array(Image.open(FLUX_INVERSE_UV_DIR / "skin-mask.png"))
        decor_mask_np = np.array(Image.open(FLUX_INVERSE_UV_DIR / "skin-decor-mask.png"))
        self.active_mask_64 = (mask_np[..., 3] > 0) | (decor_mask_np[..., 3] > 0)

        # Scan filenames from target_imgs_dir or photos_dir
        scan_dir = self.target_imgs_dir if os.path.exists(self.target_imgs_dir) else self.photos_dir
        if not os.path.exists(scan_dir):
            raise FileNotFoundError(f"Neither target_imgs_dir '{self.target_imgs_dir}' nor photos_dir '{self.photos_dir}' exists.")

        self.filenames = sorted([
            f for f in os.listdir(scan_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")) and not f.startswith("half_")
        ])

        if len(self.filenames) == 0:
            print(f"WARNING: No image files found in directory: {scan_dir}")

        self.transform_cond = transforms.Compose([
            transforms.Resize((self.target_height, self.target_width), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
        ])

    def _infer_dot_mask_64(self, target_img):
        """Infer transparent-dot blocks from the canonical target encoding."""
        if self.target_width % 64 != 0 or self.target_height % 64 != 0:
            raise ValueError(
                f"Target size must be divisible by 64, got {self.target_width}x{self.target_height}"
            )

        ratio_x = self.target_width // 64
        ratio_y = self.target_height // 64
        if ratio_x != ratio_y:
            raise ValueError(
                f"Target blocks must be square, got block size {ratio_x}x{ratio_y}"
            )

        target_rgb = target_img.convert("RGB").resize(
            (self.target_width, self.target_height),
            resample=self.target_resample,
        )
        arr = np.array(target_rgb, dtype=np.int16)
        blocks = arr.reshape(64, ratio_y, 64, ratio_x, 3).transpose(0, 2, 1, 3, 4)

        dot_size = max(1, ratio_x // 2)
        dot_start = (ratio_x - dot_size) // 2
        dot_end = dot_start + dot_size

        core = blocks[:, :, dot_start:dot_end, dot_start:dot_end, :]
        core_is_white = (np.abs(core - 255).max(axis=-1) <= 8).mean(axis=(2, 3)) > 0.75

        bg = np.array(self.bg_color, dtype=np.int16)
        corners = np.stack(
            [
                blocks[:, :, 0, 0, :],
                blocks[:, :, 0, -1, :],
                blocks[:, :, -1, 0, :],
                blocks[:, :, -1, -1, :],
            ],
            axis=2,
        )
        corners_are_bg = (np.abs(corners - bg).max(axis=-1) <= 8).mean(axis=2) > 0.75

        return self.active_mask_64 & core_is_white & corners_are_bg

    def _load_source_masks_64(self, stem, filename):
        if not self.data_dir:
            return None

        candidate_paths = [
            os.path.join(self.data_dir, stem + ext)
            for ext in [".png", ".webp", ".jpg", ".jpeg"]
        ]
        candidate_paths.append(os.path.join(self.data_dir, filename))

        skin_path = next((p for p in candidate_paths if os.path.exists(p)), None)
        if skin_path is None:
            return None

        skin_img = Image.open(skin_path).convert("RGBA").resize((64, 64), resample=Image.Resampling.NEAREST)
        alpha = np.array(skin_img)[..., 3]
        dot_mask = self.active_mask_64 & (alpha < self.alpha_threshold)
        opaque_mask = self.active_mask_64 & ~dot_mask
        return dot_mask, opaque_mask

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        stem, _ = os.path.splitext(filename)

        # 1. Load target image (512x512 target skin image)
        target_path = os.path.join(self.target_imgs_dir, stem + ".png")
        if not os.path.exists(target_path):
            target_path = os.path.join(self.target_imgs_dir, filename)

        if os.path.exists(target_path):
            target_img_raw = Image.open(target_path)
            target_img = target_img_raw.convert("RGB")
        else:
            target_img_raw = None
            target_img = Image.new("RGB", (self.target_width, self.target_height), self.bg_color)

        source_masks = self._load_source_masks_64(stem, filename)
        if source_masks is not None:
            dot_mask_np, opaque_mask_np = source_masks
        elif target_img_raw is not None:
            dot_mask_np = self._infer_dot_mask_64(target_img_raw)
            opaque_mask_np = self.active_mask_64 & ~dot_mask_np
        else:
            dot_mask_np = np.zeros_like(self.active_mask_64, dtype=bool)
            opaque_mask_np = self.active_mask_64.copy()

        target_img = target_img.resize((self.target_width, self.target_height), resample=self.target_resample)
        # Convert target image to tensor normalized to [-1, 1] for VAE encoding
        target_tensor = (transforms.ToTensor()(target_img) * 2.0) - 1.0
        active_mask_64 = torch.from_numpy(self.active_mask_64.astype(np.float32)).unsqueeze(0)
        dot_mask_64 = torch.from_numpy(dot_mask_np.astype(np.float32)).unsqueeze(0)
        opaque_mask_64 = torch.from_numpy(opaque_mask_np.astype(np.float32)).unsqueeze(0)

        # 2. Load conditioning control photo (front and back views)
        front_path = None
        back_path = None
        for ext in [".png", ".jpg", ".jpeg", ".webp"]:
            f_path = os.path.join(self.photos_dir, "front", stem + ext)
            b_path = os.path.join(self.photos_dir, "back", stem + ext)
            if os.path.exists(f_path) and os.path.exists(b_path):
                front_path = f_path
                back_path = b_path
                break

        if front_path is not None and back_path is not None:
            front_img = Image.open(front_path).convert("RGB")
            back_img = Image.open(back_path).convert("RGB")

            front_tensor = self.transform_cond(front_img)
            back_tensor = self.transform_cond(back_img)
            cond_tensor = torch.stack([front_tensor, back_tensor], dim=0) # (2, 3, target_height, target_width)
        else:
            # Single combined conditioning image (left half = front, right half = back)
            photo_path = None
            for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                temp_path = os.path.join(self.photos_dir, stem + ext)
                if os.path.exists(temp_path):
                    photo_path = temp_path
                    break

            if photo_path is not None:
                combined_img = Image.open(photo_path).convert("RGB")
            else:
                combined_img = Image.new("RGB", (self.target_width * 2, self.target_height), self.bg_color)

            w, h = combined_img.size
            front_img = combined_img.crop((0, 0, w // 2, h))
            back_img = combined_img.crop((w // 2, 0, w, h))

            front_tensor = self.transform_cond(front_img)
            back_tensor = self.transform_cond(back_img)
            cond_tensor = torch.stack([front_tensor, back_tensor], dim=0)

        # 3. Load prompt description if present
        caption_path = os.path.join(self.captions_dir, stem + ".txt")
        if os.path.exists(caption_path):
            with open(caption_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
        else:
            prompt = self.default_caption

        return {
            "target_latent_image": target_tensor,
            "cond_image": cond_tensor,
            "prompt": prompt,
            "active_mask_64": active_mask_64,
            "dot_mask_64": dot_mask_64,
            "opaque_mask_64": opaque_mask_64,
        }
