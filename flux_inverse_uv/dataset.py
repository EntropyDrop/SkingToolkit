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
        photos_dir=None,
        target_imgs_dir=None,
        data_dir=None,
        captions_dir=None,
        cond_size=512,
        is_square=True,
        bg_color=(128, 128, 128),
        default_caption=""
    ):
        """
        PyTorch Dataset for Flux Inverse UV Fine-tuning.
        Args:
            photos_dir: Path to conditioning control_imgs folder.
            target_imgs_dir: Path to pre-built target_imgs folder (512x512 target skin images).
            data_dir: Optional path to skins folder containing 64x64 skin PNGs.
            captions_dir: Optional path to captions folder.
            cond_size: Target/control image resolution (e.g. 512).
            is_square: If True, target_width == cond_size (512x512). Otherwise cond_size // 2.
            bg_color: Solid gray color (128,128,128) for matte background.
            default_caption: Caption used when no .txt caption exists.
        """
        self.photos_dir = photos_dir or str(FLUX_INVERSE_UV_DIR / "control_imgs")
        self.target_imgs_dir = target_imgs_dir or str(FLUX_INVERSE_UV_DIR / "target_imgs")
        self.data_dir = data_dir
        self.captions_dir = captions_dir or self.photos_dir
        self.cond_size = cond_size
        self.is_square = is_square
        self.target_height = cond_size
        self.target_width = cond_size if is_square else cond_size // 2
        self.bg_color = bg_color
        self.default_caption = default_caption

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
            target_img = Image.open(target_path).convert("RGB")
        else:
            target_img = Image.new("RGB", (self.target_width, self.target_height), self.bg_color)

        target_img = target_img.resize((self.target_width, self.target_height), resample=Image.Resampling.LANCZOS)
        # Convert target image to tensor normalized to [-1, 1] for VAE encoding
        target_tensor = (transforms.ToTensor()(target_img) * 2.0) - 1.0

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
            "prompt": prompt
        }
