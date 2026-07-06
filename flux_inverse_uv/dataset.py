import os
import sys
from pathlib import Path

# Inject workspace root and img2skin path into sys.path
FLUX_INVERSE_UV_DIR = Path(__file__).resolve().parent
TOOLKIT_ROOT = FLUX_INVERSE_UV_DIR.parent
WORKSPACE_ROOT = TOOLKIT_ROOT.parent
IMG2SKIN_DIR = TOOLKIT_ROOT / "img2skin"

for p in [str(WORKSPACE_ROOT), str(TOOLKIT_ROOT), str(IMG2SKIN_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from mc_skin_utils.alice_to_steve import alice_to_steve

class FluxInverseUVDataset(Dataset):
    def __init__(
        self,
        data_dir,
        photos_dir=None,
        target_imgs_dir=None,
        captions_dir=None,
        cond_size=1024,
        is_square=True,
        bg_color=(128, 128, 128),
        default_caption=""
    ):
        """
        PyTorch Dataset for Flux Inverse UV Fine-tuning.
        Args:
            data_dir: Path to skins folder containing target 64x64 skin PNGs.
            photos_dir: Path to conditioning control_imgs folder.
            target_imgs_dir: Path to pre-built target_imgs folder (512x512 target skin images).
            captions_dir: Path to captions folder.
            cond_size: Target/control image height.
            is_square: If True, target_width == cond_size (e.g. 512x512). Otherwise cond_size // 2.
            bg_color: Solid gray color (128,128,128) for matte background.
            default_caption: Caption used when no .txt caption exists.
        """
        self.data_dir = data_dir
        self.photos_dir = photos_dir or str(FLUX_INVERSE_UV_DIR / "control_imgs")
        if not os.path.exists(self.photos_dir):
            self.photos_dir = data_dir

        self.target_imgs_dir = target_imgs_dir or str(FLUX_INVERSE_UV_DIR / "target_imgs")
        self.captions_dir = captions_dir or data_dir
        self.cond_size = cond_size
        self.is_square = is_square
        self.target_height = cond_size
        self.target_width = cond_size if is_square else cond_size // 2
        self.bg_color = bg_color
        self.default_caption = default_caption

        # Scan skin PNGs
        self.skin_filenames = sorted([
            f for f in os.listdir(self.data_dir)
            if f.endswith(".png") and not f.startswith("half_")
        ])

        if len(self.skin_filenames) == 0:
            print(f"WARNING: No skin PNG files found in data_dir: {self.data_dir}")

        self.transform_cond = transforms.Compose([
            transforms.Resize((self.target_height, self.target_width), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.skin_filenames)

    def __getitem__(self, idx):
        filename = self.skin_filenames[idx]
        stem, _ = os.path.splitext(filename)

        # 1. Load ground truth 64x64 skin
        skin_path = os.path.join(self.data_dir, filename)
        skin = Image.open(skin_path).convert("RGBA")

        # Convert Alex to Steve if needed (slim check)
        is_slim = skin.getpixel((47, 52))[3] == 0
        if is_slim:
            skin = alice_to_steve(skin)

        # Opaque conversion check
        skin_np = np.array(skin)
        alpha = skin_np[..., 3]
        semi_transparent = (alpha > 0) & (alpha < 255)
        skin_np[semi_transparent, 3] = 255
        skin = Image.fromarray(skin_np)

        # Build RGB part on gray background
        rgb_part = Image.new("RGB", (64, 64), self.bg_color)
        rgb_part.paste(skin, (0, 0), skin)

        # Extract GT skin tensor (B x 4 x 64 x 64, normalized to [0, 1])
        gt_rgba_np = np.array(skin, dtype=np.uint8)
        rgb_np = np.array(rgb_part, dtype=np.uint8)
        transparent = gt_rgba_np[..., 3] == 0
        gt_rgba_np[transparent, :3] = rgb_np[transparent]
        gt_skin_tensor = torch.tensor(gt_rgba_np.astype(np.float32) / 255.0).permute(2, 0, 1)

        # 2. Build or load target image for VAE encoding
        target_file_path = os.path.join(self.target_imgs_dir, filename) if self.target_imgs_dir else None
        if target_file_path and os.path.exists(target_file_path):
            # Load pre-built target image from target_imgs_dir
            target_img = Image.open(target_file_path).convert("RGB")
            target_img = target_img.resize((self.target_width, self.target_height), resample=Image.Resampling.LANCZOS)
        else:
            # Fallback: Build [RGB | Alpha] top-to-bottom composite target image
            alpha_part = skin.split()[3].convert("RGB")
            part_size = self.target_width
            rgb_upscaled = rgb_part.resize((part_size, part_size), resample=Image.Resampling.BOX)
            alpha_upscaled = alpha_part.resize((part_size, part_size), resample=Image.Resampling.BOX)

            target_img = Image.new("RGB", (self.target_width, self.target_height), self.bg_color)
            target_img.paste(rgb_upscaled, (0, 0))
            target_img.paste(alpha_upscaled, (0, part_size))

        # Convert target image to tensor normalized to [-1, 1] for VAE encoding
        target_tensor = (transforms.ToTensor()(target_img) * 2.0) - 1.0

        # 3. Load conditioning control photo (front and back views)
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

        # 4. Load prompt description if present
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
            "gt_skin": gt_skin_tensor
        }
