import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def alice_to_steve(alice):
    """
    Expands 3px-wide Alex arms to 4px-wide Steve arms in-place on PIL Image.
    """
    # Create a copy so we do not mutate original directly
    img = alice.copy()
    for arm_loc, decor_offset in [((40, 16), (0, 16)), ((32, 48), (16, 0))]:
        for (size, loc, offset_x) in [
            # Top and bottom
            ((5, 4), (arm_loc[0] + 4 + 1, arm_loc[1]), 1),
            ((2, 4), (arm_loc[0] + 4 + 4 + 1, arm_loc[1]), 1),
            # Front and back
            ((9, 12), (arm_loc[0] + 4 + 1, arm_loc[1] + 4), 1),
            ((2, 12), (arm_loc[0] + 4 + 4 + 4 + 1, arm_loc[1] + 4), 1),
        ]:
            for x in range(loc[0] + size[0] - 1, loc[0] - 1, -1):
                for y in range(loc[1], loc[1] + size[1]):
                    img.putpixel((x + offset_x, y), img.getpixel((x, y)))
                    img.putpixel(
                        (x + offset_x + decor_offset[0], y + decor_offset[1]),
                        img.getpixel((x + decor_offset[0], y + decor_offset[1]))
                    )
    return img

def resolve_voxel_consistency(img):
    """
    Solves voxel edge transparent adjacent-face inconsistency.
    """
    is_slim = img.getpixel((47, 52))[3] == 0
    img = img.copy()
    
    parts = [
        # head
        [
            [
                [(8, 8, 8), (8, 8)],     # front
                [(8, 8, 8), (24, 8)],    # back
                [(8, 8, 8), (16, 8)],    # left
                [(8, 8, 8), (0, 8)],     # right
                [(8, 8, 8), (8, 0)],     # top
                [(8, 8, 8), (16, 0)],    # bottom
            ], (32, 0)
        ],
        # body
        [
            [
                [(8, 12, 4), (20, 20)],
                [(8, 12, 4), (20 + 12, 20)],
                [(4, 12, 8), (28, 20)],
                [(4, 12, 8), (16, 20)],
                [(8, 4, 12), (20, 16)],
                [(8, 4, 12), (20 + 8, 16)],
            ], (0, 16)
        ],
        # left arm
        [
            [
                [((3 if is_slim else 4), 12, 4), (32 + 4, 52)],
                [((3 if is_slim else 4), 12, 4), (32 + 12 - (1 if is_slim else 0), 52)],
                [(4, 12, 4), (32 + 8 - (1 if is_slim else 0), 52)],
                [(4, 12, 4), (32, 52)],
                [((3 if is_slim else 4), 4, 12), (32 + 4, 48)],
                [((3 if is_slim else 4), 4, 12), (32 + 8 - (1 if is_slim else 0), 48)],
            ], (16, 0)
        ],
        # right arm
        [
            [
                [((3 if is_slim else 4), 12, 4), (40 + 4, 20)],
                [((3 if is_slim else 4), 12, 4), (40 + 12 - (1 if is_slim else 0), 20)],
                [(4, 12, 4), (40 + 8 - (1 if is_slim else 0), 20)],
                [(4, 12, 4), (40, 20)],
                [((3 if is_slim else 4), 4, 12), (40 + 4, 16)],
                [((3 if is_slim else 4), 4, 12), (40 + 8 - (1 if is_slim else 0), 16)],
            ], (0, 16)
        ],
        # left leg
        [
            [
                [(4, 12, 4), (16 + 4, 52)],
                [(4, 12, 4), (16 + 12, 52)],
                [(4, 12, 4), (16 + 8, 52)],
                [(4, 12, 4), (16, 52)],
                [(4, 4, 12), (16 + 4, 48)],
                [(4, 4, 12), (16 + 8, 48)],
            ], (-16, 0)
        ],
        # right leg
        [
            [
                [(4, 12, 4), (0 + 4, 20)],
                [(4, 12, 4), (0 + 12, 20)],
                [(4, 12, 4), (0 + 8, 20)],
                [(4, 12, 4), (0, 20)],
                [(4, 4, 12), (0 + 4, 16)],
                [(4, 4, 12), (0 + 8, 16)],
            ], (0, 16)
        ],
    ]
    
    for part in parts:
        decor_offset = part[1]
        (x, y, z) = part[0][4][0]
        colors = np.zeros((x, y, z, 4))
        priorities = np.full((x, y, z), 99)
        inverse = {}
        
        for idx, (size, offset) in enumerate(part[0]):
            for dx in range(size[0]):
                for dy in range(size[1]):
                    img_x = offset[0] + dx + decor_offset[0]
                    img_y = offset[1] + dy + decor_offset[1]
                    c = img.getpixel((img_x, img_y))
                    
                    new_x = None
                    new_y = None
                    new_z = None
                    
                    if idx == 4:    # top
                        new_x, new_y, new_z = (dx, y - 1 - dy, z - 1)
                    elif idx == 5:  # bottom
                        new_x, new_y, new_z = (dx, y - 1 - dy, 0)
                    elif idx == 0:  # front
                        new_x, new_y, new_z = (dx, 0, z - 1 - dy)
                    elif idx == 1:  # back
                        new_x, new_y, new_z = (x - 1 - dx, y - 1, z - 1 - dy)
                    elif idx == 2:  # left
                        new_x, new_y, new_z = (x - 1, dx, z - 1 - dy)
                    elif idx == 3:  # right
                        new_x, new_y, new_z = (0, y - 1 - dx, z - 1 - dy)
                        
                    if (new_x, new_y, new_z) not in inverse:
                        inverse[(new_x, new_y, new_z)] = []
                    inverse[(new_x, new_y, new_z)].append((img_x, img_y))
                    
                    if c[3] == 0:
                        continue
                        
                    prio = 99
                    if idx == 0: prio = 0    # front
                    elif idx == 1: prio = 1  # back
                    elif idx == 4: prio = 2  # top
                    elif idx == 5: prio = 3  # bottom
                    elif idx == 2: prio = 4  # left
                    elif idx == 3: prio = 5  # right
                    
                    if priorities[new_x, new_y, new_z] > prio:
                        colors[new_x, new_y, new_z] = c
                        priorities[new_x, new_y, new_z] = prio
                        
        for dx in range(x):
            for dy in range(y):
                for dz in range(z):
                    if (dx, dy, dz) in inverse:
                        if priorities[dx, dy, dz] == 99:
                            continue
                        for i in inverse[(dx, dy, dz)]:
                            existing_c = img.getpixel(i)
                            if existing_c[3] == 0:
                                img.putpixel(i, tuple(colors[dx, dy, dz].astype(int)))
    return img

class MinecraftSkinDataset(Dataset):
    def __init__(
        self,
        data_dir,
        photos_dir=None,
        captions_dir=None,
        cond_size=1024,
        bg_color=(128, 128, 128),
        default_caption=""
    ):
        """
        PyTorch Dataset for Differentiable Minecraft Skin Fine-tuning.
        Args:
            data_dir: Path to skins folder containing target 64x64 skin PNGs.
            photos_dir: Path to conditioning photos folder. If None, falls back to data_dir/../control_imgs or data_dir.
            captions_dir: Path to captions folder. If None, falls back to data_dir.
            cond_size: Target/control image height. Width is cond_size // 2, matching ai-toolkit 512x1024 training.
            bg_color: Solid gray color (128,128,128) to paste RGB skin over.
            default_caption: Caption used when no .txt caption exists.
        """
        self.data_dir = data_dir
        self.photos_dir = photos_dir or os.path.abspath(os.path.join(data_dir, "..", "control_imgs"))
        if not os.path.exists(self.photos_dir):
            self.photos_dir = data_dir
            
        self.captions_dir = captions_dir or data_dir
        self.cond_size = cond_size
        self.target_height = cond_size
        self.target_width = cond_size // 2
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
            # Normalize conditioning images to [0, 1] or [-1, 1]. We output [0, 1] here
        ])

    def __len__(self):
        return len(self.skin_filenames)

    def __getitem__(self, idx):
        filename = self.skin_filenames[idx]
        stem, _ = os.path.splitext(filename)
        
        # 1. Load ground truth skin
        skin_path = os.path.join(self.data_dir, filename)
        skin = Image.open(skin_path).convert("RGBA")
        
        # Convert Alex to Steve if needed (slim check)
        is_slim = skin.getpixel((47, 52))[3] == 0
        if is_slim:
            skin = alice_to_steve(skin)
            
        # Resolve transparent voxel edge consistency
        skin = resolve_voxel_consistency(skin)
        
        # Opaque conversion check (standardizes all alpha channel to fully opaque or transparent)
        skin_np = np.array(skin)
        alpha = skin_np[..., 3]
        semi_transparent = (alpha > 0) & (alpha < 255)
        skin_np[semi_transparent, 3] = 255
        skin = Image.fromarray(skin_np)
        
        # 2. Build the RGB target on the same gray matte used by the VAE canvas.
        rgb_part = Image.new("RGB", (64, 64), self.bg_color)
        rgb_part.paste(skin, (0, 0), skin)

        # 3. Extract ground truth skin tensor (B x 4 x 64 x 64, normalized to [0, 1]).
        # Minecraft ignores RGB where alpha is 0, so store those pixels on the
        # same gray matte as the VAE target instead of supervising arbitrary PNG RGB.
        gt_rgba_np = np.array(skin, dtype=np.uint8)
        rgb_np = np.array(rgb_part, dtype=np.uint8)
        transparent = gt_rgba_np[..., 3] == 0
        gt_rgba_np[transparent, :3] = rgb_np[transparent]
        gt_skin_tensor = torch.tensor(gt_rgba_np.astype(np.float32) / 255.0).permute(2, 0, 1) # (4, 64, 64)
        
        # 4. Build [RGB | Alpha] top-to-bottom composite VAE target (512x1024 by default).
        
        # Alpha Part: Extract alpha channel as RGB grayscale
        alpha_part = skin.split()[3].convert("RGB")
        
        part_size = self.target_width
        rgb_part_upscaled = rgb_part.resize((part_size, part_size), resample=Image.Resampling.BOX)
        alpha_part_upscaled = alpha_part.resize((part_size, part_size), resample=Image.Resampling.BOX)
        
        # Place top-to-bottom into a 512x1024 image by default (no blank padding).
        target_img = Image.new("RGB", (self.target_width, self.target_height), self.bg_color)
        target_img.paste(rgb_part_upscaled, (0, 0))       # Top half
        target_img.paste(alpha_part_upscaled, (0, part_size))   # Bottom half
        
        # Convert VAE target to tensor and normalize to [-1, 1] (standard for VAE latents encoding)
        target_tensor = (transforms.ToTensor()(target_img) * 2.0) - 1.0
        
        # 4. Load conditioning photo
        # Look for front and back images first (each resized to target_width x target_height)
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
            # Fallback to single combined conditioning image
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
                
            # Split the combined image into left half (front) and right half (back)
            w, h = combined_img.size
            front_img = combined_img.crop((0, 0, w // 2, h))
            back_img = combined_img.crop((w // 2, 0, w, h))
            
            front_tensor = self.transform_cond(front_img)
            back_tensor = self.transform_cond(back_img)
            cond_tensor = torch.stack([front_tensor, back_tensor], dim=0) # (2, 3, target_height, target_width)
        
        # 5. Load prompt description
        caption_path = os.path.join(self.captions_dir, stem + ".txt")
        if os.path.exists(caption_path):
            with open(caption_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
        else:
            prompt = self.default_caption
            
        return {
            "target_latent_image": target_tensor,  # VAE target, normalized to [-1, 1]
            "cond_image": cond_tensor,            # Control image, normalized to [0, 1]
            "prompt": prompt,                      # Prompt description string
            "gt_skin": gt_skin_tensor             # GT skin UV, normalized to [0, 1] (4, 64, 64)
        }
