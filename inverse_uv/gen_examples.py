import os
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image

class RenderAugmenter:
    def __init__(self, distortion_scale=0.08, perspective_scale=0.04, translation_scale=0.02, bg_color=(128, 128, 128)):
        self.distortion_scale = distortion_scale
        self.perspective_scale = perspective_scale
        self.translation_scale = translation_scale
        self.bg_color = bg_color
        
    def __call__(self, rendered_tensor):
        device = rendered_tensor.device
        dtype = rendered_tensor.dtype
        C, H, W = rendered_tensor.shape
        
        fill_color = [self.bg_color[0] / 255.0, self.bg_color[1] / 255.0, self.bg_color[2] / 255.0, 0.0]
        
        # 1. Random translation (offset)
        if self.translation_scale > 0:
            dx = int((torch.rand(1).item() - 0.5) * 2 * self.translation_scale * W)
            dy = int((torch.rand(1).item() - 0.5) * 2 * self.translation_scale * H)
            img_batch = rendered_tensor.unsqueeze(0)
            img_batch = TF.affine(
                img_batch, angle=0.0, translate=[dx, dy], scale=1.0, shear=[0.0, 0.0],
                interpolation=TF.InterpolationMode.BILINEAR,
                fill=fill_color
            )
            rendered_tensor = img_batch.squeeze(0)
            
        # 2. Perspective warp
        if self.perspective_scale > 0:
            startpoints = [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]]
            endpoints = []
            g = torch.Generator(device=device).manual_seed(42)
            for x, y in startpoints:
                dx = (torch.rand(1, generator=g).item() - 0.5) * 2 * self.perspective_scale * W
                dy = (torch.rand(1, generator=g).item() - 0.5) * 2 * self.perspective_scale * H
                endpoints.append([x + dx, y + dy])
            
            img_batch = rendered_tensor.unsqueeze(0)
            img_batch = TF.perspective(
                img_batch, startpoints, endpoints, 
                interpolation=TF.InterpolationMode.BILINEAR,
                fill=fill_color
            )
            rendered_tensor = img_batch.squeeze(0)
            
        # 3. Local Elastic / Grid distortion
        if self.distortion_scale > 0:
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1, 1, H, device=device, dtype=dtype),
                torch.linspace(-1, 1, W, device=device, dtype=dtype),
                indexing='ij'
            )
            
            g = torch.Generator(device=device).manual_seed(100)
            noise_h, noise_w = 8, 8
            disp_noise = torch.randn(1, 2, noise_h, noise_w, device=device, dtype=dtype, generator=g) * self.distortion_scale
            disp_field = F.interpolate(
                disp_noise, size=(H, W), 
                mode='bilinear', align_corners=True
            ).squeeze(0).permute(1, 2, 0)
            
            deformed_grid = torch.stack([grid_x, grid_y], dim=-1) + disp_field
            deformed_grid = deformed_grid.clamp(-1.0, 1.0).unsqueeze(0)
            
            img_batch = rendered_tensor.unsqueeze(0)
            img_batch = F.grid_sample(
                img_batch, deformed_grid, 
                mode='bilinear', padding_mode='border', align_corners=True
            )
            rendered_tensor = img_batch.squeeze(0)
            
        return rendered_tensor

def main():
    img_path = "../test_imgs/walk_perspective_ortho_pyvista.png"
    img = Image.open(img_path).convert("RGBA")
    tensor = TF.to_tensor(img)
    
    bg_color_rgba = img.getpixel((0, 0))
    bg_color = bg_color_rgba[:3]
    
    # 1. Original
    img_orig = tensor.clone()
    
    # 2. Translation only (shift) 2%
    aug_t1 = RenderAugmenter(perspective_scale=0.0, distortion_scale=0.0, translation_scale=0.02, bg_color=bg_color)
    img_t1 = aug_t1(tensor.clone())
    
    # 3. Translation only (shift) 5%
    aug_t2 = RenderAugmenter(perspective_scale=0.0, distortion_scale=0.0, translation_scale=0.05, bg_color=bg_color)
    img_t2 = aug_t2(tensor.clone())
    
    # 4. Combined: distortion 0.02, perspective 0.03, translation 0.02
    aug_comb = RenderAugmenter(perspective_scale=0.03, distortion_scale=0.02, translation_scale=0.02, bg_color=bg_color)
    img_comb = aug_comb(tensor.clone())
    
    # Combined side-by-side
    combined = torch.cat([img_orig, img_t1, img_t2, img_comb], dim=2)
    
    out_dir = "examples"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "milder_visual_examples.png")
    save_image(combined, out_path)
    print(f"Successfully saved milder combined image to {out_path}")

if __name__ == "__main__":
    main()
