import torch
import numpy as np
from PIL import Image
import os
from dataset import MinecraftSkinDataset

def main():
    print("Initializing dataset...")
    dataset = MinecraftSkinDataset(
        data_dir="../SkingDataset/skins",
        photos_dir="../SkingDataset/control_imgs",
        cond_size=512
    )
    
    if len(dataset) == 0:
        print("Error: Dataset is empty.")
        return

    output_dir = "debug_target"
    os.makedirs(output_dir, exist_ok=True)
    
    # Export the first 5 items to show variety
    num_exports = min(5, len(dataset))
    print(f"Exporting the first {num_exports} samples...")
    
    for i in range(num_exports):
        batch = dataset[i]
        target_latent_image = batch["target_latent_image"]
        cond_image = batch["cond_image"]
        gt_skin_tensor = batch["gt_skin"]
        prompt = batch["prompt"]
        
        # target_latent_image shape: (3, 512, 256) in [-1, 1]
        # Denormalize from [-1, 1] back to [0, 1]
        target_tensor_norm = (target_latent_image + 1.0) / 2.0
        
        # Convert back to numpy array (H, W, C) in [0, 255]
        img_np = (target_tensor_norm.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        
        # Convert to PIL Image
        img_pil = Image.fromarray(img_np, mode="RGB")
        
        # Save the debug image
        output_path = os.path.join(output_dir, f"{i:03d}.png")
        img_pil.save(output_path)
        
        print(f"[{i+1}/{num_exports}] Saved {output_path} | Prompt: {prompt}")

if __name__ == "__main__":
    main()
