import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableRenderer(nn.Module):
    def __init__(self, mappings_dir=None, bg_color=(128/255, 128/255, 128/255)):
        """
        Differentiable Minecraft Skin Renderer in PyTorch.
        Args:
            mappings_dir: Path to directory containing '*.pt' view mapping files.
                          If None, it tries to find the default mappings directory.
            bg_color: RGB tuple for the render background.
        """
        super().__init__()
        if mappings_dir is None:
            # Try to resolve to differentiable_minecraft_renderer/mappings relative to workspace root
            # or local mappings subdirectory.
            local_mappings = os.path.join(os.path.dirname(__file__), "mappings")
            sibling_mappings = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "differentiable_minecraft_renderer", "mappings")
            )
            if os.path.exists(local_mappings):
                mappings_dir = local_mappings
            elif os.path.exists(sibling_mappings):
                mappings_dir = sibling_mappings
            else:
                mappings_dir = "mappings"  # fallback
            
        self.mappings_dir = mappings_dir
        self.register_buffer("bg_color", torch.tensor(bg_color, dtype=torch.float32))
        
        self.views = []
        
        # Load and pre-process mappings dynamically
        if os.path.exists(self.mappings_dir):
            for file_name in sorted(os.listdir(self.mappings_dir)):
                if file_name.endswith("_mapping.pt"):
                    view_name = file_name[:-11]  # Remove "_mapping.pt"
                    self.views.append(view_name)
                    
                    mapping_path = os.path.join(self.mappings_dir, file_name)
                    # Load mapping coordinates dictionary
                    data = torch.load(mapping_path, map_location="cpu")
                    
                    # Extract and process inner layer
                    inner_uv = data["inner_uv_map"]  # (H, W, 2)
                    inner_mask = data["inner_mask"]  # (H, W)
                    
                    # Extract and process outer layer
                    outer_uv = data["outer_uv_map"]  # (H, W, 2)
                    outer_mask = data["outer_mask"]  # (H, W)
                    
                    # Normalize UV coordinates [0, 63] to [-1, 1] for F.grid_sample
                    inner_grid = torch.zeros_like(inner_uv)
                    inner_grid[..., 0] = (inner_uv[..., 0] / 63.0) * 2.0 - 1.0
                    inner_grid[..., 1] = (inner_uv[..., 1] / 63.0) * 2.0 - 1.0
                    
                    outer_grid = torch.zeros_like(outer_uv)
                    outer_grid[..., 0] = (outer_uv[..., 0] / 63.0) * 2.0 - 1.0
                    outer_grid[..., 1] = (outer_uv[..., 1] / 63.0) * 2.0 - 1.0
                    
                    # Register buffer so they are automatically moved to device with the module
                    self.register_buffer(f"{view_name}_inner_grid", inner_grid)
                    self.register_buffer(f"{view_name}_inner_mask", inner_mask)
                    self.register_buffer(f"{view_name}_outer_grid", outer_grid)
                    self.register_buffer(f"{view_name}_outer_mask", outer_mask)
        else:
            print(f"WARNING: mappings_dir '{self.mappings_dir}' does not exist yet. Views list is empty.")

    def forward_view(self, skins, view_name):
        """
        Render a single view for a batch of skins.
        Args:
            skins: PyTorch tensor of shape (B, 4, 64, 64) with values in range [0, 1].
                   The channel order is RGBA.
            view_name: One of registered view names (e.g. "front", "back", "static_front").
        Returns:
            rendered: PyTorch tensor of shape (B, 4, H, W) containing RGBA render.
        """
        B, C, H_in, W_in = skins.shape
        assert C == 4, "Skins must have 4 channels (RGBA)"
        assert H_in == 64 and W_in == 64, "Skins must be 64x64"
        
        # Retrieve buffers and cast to match the input dtype (e.g. bfloat16 or float32)
        dtype = skins.dtype
        inner_grid = getattr(self, f"{view_name}_inner_grid").unsqueeze(0).expand(B, -1, -1, -1).to(dtype=dtype)
        inner_mask = getattr(self, f"{view_name}_inner_mask").unsqueeze(0).unsqueeze(1).expand(B, -1, -1, -1).to(dtype=dtype) # (B, 1, H, W)
        
        outer_grid = getattr(self, f"{view_name}_outer_grid").unsqueeze(0).expand(B, -1, -1, -1).to(dtype=dtype)
        outer_mask = getattr(self, f"{view_name}_outer_mask").unsqueeze(0).unsqueeze(1).expand(B, -1, -1, -1).to(dtype=dtype) # (B, 1, H, W)
        
        # 1. Sample inner layer using bilinear interpolation
        inner_sampled = F.grid_sample(skins, inner_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        inner_sampled = inner_sampled * inner_mask
        
        # 2. Sample outer layer using bilinear interpolation
        outer_sampled = F.grid_sample(skins, outer_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        outer_sampled = outer_sampled * outer_mask
        
        # 3. Alpha blend outer over inner
        inner_rgb = inner_sampled[:, :3, :, :]
        inner_alpha = inner_sampled[:, 3:4, :, :]
        
        outer_rgb = outer_sampled[:, :3, :, :]
        outer_alpha = outer_sampled[:, 3:4, :, :]
        
        # Prepare background color broadcasted to match dimensions
        bg = self.bg_color.view(1, 3, 1, 1).expand(B, -1, inner_rgb.shape[2], inner_rgb.shape[3]).to(dtype=dtype)
        
        # Composite inner layer over background
        inner_composite = inner_alpha * inner_rgb + (1.0 - inner_alpha) * bg
        
        # Composite outer layer over inner composite
        final_rgb = outer_alpha * outer_rgb + (1.0 - outer_alpha) * inner_composite
        
        # Calculate final alpha mask (for background/foreground separation)
        final_alpha = outer_alpha + (1.0 - outer_alpha) * inner_alpha
        
        return torch.cat([final_rgb, final_alpha], dim=1)

    def forward(self, skins):
        """
        Renders all loaded views for a batch of skins.
        Args:
            skins: PyTorch tensor of shape (B, 4, 64, 64) with values in range [0, 1].
        Returns:
            dict of rendered views. Each view has shape (B, 4, H, W).
        """
        results = {}
        for view_name in self.views:
            results[view_name] = self.forward_view(skins, view_name)
        return results
