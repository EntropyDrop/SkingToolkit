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
            env_mappings_dir = os.environ.get("RENDERER_MAPPINGS_DIR")
            if env_mappings_dir and os.path.exists(env_mappings_dir):
                mappings_dir = env_mappings_dir
            else:
                try:
                    import sys
                    dmr_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "differentiable_minecraft_renderer"))
                    if dmr_dir not in sys.path:
                        sys.path.insert(0, dmr_dir)
                    from config import size as config_size
                    size_suffix = f"_{config_size[0]}x{config_size[1]}"
                except Exception:
                    size_suffix = ""

                local_size_mappings = os.path.join(os.path.dirname(__file__), f"mappings{size_suffix}")
                local_mappings = os.path.join(os.path.dirname(__file__), "mappings")
                sibling_size_mappings = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "differentiable_minecraft_renderer", f"mappings{size_suffix}")
                )
                sibling_mappings = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "differentiable_minecraft_renderer", "mappings")
                )
                if os.path.exists(local_size_mappings):
                    mappings_dir = local_size_mappings
                elif os.path.exists(local_mappings):
                    mappings_dir = local_mappings
                elif os.path.exists(sibling_size_mappings):
                    mappings_dir = sibling_size_mappings
                elif os.path.exists(sibling_mappings):
                    mappings_dir = sibling_mappings
                else:
                    mappings_dir = "mappings"
            
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

                    if "outer_uv_layers" in data and "outer_masks" in data:
                        outer_uv_layers = data["outer_uv_layers"].float()
                        outer_mask_layers = data["outer_masks"].float()
                        outer_grid_layers = torch.zeros_like(outer_uv_layers)
                        outer_grid_layers[..., 0] = (outer_uv_layers[..., 0] / 63.0) * 2.0 - 1.0
                        outer_grid_layers[..., 1] = (outer_uv_layers[..., 1] / 63.0) * 2.0 - 1.0
                        self.register_buffer(f"{view_name}_outer_grid_layers", outer_grid_layers)
                        self.register_buffer(f"{view_name}_outer_mask_layers", outer_mask_layers)

                    if "composite_uv_layers" in data and "composite_masks" in data:
                        composite_uv_layers = data["composite_uv_layers"].float()
                        composite_mask_layers = data["composite_masks"].float()
                        composite_grid_layers = torch.zeros_like(composite_uv_layers)
                        composite_grid_layers[..., 0] = (composite_uv_layers[..., 0] / 63.0) * 2.0 - 1.0
                        composite_grid_layers[..., 1] = (composite_uv_layers[..., 1] / 63.0) * 2.0 - 1.0
                        self.register_buffer(f"{view_name}_composite_grid_layers", composite_grid_layers)
                        self.register_buffer(f"{view_name}_composite_mask_layers", composite_mask_layers)
                        if "composite_is_decor_layers" in data:
                            self.register_buffer(
                                f"{view_name}_composite_is_decor_layers",
                                data["composite_is_decor_layers"].bool(),
                            )

                    if (
                        "geometry_uv_layers" in data
                        and "geometry_masks" in data
                        and ("geometry_sort_indices" in data or "geometry_depth_layers" in data)
                    ):
                        geometry_uv_layers = data["geometry_uv_layers"].float()
                        geometry_mask_layers = data["geometry_masks"].float()
                        geometry_grid_layers = torch.zeros_like(geometry_uv_layers)
                        geometry_grid_layers[..., 0] = (geometry_uv_layers[..., 0] / 63.0) * 2.0 - 1.0
                        geometry_grid_layers[..., 1] = (geometry_uv_layers[..., 1] / 63.0) * 2.0 - 1.0
                        if "geometry_sort_indices" in data:
                            geometry_sort_indices = data["geometry_sort_indices"].long()
                        else:
                            geometry_depth_layers = data["geometry_depth_layers"].float()
                            geometry_sort_indices = torch.argsort(geometry_depth_layers, dim=0)
                        self.register_buffer(f"{view_name}_geometry_grid_layers", geometry_grid_layers)
                        self.register_buffer(f"{view_name}_geometry_mask_layers", geometry_mask_layers)
                        self.register_buffer(f"{view_name}_geometry_sort_indices", geometry_sort_indices)
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

        composite_grid_name = f"{view_name}_composite_grid_layers"
        composite_decor_name = f"{view_name}_composite_is_decor_layers"
        if hasattr(self, composite_grid_name) and hasattr(self, composite_decor_name):
            composite_grid_layers = getattr(self, composite_grid_name).to(dtype=dtype)
            composite_mask_layers = getattr(self, f"{view_name}_composite_mask_layers").to(dtype=dtype)
            composite_is_decor_layers = getattr(self, composite_decor_name)
            num_layers = composite_grid_layers.shape[0]
            out_h, out_w = composite_grid_layers.shape[1:3]

            composite_grid = composite_grid_layers.unsqueeze(0).expand(B, -1, -1, -1, -1)
            composite_grid = composite_grid.reshape(B * num_layers, out_h, out_w, 2)
            skin_layers = skins.unsqueeze(1).expand(-1, num_layers, -1, -1, -1)
            skin_layers = skin_layers.reshape(B * num_layers, C, H_in, W_in)
            composite_sampled = F.grid_sample(
                skin_layers,
                composite_grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True,
            )
            composite_sampled = composite_sampled.reshape(B, num_layers, C, out_h, out_w)
            composite_mask = composite_mask_layers.unsqueeze(0).unsqueeze(2).expand(B, -1, -1, -1, -1)
            composite_sampled = composite_sampled * composite_mask

            composite_render_rgb = bg
            composite_render_alpha = torch.zeros_like(final_alpha)
            for layer_index in range(num_layers - 1, -1, -1):
                layer = composite_sampled[:, layer_index]
                layer_rgb = layer[:, :3]
                layer_alpha = layer[:, 3:4]
                composite_render_rgb = layer_alpha * layer_rgb + (1.0 - layer_alpha) * composite_render_rgb
                composite_render_alpha = layer_alpha + (1.0 - layer_alpha) * composite_render_alpha

            layer_alpha = composite_sampled[:, :, 3:4]
            visible = (layer_alpha > 1e-4) & (composite_mask > 0.5)
            layer_indices = torch.arange(num_layers, device=skins.device).view(1, num_layers, 1, 1, 1)
            fallback_indices = torch.full(visible.shape, num_layers, device=skins.device, dtype=torch.long)
            visible_indices = torch.where(visible, layer_indices, fallback_indices)
            first_visible_index = visible_indices.amin(dim=1)
            has_visible = first_visible_index < num_layers

            decor_layers = composite_is_decor_layers.unsqueeze(0).unsqueeze(2).expand(B, -1, -1, -1, -1)
            first_layer = layer_indices == first_visible_index.unsqueeze(1)
            first_visible_is_decor = (visible & first_layer & decor_layers).any(dim=1)
            trust_composite = has_visible & (first_visible_is_decor | (first_visible_index <= 1))

            final_rgb = torch.where(trust_composite, composite_render_rgb, final_rgb)
            final_alpha = torch.where(trust_composite, composite_render_alpha, final_alpha)

            geometry_grid_name = f"{view_name}_geometry_grid_layers"
            if hasattr(self, geometry_grid_name) and num_layers > 0:
                geometry_grid_layers = getattr(self, geometry_grid_name).to(dtype=dtype)
                geometry_mask_layers = getattr(self, f"{view_name}_geometry_mask_layers").to(dtype=dtype)
                geometry_sort_indices = getattr(self, f"{view_name}_geometry_sort_indices").to(device=skins.device)
                geometry_layers = geometry_grid_layers.shape[0]
                if geometry_layers > 0:
                    geometry_grid = geometry_grid_layers.unsqueeze(0).expand(B, -1, -1, -1, -1)
                    geometry_grid = geometry_grid.reshape(B * geometry_layers, out_h, out_w, 2)
                    geometry_skin_layers = skins.unsqueeze(1).expand(-1, geometry_layers, -1, -1, -1)
                    geometry_skin_layers = geometry_skin_layers.reshape(B * geometry_layers, C, H_in, W_in)
                    geometry_sampled = F.grid_sample(
                        geometry_skin_layers,
                        geometry_grid,
                        mode='bilinear',
                        padding_mode='zeros',
                        align_corners=True,
                    )
                    geometry_sampled = geometry_sampled.reshape(B, geometry_layers, C, out_h, out_w)
                    geometry_mask = geometry_mask_layers.unsqueeze(0).unsqueeze(2).expand(
                        B, -1, -1, -1, -1
                    )
                    geometry_sampled = geometry_sampled * geometry_mask
                    geometry_order = geometry_sort_indices.unsqueeze(0).unsqueeze(2).expand(
                        B, -1, C, -1, -1
                    )
                    geometry_sampled = torch.gather(geometry_sampled, dim=1, index=geometry_order)

                    geometry_render_rgb = bg
                    geometry_render_alpha = torch.zeros_like(final_alpha)
                    for layer_index in range(geometry_layers):
                        layer = geometry_sampled[:, layer_index]
                        layer_rgb = layer[:, :3]
                        layer_alpha = layer[:, 3:4]
                        geometry_render_rgb = (
                            layer_alpha * layer_rgb + (1.0 - layer_alpha) * geometry_render_rgb
                        )
                        geometry_render_alpha = layer_alpha + (1.0 - layer_alpha) * geometry_render_alpha

                    front_decor = composite_is_decor_layers[0].unsqueeze(0).unsqueeze(1)
                    front_mask = (composite_mask_layers[0] > 0.5).unsqueeze(0).unsqueeze(1)
                    front_alpha = composite_sampled[:, 0, 3:4]
                    geometry_fallback = (
                        front_mask
                        & front_decor
                        & (front_alpha <= 1e-4)
                        & (geometry_render_alpha > 1e-4)
                    )
                    final_rgb = torch.where(geometry_fallback, geometry_render_rgb, final_rgb)
                    final_alpha = torch.where(geometry_fallback, geometry_render_alpha, final_alpha)
        
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
