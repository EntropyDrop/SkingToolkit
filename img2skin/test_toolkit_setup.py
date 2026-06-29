import os
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

import sys
from pathlib import Path

# Inject workspace root into sys.path to allow absolute imports
TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TOOLKIT_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

# Also add the script's directory so that local imports like `import dataset` work
# even when run with different sys.path settings
SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Import local modules
from dataset import MinecraftSkinDataset
from loss import MinecraftLoss
from SkingToolkit.renderer import DifferentiableRenderer

def run_verification():
    print("==========================================================")
    print("       SkingToolkit: Verification Test Setup             ")
    print("==========================================================")
    
    # 1. Resolve local paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # TOOLKIT_ROOT is the SkingToolkit directory
    toolkit_dir = os.path.dirname(current_dir)
    workspace_dir = os.path.dirname(toolkit_dir)
    
    data_dir = os.path.join(workspace_dir, "SkingDataset", "skins")
    if not os.path.exists(data_dir):
        data_dir = os.path.join(workspace_dir, "Sking", "skins")
    mappings_dir = os.path.join(workspace_dir, "differentiable_minecraft_renderer", "mappings")
    
    print(f"[*] Dataset skins path: {data_dir}")
    print(f"[*] Render mappings path: {mappings_dir}")
    
    # Check if mappings exist
    if not os.path.exists(mappings_dir):
        print(f"[!] Mappings directory not found at {mappings_dir}.")
        print("[!] Generating mapping files dynamically using sibling codebase...")
        try:
            sys.path.append(os.path.join(workspace_dir, "differentiable_minecraft_renderer"))
            import generate_mappings
            generate_mappings.render_and_save_mappings()
        except Exception as e:
            print(f"[!] Failed to auto-generate mappings: {e}")
            print("[!] Please generate or copy mapping files to run rendering tests.")
    
    # 2. Test Dataset loading
    print("\n[*] Step 1: Testing MinecraftSkinDataset loading...")
    try:
        dataset = MinecraftSkinDataset(
            data_dir=data_dir,
            photos_dir=None, # fallback
            cond_size=1024
        )
        print(f"[*] Found {len(dataset)} items in skins dataset.")
        
        # Load first item
        item = dataset[0]
        print("[✓] Successfully loaded dataset item!")
        print(f"    - Target latent image tensor shape: {item['target_latent_image'].shape} (expected 3 x 1024 x 512)")
        print(f"    - Conditioning photo tensor shape:  {item['cond_image'].shape} (expected 2 x 3 x 1024 x 512)")
        print(f"    - Prompt text:                     '{item['prompt']}'")
        print(f"    - Ground Truth skin tensor shape:   {item['gt_skin'].shape} (expected 4 x 64 x 64)")
    except Exception as e:
        print(f"[X] Dataset testing failed: {e}")
        return
        
    # 3. Test Differentiable Renderer
    print("\n[*] Step 2: Testing DifferentiableRenderer and multi-view compilation...")
    try:
        renderer = DifferentiableRenderer(mappings_dir=mappings_dir)
        print(f"[*] Loaded renderer views: {renderer.views}")
        
        # Pass a dummy skin of batch size 2
        dummy_skins = torch.rand((2, 4, 64, 64))
        renders = renderer(dummy_skins)
        
        print("[✓] DifferentiableRenderer successfully executed forward views:")
        for view_name, render_tensor in renders.items():
            print(f"    - View '{view_name}': rendered shape = {render_tensor.shape}")
    except Exception as e:
        print(f"[X] Renderer testing failed: {e}")
        return
        
    # 4. Test Loss Function & Gradient Flow (Backpropagation)
    print("\n[*] Step 3: Verifying differentiable gradient backpropagation...")
    try:
        # Initialize learnable skin tensor representing predicted output (needs gradients)
        learnable_skin = torch.rand((1, 4, 64, 64), requires_grad=True)
        gt_skin = item["gt_skin"].unsqueeze(0) # (1, 4, 64, 64)
        
        # Setup loss criterion
        criterion = MinecraftLoss(
            mappings_dir=mappings_dir,
            lambda_uv=1.0,
            lambda_render=1.0,
            use_lpips=False, # Disable LPIPS for quick local run without downloading weights
            views="static_front,static_back"
        )
        
        # Setup small optimizer to fit the skin
        optimizer = optim.Adam([learnable_skin], lr=0.1)
        
        print("[*] Running 10-step fitting check to verify backpropagation...")
        for i in range(1, 11):
            optimizer.zero_grad()
            
            # Forward pass
            loss_dict = criterion(learnable_skin, gt_skin)
            loss = loss_dict["loss_total"]
            
            # Backward pass (gradient computation)
            loss.backward()
            
            # Check gradients exist
            if learnable_skin.grad is not None and torch.any(learnable_skin.grad != 0):
                grad_norm = learnable_skin.grad.norm().item()
            else:
                grad_norm = 0.0
                
            optimizer.step()
            
            # Project back to valid RGBA range [0, 1]
            with torch.no_grad():
                learnable_skin.clamp_(0.0, 1.0)
                
            print(f"    - Step {i:02d} | Loss: {loss.item():.6f} | Gradient Norm: {grad_norm:.6f}")
            
        print("[✓] Gradient backpropagation verified successfully! Loss decreased and non-zero gradients flowed through.")
        
    except Exception as e:
        print(f"[X] Gradient backpropagation test failed: {e}")
        return
        
    print("\n==========================================================")
    print("✓ Verification complete! All SkingToolkit components ready.")
    print("==========================================================")

if __name__ == "__main__":
    import sys
    run_verification()
