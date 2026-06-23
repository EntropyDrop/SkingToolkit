import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm
from PIL import Image

# Import local modules
from dataset import MinecraftSkinDataset
from loss import MinecraftLoss
from flux2_src.model import Flux2, Klein4BParams, Klein9BParams
from flux2_src.autoencoder import AutoEncoder, AutoEncoderParams, AutoEncoderSmallDecoderParams
from flux2_src.sampling import batched_prc_img, batched_prc_txt, scatter_ids, encode_image_refs

# Import diffusers and transformers
from diffusers import (
    FluxTransformer2DModel,
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

def convert_diffusers_vae_to_custom(state_dict):
    # If the state dict is already in custom format, return as is
    if "encoder.quant_conv.weight" in state_dict:
        return state_dict

    # Check if this is indeed a diffusers format VAE
    is_diffusers = any(k.startswith("encoder.down_blocks") for k in state_dict.keys())
    if not is_diffusers:
        return state_dict

    custom_state_dict = {}

    # 1. Direct key mappings
    direct_mappings = {
        "quant_conv.weight": "encoder.quant_conv.weight",
        "quant_conv.bias": "encoder.quant_conv.bias",
        "post_quant_conv.weight": "decoder.post_quant_conv.weight",
        "post_quant_conv.bias": "decoder.post_quant_conv.bias",
        
        "encoder.conv_in.weight": "encoder.conv_in.weight",
        "encoder.conv_in.bias": "encoder.conv_in.bias",
        "encoder.conv_out.weight": "encoder.conv_out.weight",
        "encoder.conv_out.bias": "encoder.conv_out.bias",
        "encoder.conv_norm_out.weight": "encoder.norm_out.weight",
        "encoder.conv_norm_out.bias": "encoder.norm_out.bias",
        
        "decoder.conv_in.weight": "decoder.conv_in.weight",
        "decoder.conv_in.bias": "decoder.conv_in.bias",
        "decoder.conv_out.weight": "decoder.conv_out.weight",
        "decoder.conv_out.bias": "decoder.conv_out.bias",
        "decoder.conv_norm_out.weight": "decoder.norm_out.weight",
        "decoder.conv_norm_out.bias": "decoder.norm_out.bias",
    }
    
    for k, v in direct_mappings.items():
        if k in state_dict:
            custom_state_dict[v] = state_dict[k]

    # 2. Encoder down blocks
    for i in range(4): # 4 down blocks
        # Downsamplers
        old_ds_w = f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"
        old_ds_b = f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"
        if old_ds_w in state_dict:
            custom_state_dict[f"encoder.down.{i}.downsample.conv.weight"] = state_dict[old_ds_w]
        if old_ds_b in state_dict:
            custom_state_dict[f"encoder.down.{i}.downsample.conv.bias"] = state_dict[old_ds_b]

        # Resnets (2 resnets per down block)
        for j in range(2):
            for layer in ["norm1", "norm2", "conv1", "conv2", "conv_shortcut"]:
                custom_layer = "nin_shortcut" if layer == "conv_shortcut" else layer
                old_key = f"encoder.down_blocks.{i}.resnets.{j}.{layer}.weight"
                new_key = f"encoder.down.{i}.block.{j}.{custom_layer}.weight"
                if old_key in state_dict:
                    custom_state_dict[new_key] = state_dict[old_key]
                
                old_bias = f"encoder.down_blocks.{i}.resnets.{j}.{layer}.bias"
                new_bias = f"encoder.down.{i}.block.{j}.{custom_layer}.bias"
                if old_bias in state_dict:
                    custom_state_dict[new_bias] = state_dict[old_bias]

    # 3. Encoder mid block
    for j in range(2): # 2 resnets in mid_block
        for layer in ["norm1", "norm2", "conv1", "conv2", "conv_shortcut"]:
            custom_layer = "nin_shortcut" if layer == "conv_shortcut" else layer
            old_key = f"encoder.mid_block.resnets.{j}.{layer}.weight"
            new_key = f"encoder.mid.block_{j+1}.{custom_layer}.weight"
            if old_key in state_dict:
                custom_state_dict[new_key] = state_dict[old_key]
            old_bias = f"encoder.mid_block.resnets.{j}.{layer}.bias"
            new_bias = f"encoder.mid.block_{j+1}.{custom_layer}.bias"
            if old_bias in state_dict:
                custom_state_dict[new_bias] = state_dict[old_bias]

    # Encoder mid block Attention
    attn_mappings = {
        "group_norm": "norm",
        "to_q": "q",
        "to_k": "k",
        "to_v": "v",
        "to_out.0": "proj_out"
    }
    for old_name, new_name in attn_mappings.items():
        for suffix in ["weight", "bias"]:
            old_key = f"encoder.mid_block.attentions.0.{old_name}.{suffix}"
            new_key = f"encoder.mid.attn_1.{new_name}.{suffix}"
            if old_key in state_dict:
                val = state_dict[old_key]
                if suffix == "weight" and old_name in ["to_q", "to_k", "to_v", "to_out.0"]:
                    if val.ndim == 2:
                        val = val.unsqueeze(-1).unsqueeze(-1)
                custom_state_dict[new_key] = val

    # 4. Decoder up blocks
    for i in range(4): # 4 up blocks in diffusers (0 to 3)
        custom_i = 3 - i
        # Upsamplers (exist in up_blocks 0, 1, 2 for diffusers)
        old_us_w = f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"
        old_us_b = f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"
        if old_us_w in state_dict:
            custom_state_dict[f"decoder.up.{custom_i}.upsample.conv.weight"] = state_dict[old_us_w]
        if old_us_b in state_dict:
            custom_state_dict[f"decoder.up.{custom_i}.upsample.conv.bias"] = state_dict[old_us_b]

        # Resnets (3 resnets per up block)
        for j in range(3):
            for layer in ["norm1", "norm2", "conv1", "conv2", "conv_shortcut"]:
                custom_layer = "nin_shortcut" if layer == "conv_shortcut" else layer
                old_key = f"decoder.up_blocks.{i}.resnets.{j}.{layer}.weight"
                new_key = f"decoder.up.{custom_i}.block.{j}.{custom_layer}.weight"
                if old_key in state_dict:
                    custom_state_dict[new_key] = state_dict[old_key]
                
                old_bias = f"decoder.up_blocks.{i}.resnets.{j}.{layer}.bias"
                new_bias = f"decoder.up.{custom_i}.block.{j}.{custom_layer}.bias"
                if old_bias in state_dict:
                    custom_state_dict[new_bias] = state_dict[old_bias]

    # 5. Decoder mid block
    for j in range(2): # 2 resnets in mid_block
        for layer in ["norm1", "norm2", "conv1", "conv2", "conv_shortcut"]:
            custom_layer = "nin_shortcut" if layer == "conv_shortcut" else layer
            old_key = f"decoder.mid_block.resnets.{j}.{layer}.weight"
            new_key = f"decoder.mid.block_{j+1}.{custom_layer}.weight"
            if old_key in state_dict:
                custom_state_dict[new_key] = state_dict[old_key]
            old_bias = f"decoder.mid_block.resnets.{j}.{layer}.bias"
            new_bias = f"decoder.mid.block_{j+1}.{custom_layer}.bias"
            if old_bias in state_dict:
                custom_state_dict[new_bias] = state_dict[old_bias]

    # Decoder mid block Attention
    for old_name, new_name in attn_mappings.items():
        for suffix in ["weight", "bias"]:
            old_key = f"decoder.mid_block.attentions.0.{old_name}.{suffix}"
            new_key = f"decoder.mid.attn_1.{new_name}.{suffix}"
            if old_key in state_dict:
                val = state_dict[old_key]
                if suffix == "weight" and old_name in ["to_q", "to_k", "to_v", "to_out.0"]:
                    if val.ndim == 2:
                        val = val.unsqueeze(-1).unsqueeze(-1)
                custom_state_dict[new_key] = val

    return custom_state_dict

def parse_args():
    parser = argparse.ArgumentParser(description="Flux2Klein Differentiable Minecraft Skin Trainer")
    
    # Model and paths
    parser.add_argument("--model_path", type=str, required=True, help="Base Flux model path on Hugging Face or local path.")
    parser.add_argument("--model_type", type=str, default="standard", choices=["standard", "flux2klein"], help="Model structure type ('standard' for diffusers, 'flux2klein' for custom Qwen-based Flux2 model).")
    parser.add_argument("--text_encoder_type", type=str, default="t5_clip", choices=["t5_clip", "qwen"], help="Text encoder type ('t5_clip' for CLIP+T5, 'qwen' for Qwen3).")
    parser.add_argument("--text_encoder_path", type=str, default=None, help="Path to the Qwen text encoder (defaults to Qwen/Qwen3-4B if text_encoder_type is qwen).")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to skins folder containing target 64x64 skin PNGs.")
    parser.add_argument("--photos_dir", type=str, default=None, help="Path to conditioning photos folder.")
    parser.add_argument("--output_dir", type=str, default="output", help="Path to save checkpoints.")
    parser.add_argument("--mappings_dir", type=str, default=None, help="Path to differentiable renderer mappings folder.")
    
    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--save_every_epochs", type=int, default=5, help="Save checkpoints every N epochs.")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision mode.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm clipping.")
    
    # Differentiable Render Loss coefficients
    parser.add_argument("--lambda_latent", type=float, default=1.0, help="Flow matching latent loss coefficient.")
    parser.add_argument("--lambda_uv", type=float, default=1.0, help="Flat skin UV reconstruction loss coefficient.")
    parser.add_argument("--lambda_render", type=float, default=1.0, help="Rendered view reconstruction loss coefficient.")
    parser.add_argument("--lambda_lpips", type=float, default=0.1, help="Perceptual LPIPS render loss coefficient.")
    parser.add_argument("--use_lpips", action="store_true", help="Enable LPIPS perceptual loss on renders.")
    parser.add_argument("--foreground_weight", type=float, default=1.0, help="Foreground pixel weight multiplier.")
    parser.add_argument("--views", type=str, default="static_front,static_back", help="Comma-separated render views to use for training loss.")
    
    # LoRA fine-tuning parameters
    parser.add_argument("--use_lora", type=bool, default=True, help="Enable PEFT LoRA fine-tuning instead of full parameter training.")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank parameter.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha parameter.")
    parser.add_argument("--lora_target_modules", type=str, default=None, help="Comma-separated LoRA target module names (if None, targets suitable defaults based on model type).")
    
    # Validation / Sampling parameters
    parser.add_argument("--validation_photos_dir", type=str, default=None, help="Folder containing validation/test conditioning images.")
    parser.add_argument("--validation_steps", type=int, default=500, help="Run validation sampling once every N update steps.")
    
    return parser.parse_args()


def encode_prompt(tokenizer1, tokenizer2, text_encoder1, text_encoder2, prompt, device):
    """
    Standard Flux prompt encoding (CLIP + T5).
    """
    # 1. CLIP encoding
    inputs1 = tokenizer1(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
    prompt_embeds1 = text_encoder1(inputs1.input_ids)[0]
    
    # 2. T5 encoding
    inputs2 = tokenizer2(prompt, padding="max_length", max_length=256, truncation=True, return_tensors="pt").to(device)
    prompt_embeds2 = text_encoder2(inputs2.input_ids)[0]
    
    # Concatenate/pad embeddings to match Flux input expectations (CLIP + T5)
    # T5 projection is usually padded/concat.
    # For standard Flux, prompt_embeds is of shape (B, 512, 4096)
    # We return prompt_embeds and pooled_prompt_embeds (CLIP pooled output)
    pooled_prompt_embeds = text_encoder1(inputs1.input_ids, output_hidden_states=True).pooler_output
    
    # Shape of CLIP output is (B, 77, 768), T5 is (B, 256, 4096)
    # Pad CLIP embedding to match 4096 dimension or concat:
    # Flux expects pooled_embeds: (B, 768) and text_embeds: (B, 512, 4096)
    # Let's pad clip embeds along the channel dimension and concatenate
    B = prompt_embeds1.shape[0]
    clip_padded = F.pad(prompt_embeds1, (0, 4096 - 768)) # Pad 768 to 4096
    
    # Combine CLIP (77 tokens) and T5 (256 tokens) padded/aligned to 512 sequence length
    combined_embeds = torch.cat([clip_padded, prompt_embeds2], dim=1) # (B, 333, 4096)
    # Pad sequence length from 333 to 512
    combined_embeds = F.pad(combined_embeds, (0, 0, 0, 512 - combined_embeds.shape[1])) # (B, 512, 4096)
    
    return combined_embeds, pooled_prompt_embeds

def encode_prompt_qwen(tokenizer, text_encoder, prompt, device, max_sequence_length=512):
    """
    Qwen-based prompt encoding for Flux2Klein (Mistral/Qwen structure).
    """
    from einops import rearrange
    all_input_ids = []
    all_attention_masks = []
    
    for p in prompt:
        messages = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )
        all_input_ids.append(model_inputs["input_ids"])
        all_attention_masks.append(model_inputs["attention_mask"])
        
    input_ids = torch.cat(all_input_ids, dim=0).to(device)
    attention_mask = torch.cat(all_attention_masks, dim=0).to(device)
    
    output = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    
    # Extract Qwen layers [9, 18, 27] and stack
    OUTPUT_LAYERS_QWEN3 = [9, 18, 27]
    out = torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS_QWEN3], dim=1)
    prompt_embeds = rearrange(out, "b c l d -> b l (c d)")
    
    return prompt_embeds, None

def run_validation(args, transformer, vae, tokenizer1, tokenizer2, text_encoder1, text_encoder2, device, weight_dtype, global_step, accelerator):
    if not args.validation_photos_dir or not os.path.exists(args.validation_photos_dir):
        return
        
    print(f"\n[*] Running validation sampling at step {global_step}...")
    
    # 1. Scan validation directory for images
    # Supports separate front/back subfolders or single images
    front_dir = os.path.join(args.validation_photos_dir, "front")
    back_dir = os.path.join(args.validation_photos_dir, "back")
    
    val_pairs = []
    image_extensions = (".png", ".jpg", ".jpeg", ".webp")
    
    if os.path.exists(front_dir) and os.path.exists(back_dir):
        # Scan front directory
        for f in sorted(os.listdir(front_dir)):
            if f.lower().endswith(image_extensions):
                stem, ext = os.path.splitext(f)
                # Find matching back view image
                for b_ext in image_extensions:
                    b_path = os.path.join(back_dir, stem + b_ext)
                    if os.path.exists(b_path):
                        val_pairs.append({
                            "type": "split",
                            "stem": stem,
                            "front": os.path.join(front_dir, f),
                            "back": b_path
                        })
                        break
    else:
        # Scan root folder for single images
        for f in sorted(os.listdir(args.validation_photos_dir)):
            if os.path.isfile(os.path.join(args.validation_photos_dir, f)) and f.lower().endswith(image_extensions):
                stem, _ = os.path.splitext(f)
                val_pairs.append({
                    "type": "single",
                    "stem": stem,
                    "path": os.path.join(args.validation_photos_dir, f)
                })
                
    if not val_pairs:
        print("[!] No validation images found in validation_photos_dir.")
        return
        
    # Unstage/unwrap PEFT model for inference mode
    unwrapped_transformer = accelerator.unwrap_model(transformer)
    unwrapped_transformer.eval()
    
    # Define validation output directory
    val_output_dir = os.path.join(args.output_dir, "validation_samples")
    os.makedirs(val_output_dir, exist_ok=True)
    
    bg_color = (128, 128, 128) # solid gray
    
    # 2. Loop over validation pairs
    for item in val_pairs:
        stem = item["stem"]
        try:
            # Load images
            if item["type"] == "split":
                front_img = Image.open(item["front"]).convert("RGB")
                back_img = Image.open(item["back"]).convert("RGB")
                
                # Resize both to 256x512
                front_img = front_img.resize((256, 512), resample=Image.Resampling.LANCZOS)
                back_img = back_img.resize((256, 512), resample=Image.Resampling.LANCZOS)
            else:
                combined_img = Image.open(item["path"]).convert("RGB")
                w, h = combined_img.size
                front_img = combined_img.crop((0, 0, w // 2, h)).resize((256, 512), resample=Image.Resampling.LANCZOS)
                back_img = combined_img.crop((w // 2, 0, w, h)).resize((256, 512), resample=Image.Resampling.LANCZOS)
                
            # Load prompt txt if it exists
            caption_path = os.path.join(args.validation_photos_dir, stem + ".txt")
            if os.path.exists(caption_path):
                with open(caption_path, "r", encoding="utf-8") as f:
                    prompt = f.read().strip()
            else:
                prompt = "minecraft skin character"
                
            print(f"  - Sampling: {stem} | Prompt: '{prompt}'")
            
            # Prepare conditioning tensors (2, 3, 512, 256), scaled to [-1, 1] range for VAE
            from torchvision.transforms.functional import to_tensor
            front_tensor = (to_tensor(front_img) * 2.0 - 1.0).to(device, dtype=weight_dtype)
            back_tensor = (to_tensor(back_img) * 2.0 - 1.0).to(device, dtype=weight_dtype)
            controls_item = [front_tensor, back_tensor]
            
            # 3. Encode prompt
            with torch.no_grad():
                if args.text_encoder_type == "qwen":
                    prompt_embeds, pooled_prompt_embeds = encode_prompt_qwen(
                        tokenizer1, text_encoder1, [prompt], device
                    )
                else:
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        tokenizer1, tokenizer2, text_encoder1, text_encoder2, [prompt], device
                    )
                prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
                if pooled_prompt_embeds is not None:
                    pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=weight_dtype)
                    
            # 4. Generate random initial noise latents (1, 16, 64, 32)
            # Since VAE target canvas is 256x512, downsampling by 8 gives H_latent=64, W_latent=32.
            # Channels is 16 for Flux.
            # We must run generation in a torch.no_grad() block
            with torch.no_grad():
                # Encode conditioning images to sequence tokens for custom Flux2 model
                if args.model_type == "flux2klein":
                    img_cond_seq, img_cond_seq_ids = encode_image_refs(vae, controls_item)
                    img_cond_seq = img_cond_seq.to(device, dtype=weight_dtype)
                    img_cond_seq_ids = img_cond_seq_ids.to(device)
                    
                latents = torch.randn((1, 16, 64, 32), device=device, dtype=weight_dtype)
                
                # Euler ODE integration steps
                num_inference_steps = 28
                dt = 1.0 / num_inference_steps
                t = 1.0
                
                for step_idx in range(num_inference_steps):
                    t_tensor = torch.full((1,), t, device=device, dtype=weight_dtype)
                    
                    if args.model_type == "flux2klein":
                        packed_latents, img_ids = batched_prc_img(latents)
                        packed_txt, txt_ids = batched_prc_txt(prompt_embeds)
                        guidance_vec = torch.full((1,), 4.0, device=device, dtype=weight_dtype)
                        
                        img_input = torch.cat((packed_latents, img_cond_seq), dim=1)
                        img_input_ids = torch.cat((img_ids, img_cond_seq_ids), dim=1)
                        
                        model_pred_packed = unwrapped_transformer(
                            x=img_input,
                            x_ids=img_input_ids,
                            timesteps=t_tensor,
                            ctx=packed_txt,
                            ctx_ids=txt_ids.to(device),
                            guidance=guidance_vec
                        )
                        
                        # Slice output back to original sequence length (excluding cond tokens)
                        model_pred_packed = model_pred_packed[:, :packed_latents.shape[1]]
                        
                        unpacked_list = scatter_ids(model_pred_packed, img_ids)
                        model_pred = torch.cat(unpacked_list, dim=0).squeeze(2)
                    else:
                        txt_ids = torch.zeros(1, 512, 3, device=device, dtype=weight_dtype)
                        img_ids = torch.zeros(1, 64 * 32, 3, device=device, dtype=weight_dtype)
                        
                        model_pred = unwrapped_transformer(
                            hidden_states=latents,
                            timestep=t_tensor * 1000.0,
                            encoder_hidden_states=prompt_embeds,
                            pooled_projections=pooled_prompt_embeds,
                            txt_ids=txt_ids,
                            img_ids=img_ids,
                            return_dict=False
                        )[0]
                        
                    # Euler integration step
                    latents = latents - dt * model_pred
                    t -= dt
                    
                # Decode predicted latent x_0 using VAE
                if args.model_type == "flux2klein":
                    pred_decoded = vae.decode(latents.to(dtype=vae.dtype))
                else:
                    latents_scaled = latents / vae.config.scaling_factor
                    pred_decoded = vae.decode(latents_scaled).sample
                    
                pred_decoded = (pred_decoded + 1.0) / 2.0
                pred_decoded = pred_decoded.clamp(0.0, 1.0)
                
                # Extract 64x64 RGBA Skin
                pred_rgb = pred_decoded[:, :, :256, :]
                pred_alpha = pred_decoded[:, :, 256:, :].mean(dim=1, keepdim=True)
                
                pred_rgb_64 = F.interpolate(pred_rgb, size=(64, 64), mode='bilinear', align_corners=True)
                pred_alpha_64 = F.interpolate(pred_alpha, size=(64, 64), mode='bilinear', align_corners=True)
                pred_skin = torch.cat([pred_rgb_64, pred_alpha_64], dim=1) # (1, 4, 64, 64)
                
                # Save generated images
                # 64x64 RGBA skin
                skin_tensor = pred_skin[0].cpu().float()
                skin_np = (skin_tensor.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                skin_img = Image.fromarray(skin_np, "RGBA")
                skin_img.save(os.path.join(val_output_dir, f"step_{global_step}_{stem}_skin.png"))
                
                # 256x512 VAE canvas
                decoded_tensor = pred_decoded[0].cpu().float()
                decoded_np = (decoded_tensor.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                decoded_img = Image.fromarray(decoded_np, "RGB")
                decoded_img.save(os.path.join(val_output_dir, f"step_{global_step}_{stem}_canvas.png"))
                
        except Exception as e:
            print(f"[!] Error sampling validation image '{stem}': {e}")
            
    # Set model back to train mode
    transformer.train()
    print("[*] Validation sampling complete. Resuming training.")

def main():
    args = parse_args()
    
    # 1. Setup Accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    device = accelerator.device
    
    # Create checkpoints output folder
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define precision types
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    # 2. Load Models based on model_type & text_encoder_type
    print(f"[*] Model type: {args.model_type} | Text Encoder type: {args.text_encoder_type}")
    
    # Tokenizers and Text Encoders Loading
    if args.text_encoder_type == "qwen":
        from transformers import Qwen2Tokenizer, AutoModelForCausalLM
        te_path = args.text_encoder_path or "Qwen/Qwen3-4B"
        print(f"[*] Loading Qwen text encoder from: {te_path}")
        tokenizer = Qwen2Tokenizer.from_pretrained(te_path)
        text_encoder = AutoModelForCausalLM.from_pretrained(te_path, torch_dtype=weight_dtype)
        text_encoder.requires_grad_(False)
        text_encoder.eval()
        
        tokenizer1, tokenizer2 = tokenizer, None
        text_encoder1, text_encoder2 = text_encoder, None
    else:
        # Standard CLIP + T5 Text Encoders
        print(f"[*] Loading standard CLIP + T5 text encoders from: {args.model_path}")
        tokenizer1 = CLIPTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")
        tokenizer2 = T5TokenizerFast.from_pretrained(args.model_path, subfolder="tokenizer_2")
        text_encoder1 = CLIPTextModel.from_pretrained(args.model_path, subfolder="text_encoder")
        text_encoder2 = T5EncoderModel.from_pretrained(args.model_path, subfolder="text_encoder_2")
        text_encoder1.requires_grad_(False)
        text_encoder2.requires_grad_(False)
        text_encoder1.eval()
        text_encoder2.eval()
        
    # VAE and Denoising Transformer Loading
    if args.model_type == "flux2klein":
        from safetensors.torch import load_file
        
        # Load VAE from custom safetensors file
        vae_path = os.path.join(args.model_path, "ae.safetensors")
        if not os.path.exists(vae_path):
            vae_path = os.path.join(args.model_path, "vae", "ae.safetensors")
        if not os.path.exists(vae_path):
            vae_path = os.path.join(args.model_path, "vae", "diffusion_pytorch_model.safetensors")
            
        print(f"[*] Loading custom VAE from: {vae_path}")
        vae_state_dict = load_file(vae_path, device="cpu")
        vae_state_dict = convert_diffusers_vae_to_custom(vae_state_dict)
        autoencoder_params = AutoEncoderParams()
        # Check if small VAE decoder layout is used (e.g. channels count)
        if vae_state_dict.get('decoder.up.0.block.0.conv1.bias', None) is not None:
            if vae_state_dict['decoder.up.0.block.0.conv1.bias'].shape[0] == 96:
                autoencoder_params = AutoEncoderSmallDecoderParams()
                
        # Populate BN default statistics if missing (e.g. when loading standard diffusers VAE)
        if "bn.running_mean" not in vae_state_dict:
            vae_state_dict["bn.running_mean"] = torch.zeros(128)
        if "bn.running_var" not in vae_state_dict:
            vae_state_dict["bn.running_var"] = torch.ones(128)
        if "bn.num_batches_tracked" not in vae_state_dict:
            vae_state_dict["bn.num_batches_tracked"] = torch.tensor(0, dtype=torch.long)

        vae = AutoEncoder(autoencoder_params)
        for k in vae_state_dict:
            vae_state_dict[k] = vae_state_dict[k].to(dtype=weight_dtype)
        vae.load_state_dict(vae_state_dict, strict=False, assign=True)
        vae.requires_grad_(False)
        vae.eval()
        
        # Load Scheduler
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model_path, subfolder="scheduler")
        
        # Setup Transformer parameters
        if "4b" in args.model_path.lower():
            params = Klein4BParams()
        else:
            params = Klein9BParams()
        transformer = Flux2(params)
        
        # Locate safetensors weight file
        sf_file = None
        if os.path.isdir(args.model_path):
            for f in os.listdir(args.model_path):
                if f.endswith(".safetensors") and f != "ae.safetensors":
                    sf_file = os.path.join(args.model_path, f)
                    break
            if sf_file is None:
                raise FileNotFoundError(f"No safetensors model weights found in: {args.model_path}")
        else:
            sf_file = args.model_path
            
        print(f"[*] Loading custom transformer weights from: {sf_file}")
        transformer_state_dict = load_file(sf_file, device="cpu")
        for k in transformer_state_dict:
            transformer_state_dict[k] = transformer_state_dict[k].to(dtype=weight_dtype)
        transformer.load_state_dict(transformer_state_dict, assign=True)
        
        # Define LoRA targets for custom Flux2 model
        default_lora_targets = ["qkv", "linear1", "linear2", "proj"]
    else:
        # Standard diffusers model structures
        print(f"[*] Loading standard VAE and Transformer from: {args.model_path}")
        vae = AutoencoderKL.from_pretrained(args.model_path, subfolder="vae")
        vae.requires_grad_(False)
        vae.eval()
        
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model_path, subfolder="scheduler")
        transformer = FluxTransformer2DModel.from_pretrained(args.model_path, subfolder="transformer")
        
        # Standard attention targets
        default_lora_targets = ["to_q", "to_k", "to_v", "to_out.0"]
        
    # 3. Setup LoRA if requested
    if args.use_lora:
        target_modules = args.lora_target_modules.split(",") if args.lora_target_modules else default_lora_targets
        print(f"[*] Wrapping Transformer with LoRA (Rank={args.lora_rank}, Alpha={args.lora_alpha}, Targets={target_modules})")
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none"
        )
        transformer = get_peft_model(transformer, lora_config)
        transformer.print_trainable_parameters()
    else:
        print("[*] Training FULL model (All transformer parameters will be updated)")
        transformer.requires_grad_(True)
        
    # 4. Setup Dataset & Dataloader
    dataset = MinecraftSkinDataset(
        data_dir=args.data_dir,
        photos_dir=args.photos_dir,
        cond_size=512
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )
    
    # 5. Setup Optimizer
    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(transformer.parameters(), lr=args.lr)
    
    # 6. Setup Loss criterion (Differentiable Renderer + UV Loss + LPIPS)
    criterion = MinecraftLoss(
        mappings_dir=args.mappings_dir,
        lambda_uv=args.lambda_uv,
        lambda_render=args.lambda_render,
        lambda_lpips=args.lambda_lpips,
        use_lpips=args.use_lpips,
        views=args.views,
        foreground_weight=args.foreground_weight
    )
    
    # 7. Accelerate wrap
    transformer, optimizer, dataloader = accelerator.prepare(
        transformer, optimizer, dataloader
    )
    
    # Move VAE, encoders, and loss module to GPU device
    vae.to(device)
    if args.text_encoder_type == "qwen":
        text_encoder.to(device)
    else:
        text_encoder1.to(device)
        text_encoder2.to(device)
    criterion.to(device)
    
    # Main training loop
    print("[*] Starting training loop...")
    global_step = 0
    for epoch in range(args.epochs):
        transformer.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", disable=not accelerator.is_local_main_process)
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(transformer):
                # Retrieve batch data
                target_latent_image = batch["target_latent_image"].to(device, dtype=weight_dtype) # (B, 3, 512, 256)
                cond_image = batch["cond_image"].to(device, dtype=weight_dtype)                 # (B, 3, 512, 512)
                prompts = batch["prompt"]                                                       # List of strings
                gt_skin = batch["gt_skin"].to(device, dtype=weight_dtype)                       # (B, 4, 64, 64)
                
                B = target_latent_image.shape[0]
                
                # 8. Encode prompts on the fly
                with torch.no_grad():
                    if args.text_encoder_type == "qwen":
                        prompt_embeds, pooled_prompt_embeds = encode_prompt_qwen(
                            tokenizer, text_encoder, prompts, device
                        )
                    else:
                        prompt_embeds, pooled_prompt_embeds = encode_prompt(
                            tokenizer1, tokenizer2, text_encoder1, text_encoder2, prompts, device
                        )
                    prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
                    if pooled_prompt_embeds is not None:
                        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=weight_dtype)
                
                # 9. Encode Target Images into Latent space (x0)
                with torch.no_grad():
                    # VAE target_latent_image is normalized in [-1, 1]
                    if args.model_type == "flux2klein":
                        latents_gt = vae.encode(target_latent_image.to(dtype=vae.dtype))
                    else:
                        latents_gt = vae.encode(target_latent_image).latent_dist.sample()
                        latents_gt = latents_gt * vae.config.scaling_factor
                    latents_gt = latents_gt.to(dtype=weight_dtype)
                
                # 10. Sample timesteps and add noise (Flow Matching formulation)
                # Sample random t in a safe range [0.0, 0.90] to avoid division-by-zero instabilities
                t = torch.rand((B,), device=device, dtype=weight_dtype) * 0.90
                noise = torch.randn_like(latents_gt)
                
                # Compute noisy latent at t: x_t = (1 - t) * x_0 + t * noise
                t_expanded = t.view(-1, 1, 1, 1)
                x_t = (1.0 - t_expanded) * latents_gt + t_expanded * noise
                
                # 11. Run Transformer Forward pass
                if args.model_type == "flux2klein":
                    # Pack sequence coordinates for the custom local Flux2 model
                    packed_latents, img_ids = batched_prc_img(x_t)
                    packed_txt, txt_ids = batched_prc_txt(prompt_embeds)
                    
                    # Encode conditioning images (front & back separate views, both 256x512)
                    # cond_image has shape (B, 2, 3, 512, 256) in range [0, 1]. Scale to [-1, 1] for VAE.
                    cond_image_scaled = cond_image * 2.0 - 1.0
                    img_cond_seq_list = []
                    img_cond_seq_ids_list = []
                    for i in range(B):
                        controls_item = [cond_image_scaled[i, 0], cond_image_scaled[i, 1]]
                        seq_item, ids_item = encode_image_refs(vae, controls_item)
                        img_cond_seq_list.append(seq_item)
                        img_cond_seq_ids_list.append(ids_item)
                    img_cond_seq = torch.cat(img_cond_seq_list, dim=0).to(device, dtype=weight_dtype)
                    img_cond_seq_ids = torch.cat(img_cond_seq_ids_list, dim=0).to(device)
                    
                    # Concatenate reference/control tokens to image sequence
                    img_input = torch.cat((packed_latents, img_cond_seq), dim=1)
                    img_input_ids = torch.cat((img_ids, img_cond_seq_ids), dim=1)
                    
                    # Prepare guidance vec
                    guidance_vec = torch.full((B,), 4.0, device=device, dtype=weight_dtype) # default guidance 4.0
                    
                    packed_noise_pred = transformer(
                        x=img_input,
                        x_ids=img_input_ids,
                        timesteps=t, # Already normalized to [0, 1]
                        ctx=packed_txt,
                        ctx_ids=txt_ids.to(device),
                        guidance=guidance_vec
                    )
                    
                    # Slice prediction output back to match original latents sequence length (excluding cond tokens)
                    packed_noise_pred = packed_noise_pred[:, :packed_latents.shape[1]]
                    
                    # Scatter/unpack tokens back to spatial coordinates
                    unpacked_list = scatter_ids(packed_noise_pred, img_ids)
                    model_pred = torch.cat(unpacked_list, dim=0).squeeze(2) # Shape: (B, 16, H_latent, W_latent)
                else:
                    # Standard Flux coordinates grids
                    H_latent, W_latent = latents_gt.shape[2], latents_gt.shape[3]
                    txt_ids = torch.zeros(B, 512, 3, device=device, dtype=weight_dtype)
                    img_ids = torch.zeros(B, H_latent * W_latent, 3, device=device, dtype=weight_dtype)
                    
                    model_pred = transformer(
                        hidden_states=x_t,
                        timestep=t * 1000.0, # Range [0, 1000]
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        txt_ids=txt_ids,
                        img_ids=img_ids,
                        return_dict=False
                    )[0]
                
                # Flow matching target velocity: (noise - latents_gt)
                target_velocity = noise - latents_gt
                
                # Flow Matching Latent Loss (standard MSE)
                loss_latent = F.mse_loss(model_pred, target_velocity)
                
                # 12. Differentiable VAE Decode to reconstruct Predicted Skin UV
                # Reconstruct clean latent estimation from the predicted velocity:
                pred_x0 = x_t - t_expanded * model_pred
                
                # Decode predicted latent x_0 to 256x512 RGB space using VAE
                if args.model_type == "flux2klein":
                    pred_decoded = vae.decode(pred_x0.to(dtype=vae.dtype))
                else:
                    pred_x0_scaled = pred_x0 / vae.config.scaling_factor
                    pred_decoded = vae.decode(pred_x0_scaled).sample
                
                # Normalize decoded outputs from [-1, 1] range to [0, 1] range
                pred_decoded = (pred_decoded + 1.0) / 2.0
                pred_decoded = pred_decoded.clamp(0.0, 1.0)
                
                # 13. Extract the 64x64 RGBA Skin UV from the top-to-bottom composite
                # Top half is RGB: [0:256, :]
                pred_rgb = pred_decoded[:, :, :256, :]
                # Bottom half is Alpha (grayscale representation): [256:512, :]
                pred_alpha = pred_decoded[:, :, 256:, :].mean(dim=1, keepdim=True)
                
                # Resize from 256x256 to 64x64 using bilinear interpolation
                pred_rgb_64 = F.interpolate(pred_rgb, size=(64, 64), mode='bilinear', align_corners=True)
                pred_alpha_64 = F.interpolate(pred_alpha, size=(64, 64), mode='bilinear', align_corners=True)
                
                # Combine to form the predicted 64x64 RGBA skin texture
                pred_skin = torch.cat([pred_rgb_64, pred_alpha_64], dim=1) # (B, 4, 64, 64)
                
                # 14. Compute Combined Loss
                loss_criterion_dict = criterion(pred_skin, gt_skin)
                loss_render_total = loss_criterion_dict["loss_total"]
                
                loss = args.lambda_latent * loss_latent + loss_render_total
                
                # 15. Backpropagate Loss
                accelerator.backward(loss)
                
                # Clip gradients
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                    
                optimizer.step()
                optimizer.zero_grad()
                
                if accelerator.sync_gradients:
                    global_step += 1
                    if args.validation_photos_dir and global_step % args.validation_steps == 0:
                        if accelerator.is_main_process:
                            run_validation(
                                args,
                                transformer,
                                vae,
                                tokenizer1,
                                tokenizer2,
                                text_encoder1,
                                text_encoder2,
                                device,
                                weight_dtype,
                                global_step,
                                accelerator
                            )
                
            # Log metrics
            epoch_loss += loss.item()
            progress_bar.set_postfix({
                "Total Loss": f"{loss.item():.4f}",
                "Latent MSE": f"{loss_latent.item():.4f}",
                "UV MSE": f"{loss_criterion_dict['loss_uv'].item():.4f}",
                "Render MSE": f"{loss_criterion_dict['loss_render_mse'].item():.4f}"
            })
            
        epoch_loss = epoch_loss / len(dataloader)
        print(f"[*] Epoch {epoch} complete | Average Loss: {epoch_loss:.5f}")
        
        # Save checkpoints periodically
        if (epoch + 1) % args.save_every_epochs == 0 or epoch == args.epochs - 1:
            if accelerator.is_main_process:
                checkpoint_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
                print(f"[*] Saving checkpoint to: {checkpoint_path}")
                unwrapped_model = accelerator.unwrap_model(transformer)
                unwrapped_model.save_pretrained(checkpoint_path)

if __name__ == "__main__":
    main()

