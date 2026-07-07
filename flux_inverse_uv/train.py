import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm
from PIL import Image
from pathlib import Path

# Inject workspace root, toolkit root, and img2skin path into sys.path
FLUX_INVERSE_UV_DIR = Path(__file__).resolve().parent
TOOLKIT_ROOT = FLUX_INVERSE_UV_DIR.parent
WORKSPACE_ROOT = TOOLKIT_ROOT.parent
IMG2SKIN_DIR = TOOLKIT_ROOT / "img2skin"

for p in [str(WORKSPACE_ROOT), str(TOOLKIT_ROOT), str(IMG2SKIN_DIR)]:
    if p not in sys.path:
        sys.path.append(p)
if str(FLUX_INVERSE_UV_DIR) not in sys.path:
    sys.path.insert(0, str(FLUX_INVERSE_UV_DIR))

# Local imports
from dataset import FluxInverseUVDataset
from SkingToolkit.img2skin.flux2_src.model import Flux2, Klein4BParams, Klein9BParams
from SkingToolkit.img2skin.flux2_src.autoencoder import AutoEncoder, AutoEncoderParams, AutoEncoderSmallDecoderParams
from SkingToolkit.img2skin.flux2_src.sampling import batched_prc_img, batched_prc_txt, scatter_ids, encode_image_refs

# Diffusers / Transformers / PEFT imports
from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import Qwen2Tokenizer, AutoModelForCausalLM
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model

def convert_diffusers_vae_to_custom(state_dict):
    if "encoder.quant_conv.weight" in state_dict:
        return state_dict

    is_diffusers = any(k.startswith("encoder.down_blocks") for k in state_dict.keys())
    if not is_diffusers:
        return state_dict

    custom_state_dict = {}

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

    for i in range(4):
        old_ds_w = f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"
        old_ds_b = f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"
        if old_ds_w in state_dict:
            custom_state_dict[f"encoder.down.{i}.downsample.conv.weight"] = state_dict[old_ds_w]
        if old_ds_b in state_dict:
            custom_state_dict[f"encoder.down.{i}.downsample.conv.bias"] = state_dict[old_ds_b]

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

    for j in range(2):
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

    for i in range(4):
        custom_i = 3 - i
        old_us_w = f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"
        old_us_b = f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"
        if old_us_w in state_dict:
            custom_state_dict[f"decoder.up.{custom_i}.upsample.conv.weight"] = state_dict[old_us_w]
        if old_us_b in state_dict:
            custom_state_dict[f"decoder.up.{custom_i}.upsample.conv.bias"] = state_dict[old_us_b]

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

    for j in range(2):
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
    parser = argparse.ArgumentParser(description="Flux Inverse UV Fine-Tuning Trainer")
    
    # Model and paths
    parser.add_argument("--model_path", type=str, required=True, help="Base Flux model path on Hugging Face or local path.")
    parser.add_argument("--text_encoder_path", type=str, default=None, help="Path to Qwen text encoder model.")
    parser.add_argument("--control_imgs_dir", type=str, default=None, help="Path to conditioning control_imgs folder.")
    parser.add_argument("--photos_dir", type=str, default=None, help="Alias for --control_imgs_dir.")
    parser.add_argument("--target_imgs_dir", type=str, default=None, help="Path to pre-built target_imgs folder.")
    parser.add_argument("--data_dir", type=str, default=None, help="Optional path to skins folder containing target 64x64 skin PNGs.")
    parser.add_argument("--output_dir", type=str, default="output/flux_inverse_uv_lora", help="Path to save checkpoints.")
    
    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs.")
    parser.add_argument("--save_every_epochs", type=int, default=10, help="Save checkpoints every N epochs.")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision mode.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm clipping.")
    parser.add_argument("--resolution", type=int, default=512, help="Target/control image resolution (e.g. 512x512).")
    
    # Loss coefficient
    parser.add_argument("--lambda_latent", type=float, default=1.0, help="Flow matching latent loss coefficient.")
    parser.add_argument("--lambda_pixel", type=float, default=0.0, help="Weight coefficient for pixel-level reconstruction loss (0.0 to disable).")
    parser.add_argument("--lambda_dot_weight", type=float, default=10.0, help="Weight multiplier for transparent white dot pixels.")
    parser.add_argument("--lambda_uniformity", type=float, default=1.0, help="Weight coefficient for opaque block uniformity loss.")
    
    # LoRA fine-tuning parameters
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank parameter for linear layers.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter for linear layers.")
    parser.add_argument("--lora_conv_rank", type=int, default=16, help="Compatibility knob for conv rank.")
    parser.add_argument("--lora_conv_alpha", type=int, default=16, help="Compatibility knob for conv alpha.")
    parser.add_argument("--lora_target_modules", type=str, default=None, help="Comma-separated LoRA target module names.")
    
    # Validation parameters
    parser.add_argument("--validation_dir", type=str, default=None, help="Folder containing validation/test conditioning images.")
    parser.add_argument("--validation_photos_dir", type=str, default=None, help="Alias for --validation_dir.")
    parser.add_argument("--validation_steps", type=int, default=100, help="Run validation sampling once every N update steps.")
    
    return parser.parse_args()

def encode_prompt_qwen(tokenizer, text_encoder, prompt, device, max_sequence_length=512):
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
    
    OUTPUT_LAYERS_QWEN3 = [9, 18, 27]
    out = torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS_QWEN3], dim=1)
    prompt_embeds = rearrange(out, "b c l d -> b l (c d)")
    
    return prompt_embeds, None

def run_validation(args, transformer, vae, prompt_cache, device, weight_dtype, global_step, accelerator):
    val_dir = args.validation_dir or args.validation_photos_dir
    if not val_dir or not os.path.exists(val_dir):
        return
        
    print(f"\n[*] Running validation sampling at step {global_step}...")
    target_height = args.resolution
    target_width = args.resolution
    
    val_pairs = []
    image_extensions = (".png", ".jpg", ".jpeg", ".webp")
    front_dir = os.path.join(val_dir, "front")
    back_dir = os.path.join(val_dir, "back")
    
    if os.path.exists(front_dir) and os.path.exists(back_dir):
        for f in sorted(os.listdir(front_dir)):
            if f.lower().endswith(image_extensions):
                stem, ext = os.path.splitext(f)
                for b_ext in image_extensions:
                    b_path = os.path.join(back_dir, stem + b_ext)
                    if os.path.exists(b_path):
                        val_pairs.append({"type": "split", "stem": stem, "front": os.path.join(front_dir, f), "back": b_path})
                        break
    else:
        for f in sorted(os.listdir(val_dir)):
            if os.path.isfile(os.path.join(val_dir, f)) and f.lower().endswith(image_extensions):
                stem, _ = os.path.splitext(f)
                val_pairs.append({"type": "single", "stem": stem, "path": os.path.join(val_dir, f)})
                
    if not val_pairs:
        print("[!] No validation images found in validation_dir.")
        return
        
    unwrapped_transformer = accelerator.unwrap_model(transformer)
    unwrapped_transformer.eval()
    
    val_output_dir = os.path.join(args.output_dir, "validation_samples")
    os.makedirs(val_output_dir, exist_ok=True)
    
    for item in val_pairs:
        stem = item["stem"]
        try:
            if item["type"] == "split":
                front_img = Image.open(item["front"]).convert("RGB").resize((target_width, target_height), resample=Image.Resampling.LANCZOS)
                back_img = Image.open(item["back"]).convert("RGB").resize((target_width, target_height), resample=Image.Resampling.LANCZOS)
            else:
                combined_img = Image.open(item["path"]).convert("RGB")
                w, h = combined_img.size
                front_img = combined_img.crop((0, 0, w // 2, h)).resize((target_width, target_height), resample=Image.Resampling.LANCZOS)
                back_img = combined_img.crop((w // 2, 0, w, h)).resize((target_width, target_height), resample=Image.Resampling.LANCZOS)
                
            prompt = ""
                
            from torchvision.transforms.functional import to_tensor
            front_tensor = (to_tensor(front_img) * 2.0 - 1.0).to(device, dtype=weight_dtype)
            back_tensor = (to_tensor(back_img) * 2.0 - 1.0).to(device, dtype=weight_dtype)
            controls_item = [front_tensor, back_tensor]
            
            pe, ppe = prompt_cache.get(prompt, (None, None))
            if pe is None:
                continue
                
            prompt_embeds = pe.to(device, dtype=weight_dtype)
            
            with torch.no_grad():
                img_cond_seq, img_cond_seq_ids = encode_image_refs(vae, controls_item)
                img_cond_seq = img_cond_seq.to(device, dtype=weight_dtype)
                img_cond_seq_ids = img_cond_seq_ids.to(device)
                
                latent_channels = vae.params.z_channels * int(np.prod(vae.ps))
                vae_scale_factor = (2 ** (len(vae.params.ch_mult) - 1)) * vae.ps[0]
                latent_h = target_height // vae_scale_factor
                latent_w = target_width // vae_scale_factor
                latents = torch.randn((1, latent_channels, latent_h, latent_w), device=device, dtype=weight_dtype)
                
                num_inference_steps = 28
                dt = 1.0 / num_inference_steps
                t = 1.0
                
                for step_idx in range(num_inference_steps):
                    t_tensor = torch.full((1,), t, device=device, dtype=weight_dtype)
                    
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
                    
                    model_pred_packed = model_pred_packed[:, :packed_latents.shape[1]]
                    unpacked_list = scatter_ids(model_pred_packed, img_ids)
                    model_pred = torch.cat(unpacked_list, dim=0).squeeze(2)
                    
                    latents = latents - dt * model_pred
                    t -= dt
                    
                pred_decoded = vae.decode(latents.to(dtype=vae.dtype))
                pred_decoded = ((pred_decoded + 1.0) / 2.0).clamp(0.0, 1.0)
                
                decoded_tensor = pred_decoded[0].cpu().float()
                decoded_np = (decoded_tensor.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                decoded_img = Image.fromarray(decoded_np, "RGB")
                decoded_img.save(os.path.join(val_output_dir, f"step_{global_step}_{stem}_target.png"))
                
        except Exception as e:
            print(f"[!] Error sampling validation image '{stem}': {e}")
            
    transformer.train()

def main():
    args = parse_args()
    
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    device = accelerator.device
    os.makedirs(args.output_dir, exist_ok=True)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    # Load Qwen text encoder
    te_path = args.text_encoder_path or "Qwen/Qwen3-4B"
    print(f"[*] Loading Qwen text encoder from: {te_path}")
    tokenizer = Qwen2Tokenizer.from_pretrained(te_path)
    text_encoder = AutoModelForCausalLM.from_pretrained(te_path, torch_dtype=weight_dtype)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    
    # Load VAE
    vae_path = os.path.join(args.model_path, "ae.safetensors")
    if not os.path.exists(vae_path):
        vae_path = os.path.join(args.model_path, "vae", "ae.safetensors")
    if not os.path.exists(vae_path):
        vae_path = os.path.join(args.model_path, "vae", "diffusion_pytorch_model.safetensors")
        
    print(f"[*] Loading custom VAE from: {vae_path}")
    vae_state_dict = load_file(vae_path, device="cpu")
    vae_state_dict = convert_diffusers_vae_to_custom(vae_state_dict)
    autoencoder_params = AutoEncoderParams()
    if vae_state_dict.get('decoder.up.0.block.0.conv1.bias', None) is not None:
        if vae_state_dict['decoder.up.0.block.0.conv1.bias'].shape[0] == 96:
            autoencoder_params = AutoEncoderSmallDecoderParams()
            
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
    
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model_path, subfolder="scheduler")
    
    params = Klein4BParams()
    transformer = Flux2(params)
    
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
    
    default_lora_targets = ["qkv", "linear1", "linear2", "proj"]
    target_modules = args.lora_target_modules.split(",") if args.lora_target_modules else default_lora_targets
    
    print(f"[*] Wrapping Transformer with LoRA (Rank={args.lora_rank}, Alpha={args.lora_alpha}, Targets={target_modules})")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none"
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()
    transformer.to(dtype=weight_dtype)
    
    # Load template mask for active area computation in pixel loss
    mask_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skin-mask.png")
    decor_mask_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skin-decor-mask.png")
    
    if os.path.exists(mask_path) and os.path.exists(decor_mask_path):
        mask_np = np.array(Image.open(mask_path))
        decor_mask_np = np.array(Image.open(decor_mask_path))
        active_mask_np = (mask_np[..., 3] > 0) | (decor_mask_np[..., 3] > 0)
    else:
        # Fallback if templates not in flux_inverse_uv/
        parent_mask = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Sking", "skin-mask.png")
        parent_decor = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Sking", "skin-decor-mask.png")
        if os.path.exists(parent_mask) and os.path.exists(parent_decor):
            mask_np = np.array(Image.open(parent_mask))
            decor_mask_np = np.array(Image.open(parent_decor))
            active_mask_np = (mask_np[..., 3] > 0) | (decor_mask_np[..., 3] > 0)
        else:
            # dummy active mask (all true)
            active_mask_np = np.ones((64, 64), dtype=bool)
            
    active_mask_64 = torch.from_numpy(active_mask_np).to(device)
    active_mask_512 = active_mask_64.repeat_interleave(8, dim=0).repeat_interleave(8, dim=1)
    active_mask_512 = active_mask_512.unsqueeze(0).unsqueeze(0).to(dtype=weight_dtype)

    # Dataset and DataLoader
    control_imgs_dir = args.control_imgs_dir or args.photos_dir
    dataset = FluxInverseUVDataset(
        control_imgs_dir=control_imgs_dir,
        target_imgs_dir=args.target_imgs_dir,
        data_dir=args.data_dir,
        cond_size=args.resolution,
        is_square=True,
        default_caption=""
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )
    
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=args.lr)
    
    transformer, optimizer, dataloader = accelerator.prepare(
        transformer, optimizer, dataloader
    )
    
    vae.to(device)
    text_encoder.to(device)
    
    # Pre-encode prompt cache
    print("[*] Pre-encoding training prompt...")
    prompt_cache = {}
    with torch.no_grad():
        pe, ppe = encode_prompt_qwen(tokenizer, text_encoder, [""], device)
        prompt_cache[""] = (pe.cpu(), ppe.cpu() if ppe is not None else None)
        
    print("[*] Unloading Text Encoder to free VRAM...")
    del text_encoder
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    global_step = 0
    print("[*] Starting training loop...")
    
    for epoch in range(args.epochs):
        transformer.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", disable=not accelerator.is_local_main_process)
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(transformer):
                target_latent_image = batch["target_latent_image"].to(device, dtype=weight_dtype)
                cond_image = batch["cond_image"].to(device, dtype=weight_dtype)
                prompts = batch["prompt"]
                
                B = target_latent_image.shape[0]
                
                prompt_embeds_list = []
                for p in prompts:
                    pe, ppe = prompt_cache[p]
                    prompt_embeds_list.append(pe.to(device, dtype=weight_dtype))
                prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
                
                with torch.no_grad():
                    latents_gt = vae.encode(target_latent_image.to(dtype=vae.dtype)).to(dtype=weight_dtype)
                    
                t = torch.rand((B,), device=device, dtype=weight_dtype)
                noise = torch.randn_like(latents_gt)
                
                with accelerator.autocast():
                    t_expanded = t.view(-1, 1, 1, 1)
                    x_t = (1.0 - t_expanded) * latents_gt + t_expanded * noise
                    
                    packed_latents, img_ids = batched_prc_img(x_t)
                    packed_txt, txt_ids = batched_prc_txt(prompt_embeds)
                    
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
                    
                    img_input = torch.cat((packed_latents, img_cond_seq), dim=1)
                    img_input_ids = torch.cat((img_ids, img_cond_seq_ids), dim=1)
                    guidance_vec = torch.full((B,), 4.0, device=device, dtype=weight_dtype)
                    
                    packed_noise_pred = transformer(
                        x=img_input,
                        x_ids=img_input_ids,
                        timesteps=t,
                        ctx=packed_txt,
                        ctx_ids=txt_ids.to(device),
                        guidance=guidance_vec
                    )
                    
                    packed_noise_pred = packed_noise_pred[:, :packed_latents.shape[1]]
                    unpacked_list = scatter_ids(packed_noise_pred, img_ids)
                    model_pred = torch.cat(unpacked_list, dim=0).squeeze(2)
                    
                    target_velocity = noise - latents_gt
                    loss_latent = F.mse_loss(model_pred, target_velocity)
                    loss = args.lambda_latent * loss_latent
                    
                    if args.lambda_pixel > 0:
                        # Decode predicted latent back to pixel space (using predicted x0)
                        # pred_x0 = x_t - t_expanded * model_pred
                        t_expanded = t.view(-1, 1, 1, 1)
                        pred_x0 = x_t - t_expanded * model_pred
                        
                        # Decode through VAE (decode method expects floats in [-1, 1] range)
                        pred_decoded = vae.decode(pred_x0.to(dtype=vae.dtype)).to(dtype=weight_dtype)
                        
                        # 1. Weighted Reconstruction Loss (inside active mask only)
                        # target_latent_image is normalized to [-1, 1]
                        loss_recon = F.l1_loss(pred_decoded, target_latent_image, reduction="none")
                        
                        # Find white dot pixels in target (where target is close to 1.0)
                        is_white_dot = (target_latent_image > 0.8) # (B, 3, 512, 512)
                        
                        weights = torch.ones_like(target_latent_image)
                        weights[is_white_dot] = args.lambda_dot_weight
                        
                        # Apply active mask to pixel loss
                        # active_mask_512 shape: (1, 1, 512, 512)
                        active_mask_512_expanded = active_mask_512.expand_as(loss_recon)
                        loss_pixel_recon = (loss_recon * weights * active_mask_512_expanded).sum() / (active_mask_512_expanded.sum() + 1e-8)
                        
                        # 2. Block Uniformity Loss for Opaque blocks (inside active mask only)
                        # Reshape pred_decoded to (B, C, 64, 8, 64, 8)
                        B, C, H, W = pred_decoded.shape
                        blocks = pred_decoded.view(B, C, 64, 8, 64, 8).permute(0, 2, 4, 1, 3, 5) # (B, 64, 64, C, 8, 8)
                        
                        # Reshape is_white_dot to (B, 64, 8, 64, 8)
                        dots_blocks = is_white_dot.view(B, C, 64, 8, 64, 8).permute(0, 2, 4, 1, 3, 5) # (B, 64, 64, C, 8, 8)
                        
                        # A block has a dot if any of its pixels has a dot (check across C, 8, 8)
                        block_has_dot = dots_blocks.any(dim=-3).any(dim=-2).any(dim=-1) # (B, 64, 64)
                        
                        # Block mean color
                        block_mean = blocks.mean(dim=(-2, -1), keepdim=True) # (B, 64, 64, C, 1, 1)
                        
                        # Uniformity loss: deviation from mean for blocks without dots
                        block_deviation = (blocks - block_mean) ** 2
                        
                        # Mask for blocks without dots AND inside active mask (B, 64, 64)
                        # active_mask_64 shape: (64, 64)
                        active_mask_blocks = active_mask_64.unsqueeze(0).expand(B, 64, 64)
                        opaque_mask_bool = (~block_has_dot) & active_mask_blocks # (B, 64, 64)
                        
                        opaque_mask = opaque_mask_bool.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(dtype=weight_dtype) # (B, 64, 64, 1, 1, 1)
                        opaque_mask_expanded = opaque_mask.expand_as(block_deviation)
                        
                        loss_uniformity = (block_deviation * opaque_mask_expanded).sum() / (opaque_mask_expanded.sum() + 1e-8)
                        
                        # Add to total loss
                        loss += args.lambda_pixel * (loss_pixel_recon + args.lambda_uniformity * loss_uniformity)
                        
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                
            global_step += 1
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            if global_step % args.validation_steps == 0:
                run_validation(args, transformer, vae, prompt_cache, device, weight_dtype, global_step, accelerator)
                
        # Save checkpoint per N epochs
        if (epoch + 1) % args.save_every_epochs == 0 or (epoch + 1) == args.epochs:
            if accelerator.is_main_process:
                ckpt_dir = os.path.join(args.output_dir, f"epoch_{epoch+1}")
                os.makedirs(ckpt_dir, exist_ok=True)
                unwrapped = accelerator.unwrap_model(transformer)
                unwrapped.save_pretrained(ckpt_dir)
                print(f"[*] Saved checkpoint to {ckpt_dir}")

    print("[*] Training completed!")

if __name__ == "__main__":
    main()
