#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import FlowMatchEulerDiscreteScheduler, Krea2Pipeline, Krea2Transformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from common import load_config, prompt_cache_key, read_jsonl, resolve_dtype, write_json


class LatentDataset(Dataset):
    def __init__(self, rows: list[dict], cache_dir: Path):
        self.items = []
        missing = []
        for row in rows:
            if row.get("split") != "train":
                continue
            path = cache_dir / f"{row['id']}.safetensors"
            if path.is_file():
                self.items.append((path, str(row["prompt_id"])))
            else:
                missing.append(path)
        if missing:
            print(f"warning: {len(missing)} training rows have no latent cache and will be skipped")
        if not self.items:
            raise RuntimeError("No cached training latents were found")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path, prompt_id = self.items[index]
        return load_file(path, device="cpu")["latents"], prompt_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Krea2 Transformer LoRA on cached MC preview latents.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume-lora", default=None)
    return parser.parse_args()


def save_lora(transformer: torch.nn.Module, accelerator: Accelerator, output_dir: Path, metadata: dict) -> None:
    if not accelerator.is_main_process:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(transformer)
    state_dict = get_peft_model_state_dict(unwrapped)
    Krea2Pipeline.save_lora_weights(
        output_dir,
        transformer_lora_layers=state_dict,
        safe_serialization=True,
    )
    write_json(output_dir / "adapter_metadata.json", metadata)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_config = config["model"]
    train_config = config["training"]
    data_config = config["data"]
    model_path = Path(model_config["path"]).expanduser().resolve()
    dataset_dir = Path(data_config["dataset_dir"]).expanduser().resolve()
    output_dir = Path(args.output_dir or train_config["output_dir"]).expanduser().resolve()
    max_train_steps = int(args.max_train_steps or train_config["max_train_steps"])
    gradient_accumulation_steps = int(train_config["gradient_accumulation_steps"])
    mixed_precision = str(train_config.get("mixed_precision", "bf16"))

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        project_config=ProjectConfiguration(project_dir=str(output_dir), logging_dir=str(output_dir / "logs")),
    )
    set_seed(int(train_config["seed"]), device_specific=True)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "resolved_config.json", config)

    rows = read_jsonl(dataset_dir / "metadata.jsonl")
    dataset = LatentDataset(rows, dataset_dir / "latents")
    loader = DataLoader(
        dataset,
        batch_size=int(train_config["batch_size"]),
        shuffle=True,
        num_workers=int(train_config.get("num_workers", 2)),
        pin_memory=True,
        drop_last=True,
    )
    prompt_cache = load_file(dataset_dir / "prompt_cache.safetensors", device="cpu")

    weight_dtype = resolve_dtype(model_config.get("dtype", "bf16"))
    transformer = Krea2Transformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
        local_files_only=True,
    )
    transformer.requires_grad_(False)
    if bool(train_config.get("gradient_checkpointing", True)):
        transformer.enable_gradient_checkpointing()
    target_modules = list(train_config["target_modules"])
    lora_rank = int(train_config["rank"])
    lora_alpha = int(train_config["lora_alpha"])
    resume_transformer_state = None
    if args.resume_lora:
        state = Krea2Pipeline.lora_state_dict(args.resume_lora)
        if isinstance(state, tuple):
            state = state[0]
        resume_transformer_state = {
            key.removeprefix("transformer."): value
            for key, value in state.items()
            if key.startswith("transformer.")
        }
        inferred_ranks = {
            int(value.shape[0])
            for key, value in resume_transformer_state.items()
            if key.endswith("lora_A.weight")
        }
        if len(inferred_ranks) != 1:
            raise ValueError(f"Could not infer one LoRA rank from {args.resume_lora}: {sorted(inferred_ranks)}")
        inferred_rank = inferred_ranks.pop()
        if inferred_rank != lora_rank:
            if lora_alpha == lora_rank:
                lora_alpha = inferred_rank
            print(f"overriding configured LoRA rank {lora_rank} with resumed rank {inferred_rank}")
            lora_rank = inferred_rank
    transformer.add_adapter(
        LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=float(train_config.get("lora_dropout", 0.0)),
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
    )
    if bool(train_config.get("layerwise_casting", False)):
        # Frozen base weights are stored as FP8 and upcast to BF16 per layer.
        # Diffusers' default skip list preserves precision-sensitive and PEFT layers.
        transformer.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn,
            compute_dtype=weight_dtype,
        )
    if resume_transformer_state is not None:
        incompatible = set_peft_model_state_dict(
            transformer,
            resume_transformer_state,
            adapter_name="default",
        )
        if incompatible.unexpected_keys:
            raise ValueError(f"Unexpected resumed LoRA keys: {incompatible.unexpected_keys[:8]}")
        print(f"resumed LoRA initialization: {args.resume_lora}")

    trainable_parameters = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    total_count = sum(parameter.numel() for parameter in transformer.parameters())
    if accelerator.is_main_process:
        print(f"trainable parameters: {trainable_count:,} / {total_count:,} ({100 * trainable_count / total_count:.4f}%)")

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(train_config["learning_rate"]),
        betas=(float(train_config.get("adam_beta1", 0.9)), float(train_config.get("adam_beta2", 0.999))),
        weight_decay=float(train_config.get("weight_decay", 0.01)),
        eps=float(train_config.get("adam_epsilon", 1e-8)),
    )
    lr_scheduler = get_scheduler(
        str(train_config.get("lr_scheduler", "cosine")),
        optimizer=optimizer,
        num_warmup_steps=int(train_config.get("lr_warmup_steps", 100)),
        num_training_steps=max_train_steps,
    )
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_path,
        subfolder="scheduler",
        local_files_only=True,
    )
    transformer, optimizer, loader, lr_scheduler = accelerator.prepare(transformer, optimizer, loader, lr_scheduler)

    first_latent, first_prompt_id = dataset[0]
    sequence_length = first_latent.shape[0]
    expected_channels = int(transformer.config.in_channels)
    if first_latent.shape[1] != expected_channels:
        raise ValueError(f"Cached latent channels {first_latent.shape[1]} != transformer in_channels {expected_channels}")
    height = int(rows[0]["height"])
    width = int(rows[0]["width"])
    vae_scale_factor = int(model_config.get("vae_scale_factor", 8))
    patch_size = int(model_config.get("patch_size", 2))
    grid_height = height // (vae_scale_factor * patch_size)
    grid_width = width // (vae_scale_factor * patch_size)
    if grid_height * grid_width != sequence_length:
        raise ValueError(f"Resolution implies {grid_height * grid_width} tokens but cache contains {sequence_length}")
    prompt_length = prompt_cache[prompt_cache_key("embeds", first_prompt_id)].shape[1]
    position_ids = Krea2Pipeline.prepare_position_ids(prompt_length, grid_height, grid_width, accelerator.device)
    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
    schedule_sigmas = noise_scheduler.sigmas.to(accelerator.device)
    weighting_scheme = str(train_config.get("weighting_scheme", "logit_normal"))
    global_step = 0
    progress = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process, desc="Krea2 LoRA")
    metadata = {
        "base_model": str(model_path),
        "task": "fixed Minecraft front/back orthographic preview",
        "rank": str(lora_rank),
        "lora_alpha": str(lora_alpha),
        "resolution": f"{width}x{height}",
        "layerwise_casting": str(bool(train_config.get("layerwise_casting", False))),
    }

    transformer.train()
    while global_step < max_train_steps:
        for clean_latents, prompt_ids in loader:
            with accelerator.accumulate(transformer):
                clean_latents = clean_latents.to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                noise = torch.randn_like(clean_latents)
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=weighting_scheme,
                    batch_size=clean_latents.shape[0],
                    logit_mean=float(train_config.get("logit_mean", 0.0)),
                    logit_std=float(train_config.get("logit_std", 1.0)),
                    mode_scale=float(train_config.get("mode_scale", 1.29)),
                    device=accelerator.device,
                )
                indices = (u * noise_scheduler.config.num_train_timesteps).long().clamp_(
                    0, noise_scheduler.config.num_train_timesteps - 1
                )
                timesteps = schedule_timesteps[indices]
                sigmas = schedule_sigmas[indices].to(dtype=clean_latents.dtype).view(-1, 1, 1)
                noisy_latents = (1.0 - sigmas) * clean_latents + sigmas * noise
                embeds = torch.cat(
                    [prompt_cache[prompt_cache_key("embeds", prompt_id)] for prompt_id in prompt_ids], dim=0
                ).to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                masks = torch.cat(
                    [prompt_cache[prompt_cache_key("mask", prompt_id)] for prompt_id in prompt_ids], dim=0
                ).to(accelerator.device, non_blocking=True)
                model_prediction = transformer(
                    hidden_states=noisy_latents,
                    encoder_hidden_states=embeds,
                    timestep=(timesteps / noise_scheduler.config.num_train_timesteps).to(weight_dtype),
                    position_ids=position_ids,
                    encoder_attention_mask=masks,
                    return_dict=False,
                )[0]
                target = noise - clean_latents
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=weighting_scheme, sigmas=sigmas)
                loss = (weighting.float() * F.mse_loss(model_prediction.float(), target.float(), reduction="none")).reshape(
                    clean_latents.shape[0], -1
                ).mean(dim=1).mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_parameters, float(train_config.get("max_grad_norm", 1.0)))
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=f"{loss.detach().item():.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")
                save_every = int(train_config.get("save_every", 250))
                if save_every > 0 and global_step % save_every == 0:
                    accelerator.wait_for_everyone()
                    save_lora(transformer, accelerator, output_dir / f"checkpoint-{global_step}", metadata)
            if global_step >= max_train_steps:
                break

    accelerator.wait_for_everyone()
    save_lora(transformer, accelerator, output_dir / "final", metadata)
    if accelerator.is_main_process:
        write_json(
            output_dir / "training_state.json",
            {
                "global_step": global_step,
                "trainable_parameters": trainable_count,
                "dataset_items": len(dataset),
                "final_lora": str(output_dir / "final"),
            },
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
