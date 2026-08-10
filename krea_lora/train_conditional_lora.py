#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import FlowMatchEulerDiscreteScheduler, Krea2Pipeline, Krea2Transformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from common import load_config, prompt_cache_key, read_jsonl, resolve_dtype, write_json
from reference_conditioning import prepare_paired_position_ids


class ConditionalLatentDataset(Dataset):
    def __init__(self, rows: list[dict], cache_dir: Path):
        self.items: list[tuple[Path, str]] = []
        missing = 0
        for row in rows:
            if row.get("split") != "train":
                continue
            path = cache_dir / f"{row['id']}.safetensors"
            if path.is_file():
                self.items.append((path, str(row["prompt_id"])))
            else:
                missing += 1
        if missing:
            print(f"warning: {missing} training pairs have no latent cache and will be skipped")
        if not self.items:
            raise RuntimeError("No cached paired training latents were found")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        path, prompt_id = self.items[index]
        tensors = load_file(path, device="cpu")
        return tensors["target_latents"], tensors["source_latents"], prompt_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Krea2 LoRA with [noisy target | clean reference] conditional latent tokens."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume-lora", default=None)
    return parser.parse_args()


def save_lora(transformer: torch.nn.Module, accelerator: Accelerator, output_dir: Path, metadata: dict) -> None:
    if not accelerator.is_main_process:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict = get_peft_model_state_dict(accelerator.unwrap_model(transformer))
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
    data_config = config["data"]
    train_config = config["training"]
    model_path = Path(model_config["path"]).expanduser().resolve()
    dataset_dir = Path(data_config["dataset_dir"]).expanduser().resolve()
    output_dir = Path(args.output_dir or train_config["output_dir"]).expanduser().resolve()
    max_train_steps = int(args.max_train_steps or train_config["max_train_steps"])
    accelerator = Accelerator(
        gradient_accumulation_steps=int(train_config["gradient_accumulation_steps"]),
        mixed_precision=str(train_config.get("mixed_precision", "bf16")),
        project_config=ProjectConfiguration(project_dir=str(output_dir), logging_dir=str(output_dir / "logs")),
    )
    set_seed(int(train_config["seed"]), device_specific=True)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "resolved_config.json", config)

    rows = read_jsonl(dataset_dir / "metadata.jsonl")
    dataset = ConditionalLatentDataset(rows, dataset_dir / "latents")
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
    transformer.add_adapter(
        LoraConfig(
            r=int(train_config["rank"]),
            lora_alpha=int(train_config["lora_alpha"]),
            lora_dropout=float(train_config.get("lora_dropout", 0.0)),
            init_lora_weights="gaussian",
            target_modules=list(train_config["target_modules"]),
        )
    )
    if bool(train_config.get("layerwise_casting", False)):
        transformer.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn,
            compute_dtype=weight_dtype,
        )
    if args.resume_lora:
        state = Krea2Pipeline.lora_state_dict(args.resume_lora)
        if isinstance(state, tuple):
            state = state[0]
        Krea2Pipeline.load_lora_into_transformer(state, transformer=transformer)

    trainable_parameters = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    total_count = sum(parameter.numel() for parameter in transformer.parameters())
    if accelerator.is_main_process:
        print(f"trainable parameters: {trainable_count:,} / {total_count:,} ({100 * trainable_count / total_count:.4f}%)")
        print("conditional sequence: [noisy target tokens | clean source tokens]")

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

    first_target, first_source, first_prompt_id = dataset[0]
    if first_target.shape != first_source.shape:
        raise ValueError(f"Target/source latent shapes differ: {first_target.shape} vs {first_source.shape}")
    target_sequence_length, channels = first_target.shape
    unwrapped_config = accelerator.unwrap_model(transformer).config
    if channels != int(unwrapped_config.in_channels):
        raise ValueError(f"Latent channels {channels} != transformer in_channels {unwrapped_config.in_channels}")
    height = int(rows[0]["height"])
    width = int(rows[0]["width"])
    grid_height = height // (int(model_config.get("vae_scale_factor", 8)) * int(model_config.get("patch_size", 2)))
    grid_width = width // (int(model_config.get("vae_scale_factor", 8)) * int(model_config.get("patch_size", 2)))
    if grid_height * grid_width != target_sequence_length:
        raise ValueError(
            f"Resolution implies {grid_height * grid_width} target tokens, cache has {target_sequence_length}"
        )
    prompt_length = prompt_cache[prompt_cache_key("embeds", first_prompt_id)].shape[1]
    position_ids = prepare_paired_position_ids(prompt_length, grid_height, grid_width, accelerator.device)
    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
    schedule_sigmas = noise_scheduler.sigmas.to(accelerator.device)
    weighting_scheme = str(train_config.get("weighting_scheme", "logit_normal"))
    reference_dropout = float(train_config.get("reference_dropout", 0.0))
    global_step = 0
    progress = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process, desc="Krea2 paired LoRA")
    metadata = {
        "base_model": str(model_path),
        "task": "arbitrary reference image to strict Minecraft front/back preview",
        "conditioning_schema": "target_then_reference_latents",
        "reference_position_axis": "t=1",
        "rank": str(train_config["rank"]),
        "resolution": f"{width}x{height}",
        "layerwise_casting": str(bool(train_config.get("layerwise_casting", False))),
    }

    transformer.train()
    while global_step < max_train_steps:
        for clean_target, clean_source, prompt_ids in loader:
            with accelerator.accumulate(transformer):
                clean_target = clean_target.to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                clean_source = clean_source.to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                if reference_dropout > 0:
                    keep = (torch.rand(clean_source.shape[0], device=accelerator.device) >= reference_dropout).to(
                        clean_source.dtype
                    ).view(-1, 1, 1)
                    clean_source = clean_source * keep
                noise = torch.randn_like(clean_target)
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=weighting_scheme,
                    batch_size=clean_target.shape[0],
                    logit_mean=float(train_config.get("logit_mean", 0.0)),
                    logit_std=float(train_config.get("logit_std", 1.0)),
                    mode_scale=float(train_config.get("mode_scale", 1.29)),
                    device=accelerator.device,
                )
                indices = (u * noise_scheduler.config.num_train_timesteps).long().clamp_(
                    0, noise_scheduler.config.num_train_timesteps - 1
                )
                timesteps = schedule_timesteps[indices]
                sigmas = schedule_sigmas[indices].to(dtype=clean_target.dtype).view(-1, 1, 1)
                noisy_target = (1.0 - sigmas) * clean_target + sigmas * noise
                model_input = torch.cat([noisy_target, clean_source], dim=1)
                embeds = torch.cat(
                    [prompt_cache[prompt_cache_key("embeds", prompt_id)] for prompt_id in prompt_ids], dim=0
                ).to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                masks = torch.cat(
                    [prompt_cache[prompt_cache_key("mask", prompt_id)] for prompt_id in prompt_ids], dim=0
                ).to(accelerator.device, non_blocking=True)
                full_prediction = transformer(
                    hidden_states=model_input,
                    encoder_hidden_states=embeds,
                    timestep=(timesteps / noise_scheduler.config.num_train_timesteps).to(weight_dtype),
                    position_ids=position_ids,
                    encoder_attention_mask=masks,
                    return_dict=False,
                )[0]
                target_prediction = full_prediction[:, :target_sequence_length]
                flow_target = noise - clean_target
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=weighting_scheme, sigmas=sigmas)
                loss = (
                    weighting.float()
                    * F.mse_loss(target_prediction.float(), flow_target.float(), reduction="none")
                ).reshape(clean_target.shape[0], -1).mean(dim=1).mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_parameters,
                        float(train_config.get("max_grad_norm", 1.0)),
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=f"{loss.detach().item():.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")
                save_every = int(train_config.get("save_every", 500))
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
                "conditioning_schema": "target_then_reference_latents",
                "final_lora": str(output_dir / "final"),
            },
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
