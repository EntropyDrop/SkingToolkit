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
from diffusers.pipelines.krea2.pipeline_krea2 import calculate_shift
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from checkpoint_preview import CheckpointPreviewer, checkpoint_test_image_paths
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
        description="Train Krea2 LoRA with paired source-to-edited rectified flow."
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


def save_checkpoint_previews(
    previewer: CheckpointPreviewer,
    transformer: torch.nn.Module,
    checkpoint_dir: Path,
    checkpoint: int | str,
    source_images: list[Path],
    prompt_id: str,
    conditioning_schema: str,
) -> None:
    generated = previewer.save(transformer, checkpoint_dir / "tests")
    write_json(
        checkpoint_dir / "tests" / "preview.json",
        {
            "checkpoint": checkpoint,
            "source_images": [str(path) for path in source_images],
            "generated_images": [str(path) for path in generated],
            "steps": previewer.steps,
            "guidance_scale": 0.0,
            "prompt_id": prompt_id,
            "conditioning_schema": conditioning_schema,
        },
    )
    print(f"checkpoint previews saved: {checkpoint_dir / 'tests'}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_config = config["model"]
    data_config = config["data"]
    train_config = config["training"]
    conditioning_mode = str(train_config.get("conditioning_mode", "source_bridge"))
    if conditioning_mode not in {"source_bridge", "target_reference_concat"}:
        raise ValueError(f"Unsupported conditioning_mode: {conditioning_mode}")
    checkpoint_preview_config = config.get("checkpoint_preview", {})
    checkpoint_test_paths = checkpoint_test_image_paths(checkpoint_preview_config.get("test_images", []))
    if checkpoint_test_paths and conditioning_mode != "source_bridge":
        raise ValueError("Checkpoint previews require training.conditioning_mode=source_bridge")
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
            target_modules=list(train_config["target_modules"]),
        )
    )
    if bool(train_config.get("layerwise_casting", False)):
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
        if conditioning_mode == "source_bridge":
            print("conditioning: paired rectified flow source_latents -> target_latents")
        else:
            print("conditioning: legacy [noisy target tokens | clean source tokens]")

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
    if conditioning_mode == "source_bridge":
        position_ids = Krea2Pipeline.prepare_position_ids(
            prompt_length,
            grid_height,
            grid_width,
            accelerator.device,
        )
    else:
        position_ids = prepare_paired_position_ids(prompt_length, grid_height, grid_width, accelerator.device)
    image_seq_len = target_sequence_length
    mu = calculate_shift(
        image_seq_len,
        noise_scheduler.config.get("base_image_seq_len", 256),
        noise_scheduler.config.get("max_image_seq_len", 6400),
        noise_scheduler.config.get("base_shift", 0.5),
        noise_scheduler.config.get("max_shift", 1.15),
    )
    noise_scheduler.set_timesteps(
        noise_scheduler.config.num_train_timesteps,
        device=accelerator.device,
        mu=mu,
    )
    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
    schedule_sigmas = noise_scheduler.sigmas.to(accelerator.device)
    weighting_scheme = str(train_config.get("weighting_scheme", "logit_normal"))
    reference_dropout = float(train_config.get("reference_dropout", 0.0))
    source_noise_strength = float(train_config.get("source_noise_strength", 0.0))
    if not 0.0 <= source_noise_strength <= 1.0:
        raise ValueError("source_noise_strength must be between 0 and 1")
    checkpoint_previewer = None
    if checkpoint_test_paths and accelerator.is_main_process:
        inference_config = config["inference"]
        checkpoint_previewer = CheckpointPreviewer(
            image_paths=checkpoint_test_paths,
            model_path=model_path,
            prompt_embeds=prompt_cache[prompt_cache_key("embeds", first_prompt_id)],
            prompt_mask=prompt_cache[prompt_cache_key("mask", first_prompt_id)],
            device=accelerator.device,
            dtype=weight_dtype,
            width=int(inference_config.get("width", width)),
            height=int(inference_config.get("height", height)),
            vae_scale_factor=int(model_config.get("vae_scale_factor", 8)),
            patch_size=int(model_config.get("patch_size", 2)),
            steps=int(inference_config.get("steps", 28)),
            seed=int(inference_config.get("seed", train_config["seed"])),
            source_noise_strength=float(inference_config.get("source_noise_strength", 0.0)),
        )
        print(
            f"checkpoint previews: {len(checkpoint_test_paths)} image(s), "
            f"{checkpoint_previewer.steps} steps, positive prompt without CFG"
        )
    global_step = 0
    progress = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process, desc="Krea2 paired LoRA")
    metadata = {
        "base_model": str(model_path),
        "task": "arbitrary reference image to strict Minecraft front/back preview",
        "conditioning_schema": conditioning_mode,
        "source_noise_strength": str(source_noise_strength),
        "rank": str(lora_rank),
        "lora_alpha": str(lora_alpha),
        "resolution": f"{width}x{height}",
        "layerwise_casting": str(bool(train_config.get("layerwise_casting", False))),
    }

    if checkpoint_test_paths:
        accelerator.wait_for_everyone()
        checkpoint_dir = output_dir / "checkpoint-0"
        save_lora(transformer, accelerator, checkpoint_dir, metadata)
        if accelerator.is_main_process:
            if checkpoint_previewer is None:
                raise RuntimeError("Main process did not initialize the checkpoint previewer")
            save_checkpoint_previews(
                checkpoint_previewer,
                accelerator.unwrap_model(transformer),
                checkpoint_dir,
                0,
                checkpoint_test_paths,
                first_prompt_id,
                conditioning_mode,
            )
        accelerator.wait_for_everyone()

    transformer.train()
    while global_step < max_train_steps:
        for clean_target, clean_source, prompt_ids in loader:
            with accelerator.accumulate(transformer):
                clean_target = clean_target.to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                clean_source = clean_source.to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                if conditioning_mode == "target_reference_concat" and reference_dropout > 0:
                    keep = (torch.rand(clean_source.shape[0], device=accelerator.device) >= reference_dropout).to(
                        clean_source.dtype
                    ).view(-1, 1, 1)
                    clean_source = clean_source * keep
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
                if conditioning_mode == "source_bridge":
                    source_start = clean_source
                    if source_noise_strength > 0:
                        source_noise = torch.randn_like(clean_source)
                        source_start = (
                            (1.0 - source_noise_strength) * clean_source
                            + source_noise_strength * source_noise
                        )
                    model_input = (1.0 - sigmas) * clean_target + sigmas * source_start
                    flow_target = source_start - clean_target
                else:
                    noise = torch.randn_like(clean_target)
                    noisy_target = (1.0 - sigmas) * clean_target + sigmas * noise
                    model_input = torch.cat([noisy_target, clean_source], dim=1)
                    flow_target = noise - clean_target
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
                target_prediction = (
                    full_prediction
                    if conditioning_mode == "source_bridge"
                    else full_prediction[:, :target_sequence_length]
                )
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
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    save_lora(transformer, accelerator, checkpoint_dir, metadata)
                    if accelerator.is_main_process and checkpoint_previewer is not None:
                        save_checkpoint_previews(
                            checkpoint_previewer,
                            accelerator.unwrap_model(transformer),
                            checkpoint_dir,
                            global_step,
                            checkpoint_test_paths,
                            first_prompt_id,
                            conditioning_mode,
                        )
                    accelerator.wait_for_everyone()
            if global_step >= max_train_steps:
                break

    accelerator.wait_for_everyone()
    final_dir = output_dir / "final"
    save_lora(transformer, accelerator, final_dir, metadata)
    if accelerator.is_main_process and checkpoint_previewer is not None:
        save_checkpoint_previews(
            checkpoint_previewer,
            accelerator.unwrap_model(transformer),
            final_dir,
            "final",
            checkpoint_test_paths,
            first_prompt_id,
            conditioning_mode,
        )
    if accelerator.is_main_process:
        write_json(
            output_dir / "training_state.json",
            {
                "global_step": global_step,
                "trainable_parameters": trainable_count,
                "dataset_items": len(dataset),
                "conditioning_schema": conditioning_mode,
                "final_lora": str(output_dir / "final"),
            },
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
